"""
Molecule utility functions.

This module provides utilities for molecule handling, property calculation, and validation.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, AllChem, DataStructs
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from typing import Dict, List, Optional, Union, Any, Tuple
import os
from pathlib import Path

# Import the original SA scorer
from sascorer import calculateScore as calculate_sa_score_original

# TxGemma imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re

# Singleton pattern for TxGemma model to avoid loading multiple times
_txgemma_model = None
_txgemma_tokenizer = None
_tdc_prompts = None

def load_txgemma_model():
    """
    Load the TxGemma model for toxicity prediction.
    Uses singleton pattern to avoid loading multiple times.
    """
    global _txgemma_model, _txgemma_tokenizer, _tdc_prompts
    
    if _txgemma_model is None:
        model_name = "google/txgemma-9b-predict"  # 可根据需要更改为2b或27b
        _txgemma_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _txgemma_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Load TDC prompts
        prompts_file = "evaluation/tdc_prompts.json"
        if os.path.exists(prompts_file):
            with open(prompts_file, "r") as f:
                _tdc_prompts = json.load(f)
        else:
            raise FileNotFoundError(f"TDC prompts file not found: {prompts_file}")
                
        print("TxGemma model loaded successfully")

def _extract_label(response: str) -> Optional[str]:
    """
    Extract classification label (A or B) from model response.
    
    Args:
        response: Model response text
        
    Returns:
        str or None: Extracted label ('A' or 'B') or None if not found
    """
    # 清理输入文本并查找第一个出现的A或B
    match = re.search(r"[AB]", response.upper())
    return match.group(0) if match else None

def predict_toxicity(smiles: str, endpoint: str) -> Tuple[str, float]:
    """
    Predict toxicity for a molecule using TxGemma model.
    使用生成+解析文本的方法，而不是logits。
    
    Args:
        smiles: SMILES string of the molecule
        endpoint: Toxicity endpoint to predict
        
    Returns:
        Tuple[str, float]: Prediction value and safety probability
    """
    global _txgemma_model, _txgemma_tokenizer, _tdc_prompts
    
    # Load model if not already loaded
    if _txgemma_model is None:
        load_txgemma_model()
        
    # Check if prompts are available
    if not _tdc_prompts or endpoint not in _tdc_prompts:
        raise ValueError(f"Prompt not found for endpoint {endpoint}")
        
    # Generate prompt
    prompt = _tdc_prompts[endpoint].replace("{Drug SMILES}", smiles)
    
    # Generate input tensors
    inputs = _txgemma_tokenizer(prompt, return_tensors="pt").to(_txgemma_model.device)

    # Generate text output
    max_new_tokens = 6 if endpoint == "LD50_Zhu" else 2  # 较短的token数，足够获取首个A/B
    gen_out = _txgemma_model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens, 
        do_sample=False
    )
    new_tokens = gen_out[0][inputs["input_ids"].shape[-1]:]
    response = _txgemma_tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Process based on endpoint type
    if endpoint != "LD50_Zhu":
        # 从生成的文本中提取第一个A或B
        label = _extract_label(response)
        
        # 如果无法找到标签，默认为B（有毒）
        if label is None:
            label = "B"
            safety_probability = 0.0
        else:
            # 安全概率：A=1.0, B=0.0
            safety_probability = 1.0 if label == "A" else 0.0
        
        value = label
    else:
        # LD50 endpoint: 解析数值
        ld50_value = parse_ld50_value(response)
        safety_probability = min(1.0, max(0.0, ld50_value / 1000.0))
        value = str(ld50_value)
        
    return value, safety_probability

def parse_ld50_value(response: str) -> int:
    """
    Extract LD50 value from model response.
    
    Args:
        response: Model response text
        
    Returns:
        int: Parsed LD50 value (0-1000)
    """
    number_matches = re.findall(r"[0-9]+(?:\.[0-9]+)?", response)
    if number_matches:
        try:
            value = int(float(number_matches[0]))
            return max(0, min(1000, value))
        except (ValueError, IndexError):
            return 0
    return 0

def validate_smiles(smiles: str) -> bool:
    """
    Validate if a SMILES string is valid.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        bool: True if SMILES is valid, False otherwise
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def calculate_properties(smiles: str) -> Dict[str, float]:
    """
    Calculate molecular properties from SMILES.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dict[str, float]: Dictionary of property values
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "qed": 0.0,
            "sas_score": 10.0,
            "logp": 0.0,
            "molecular_weight": 0.0,
            "h_donors": 0,
            "h_acceptors": 0,
            "rotatable_bonds": 0,
            "heavy_atoms": 0,
            "rings": 0,
            "lipinski_violations": 4,
        }
    
    # Calculate Lipinski violations
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Lipinski.NumHDonors(mol)
    h_acceptors = Lipinski.NumHAcceptors(mol)
    
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if h_donors > 5: violations += 1
    if h_acceptors > 10: violations += 1
    
    # Calculate other properties
    qed_value = QED.qed(mol)
    
    # Use the original SA score calculation instead of the simplified version
    sa_score = calculate_sa_score_original(mol)
    
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    rings = mol.GetRingInfo().NumRings()
    
    return {
        "qed": qed_value,
        "sas_score": sa_score,
        "logp": logp,
        "molecular_weight": mw,
        "h_donors": h_donors,
        "h_acceptors": h_acceptors,
        "rotatable_bonds": rotatable_bonds,
        "heavy_atoms": heavy_atoms,
        "rings": rings,
        "lipinski_violations": violations,
    }

def calculate_similarity(smiles1: str, smiles2: str) -> float:
    """
    Calculate similarity between two molecules (Tanimoto similarity).
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        
    Returns:
        float: Similarity score (0-1)
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
    fp1 = Chem.RDKFingerprint(mol1)
    fp2 = Chem.RDKFingerprint(mol2)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2) 