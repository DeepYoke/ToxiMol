"""
Data models for toxicity evaluation.

This module defines the data structures used in the toxicity evaluation process.
"""
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

@dataclass
class ToxicityEndpointResult:
    """Toxicity endpoint prediction result."""
    endpoint: str  # Endpoint name
    value: str  # Prediction value (A/B or numeric for LD50)
    probability: float  # Safety probability (probability of being non-toxic)

@dataclass
class ToxicityDelta:
    """Change in toxicity between original and modified molecule."""
    endpoint: str  # Endpoint name
    original: str  # Original prediction value
    modified: str  # Modified prediction value
    original_prob: float  # Original safety probability
    modified_prob: float  # Modified safety probability
    changed: bool  # Whether the prediction changed
    improved: bool  # Whether the toxicity improved

@dataclass
class EvaluationResult:
    """Evaluation result."""
    molecule_id: int  # Molecule ID
    task: str  # Task name
    original_smiles: str  # Original SMILES
    modified_smiles: str  # Modified SMILES
    model: str  # Model name
    success: bool  # Whether the modification was successful
    toxicity_endpoints: Dict[str, ToxicityEndpointResult]  # Endpoint results
    toxicity_deltas: List[ToxicityDelta]  # Toxicity changes
    details: Dict[str, Any]  # Detailed evaluation metrics
    message: str  # Feedback message
    
@dataclass
class PropertyResult:
    """Molecular property calculation result."""
    qed: float  # Quantitative Estimate of Drug-likeness
    sas_score: float  # Synthetic Accessibility Score
    lipinski_violations: int  # Number of Lipinski rule violations
    logp: float  # Octanol-water partition coefficient
    
@dataclass
class RepairResults:
    """Results from the repair evaluation."""
    toxicity_endpoints: Dict[str, ToxicityEndpointResult]  # Endpoint results
    toxicity_improved: bool  # Whether toxicity was improved overall
    qed: float  # Drug-likeness
    sas_score: float  # Synthetic accessibility
    lipinski_violations: int  # Number of Lipinski rule violations
    similarity: float  # Similarity to original molecule
    passed_repair: bool  # Whether passed repair criteria
    fails: List[str]  # List of failure reasons 