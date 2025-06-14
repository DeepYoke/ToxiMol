from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

@dataclass
class ToxicityEndpointResult:
    """Toxicity endpoint prediction result."""
    endpoint: str  
    value: str  
    probability: float

@dataclass
class ToxicityDelta:
    """Change in toxicity between original and modified molecule."""
    endpoint: str 
    original: str  
    modified: str  
    original_prob: float  
    modified_prob: float  
    changed: bool  
    improved: bool 

@dataclass
class EvaluationResult:
    """Evaluation result."""
    molecule_id: int  
    task: str  
    original_smiles: str  
    modified_smiles: str  
    model: str  
    success: bool 
    toxicity_endpoints: Dict[str, ToxicityEndpointResult]  
    toxicity_deltas: List[ToxicityDelta]  
    details: Dict[str, Any]  
    message: str 
    
@dataclass
class PropertyResult:
    """Molecular property calculation result."""
    qed: float  
    sas_score: float  
    lipinski_violations: int  
    logp: float  
    
@dataclass
class RepairResults:
    """Results from the repair evaluation."""
    toxicity_endpoints: Dict[str, ToxicityEndpointResult] 
    toxicity_improved: bool  
    qed: float  
    sas_score: float  
    lipinski_violations: int 
    similarity: float  
    passed_repair: bool 
    fails: List[str] 