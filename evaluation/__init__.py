"""
TxGemma-based toxicity repair evaluation package.

This package provides tools for evaluating the results of toxicity repair experiments
using TxGemma for toxicity prediction.
"""

from experiments.evaluation.evaluation_models import (
    EvaluationResult,
    PropertyResult,
    RepairResults
)

from experiments.evaluation.task_mappings import (
    get_task_mapping
)

from experiments.evaluation.molecule_utils import (
    validate_smiles,
    calculate_properties,
    calculate_similarity
)

from experiments.evaluation.result_evaluator import (
    ResultEvaluator,
    analyze_experiment_results
)

__all__ = [
    # Evaluation models
    "EvaluationResult",
    "PropertyResult",
    "RepairResults",
    
    # Task mappings
    "get_task_mapping",
    
    # Molecule utilities
    "validate_smiles",
    "calculate_properties",
    "calculate_similarity",
    
    # Result evaluator
    "ResultEvaluator",
    "analyze_experiment_results"
] 