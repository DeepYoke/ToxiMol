"""
Toxicity repair result evaluator.

This module evaluates the success of molecular structure modifications for toxicity reduction.
"""
import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import re

from evaluation_models import (
    EvaluationResult,
    RepairResults,
    ToxicityEndpointResult,
    ToxicityDelta
)
from task_mappings import get_task_mapping
from molecule_utils import (
    validate_smiles,
    calculate_properties,
    calculate_similarity,
    predict_toxicity
)

class ResultEvaluator:
    """
    Evaluator for toxicity repair experiments.
    """
    
    def __init__(self, results_dir: str = "results/gpt", output_dir: str = "experiments/eval_results"):
        """
        Initialize the evaluator.
        
        Args:
            results_dir: Directory containing experiment results
            output_dir: Directory to save evaluation results
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        
    def evaluate_single_result(
        self, 
        result_file: Union[str, Path],
        full_evaluation: bool = False
    ) -> List[EvaluationResult]:
        """
        Evaluate a single result file from the experiment.
        
        Args:
            result_file: Path to the result JSON file
            full_evaluation: Whether to perform a full evaluation including property calculations
            
        Returns:
            List[EvaluationResult]: List of evaluation results for all modified molecules
        """
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        model = data.get('model', '')
        task = data.get('task', '')
        molecule_id = data.get('molecule_id', 0)
        original_smiles = data.get('original_smiles', '')
        raw_response = data.get('raw_response', '')
        
        modified_smiles_list = data.get('modified_smiles', [])
        
        modified_smiles_list = modified_smiles_list[:3]
        
        if not modified_smiles_list:
            details = {
                'validation': {
                    'valid_smiles': False,
                    'error': "No modified SMILES found"
                },
                'toxicity': {
                    'improved': False
                },
                'properties': {}
            }
            
            return [EvaluationResult(
                molecule_id=molecule_id,
                task=task,
                original_smiles=original_smiles,
                modified_smiles="",
                model=model,
                success=False,
                toxicity_endpoints={},
                toxicity_deltas=[],
                details=details,
                message="Failed: No modified SMILES provided"
            )]
        
        evaluation_results = []
        valid_smiles_found = False
        any_repair_success = False
        
        for modified_smiles in modified_smiles_list:
            details = {
                'validation': {
                    'valid_smiles': False,
                    'error': None
                },
                'toxicity': {
                    'improved': False
                },
                'properties': {}
            }
            
            if not modified_smiles:
                details['validation']['valid_smiles'] = False
                details['validation']['error'] = "Empty SMILES string"
                
                evaluation_results.append(EvaluationResult(
                    molecule_id=molecule_id,
                    task=task,
                    original_smiles=original_smiles,
                    modified_smiles=modified_smiles,
                    model=model,
                    success=False,
                    toxicity_endpoints={},
                    toxicity_deltas=[],
                    details=details,
                    message="Failed: Empty SMILES string"
                ))
                continue
            
            if not validate_smiles(modified_smiles):
                details['validation']['valid_smiles'] = False
                details['validation']['error'] = "Invalid SMILES string"
                
                evaluation_results.append(EvaluationResult(
                    molecule_id=molecule_id,
                    task=task,
                    original_smiles=original_smiles,
                    modified_smiles=modified_smiles,
                    model=model,
                    success=False,
                    toxicity_endpoints={},
                    toxicity_deltas=[],
                    details=details,
                    message="Failed: Invalid SMILES structure"
                ))
                continue
            
            valid_smiles_found = True
            details['validation']['valid_smiles'] = True
            
            if full_evaluation:
                repair_results = self._evaluate_repair_criteria(
                    task, 
                    original_smiles, 
                    modified_smiles
                )
                
                details['properties'] = {
                    'qed': repair_results.qed,
                    'sas_score': repair_results.sas_score,
                    'lipinski_violations': repair_results.lipinski_violations,
                    'similarity': repair_results.similarity
                }
                
                success = repair_results.passed_repair
                    
                if success:
                    any_repair_success = True
                    
                message = "Success: Molecule meets all repair criteria" if success else f"Failed: {', '.join(repair_results.fails)}"
                
                toxicity_endpoints = repair_results.toxicity_endpoints
                
                toxicity_deltas = self._calculate_toxicity_deltas(repair_results.toxicity_endpoints)
                
                details['toxicity']['endpoints'] = {
                    endpoint: {
                        'value': endpoint_result.value, 
                        'probability': endpoint_result.probability
                    } for endpoint, endpoint_result in toxicity_endpoints.items()
                }
                details['toxicity']['improved'] = repair_results.toxicity_improved
            else:   
                success = True
                message = "Basic validation passed"
                toxicity_endpoints = {}
                toxicity_deltas = []
            
            evaluation_results.append(EvaluationResult(
                molecule_id=molecule_id,
                task=task,
                original_smiles=original_smiles,
                modified_smiles=modified_smiles,
                model=model,
                success=success,
                toxicity_endpoints=toxicity_endpoints,
                toxicity_deltas=toxicity_deltas,
                details=details,
                message=message
            ))
        
        if not valid_smiles_found and not evaluation_results:
            details = {
                'validation': {
                    'valid_smiles': False,
                    'error': "No valid SMILES found among modifications"
                },
                'toxicity': {
                    'improved': False
                },
                'properties': {}
            }
            
            return [EvaluationResult(
                molecule_id=molecule_id,
                task=task,
                original_smiles=original_smiles,
                modified_smiles="",
                model=model,
                success=False,
                toxicity_endpoints={},
                toxicity_deltas=[],
                details=details,
                message="Failed: No valid SMILES found among modifications"
            )]
        
        return evaluation_results
    
    def evaluate_task_results(
        self, 
        model: str,
        task: str,
        full_evaluation: bool = False
    ) -> Tuple[List[EvaluationResult], Dict[str, Any]]:
        """
        Evaluate all results for a specific task and model.
        
        Args:
            model: Model name
            task: Task name
            full_evaluation: Whether to perform full evaluation
            
        Returns:
            Tuple[List[EvaluationResult], Dict[str, Any]]: List of evaluation results and summary
        """
        task_dir = self.results_dir / model / task
        consolidated_file = task_dir / f"{task}_results.json"
        
        if consolidated_file.exists():
            print(f"Found consolidated results file: {consolidated_file}")
            evaluation_results = self._process_consolidated_results(consolidated_file, full_evaluation)
        else:
            result_files = list(task_dir.glob(f"{task}_*.json"))
            
            result_files = [f for f in result_files if "summary" not in f.name and "error" not in f.name]
            
            evaluation_results = []
            for result_file in result_files:
                try:
                    eval_result = self.evaluate_single_result(result_file, full_evaluation)
                    evaluation_results.extend(eval_result)
                except Exception as e:
                    print(f"Error evaluating {result_file}: {e}")
        
        results_by_molecule = {}
        for result in evaluation_results:
            if result.molecule_id not in results_by_molecule:
                results_by_molecule[result.molecule_id] = []
            results_by_molecule[result.molecule_id].append(result)
        
        original_molecule_count = len(results_by_molecule)
        
        molecules_per_original = 3 
        total_modified_molecules = original_molecule_count * molecules_per_original
        
        valid_smiles_count = sum(1 for r in evaluation_results if r.details['validation']['valid_smiles'])
        
        success_count = 0
        for molecule_id, results in results_by_molecule.items():
            if any(r.success for r in results):
                success_count += 1
        
        summary = {
            'task': task,
            'model': model,
            'original_molecule_count': original_molecule_count,
            'total_molecules': total_modified_molecules,
            'valid_smiles_count': valid_smiles_count,
            'success_count': success_count,
            'valid_percentage': valid_smiles_count / total_modified_molecules * 100 if total_modified_molecules > 0 else 0,
            'success_percentage': success_count / original_molecule_count * 100 if original_molecule_count > 0 else 0
        }
        
        return evaluation_results, summary
    
    def _process_consolidated_results(
        self,
        consolidated_file: Path,
        full_evaluation: bool = False
    ) -> List[EvaluationResult]:
        """
        Process a consolidated results file containing multiple molecule results.
        
        Args:
            consolidated_file: Path to the consolidated results JSON file
            full_evaluation: Whether to perform full evaluation
            
        Returns:
            List[EvaluationResult]: List of evaluation results
        """     
        with open(consolidated_file, 'r') as f:
            data = json.load(f)
        
        model = data.get('model', '')
        task_name = data.get('task_name', '')
        
        evaluation_results = []
        for result in data.get('results', []):
            molecule_id = result.get('molecule_id', 0)
            task = result.get('task', task_name)
            original_smiles = result.get('original_smiles', '')
            raw_response = result.get('raw_response', '')
            
            modified_smiles_list = result.get('modified_smiles', [])
            
            modified_smiles_list = modified_smiles_list[:3]
            
            if not modified_smiles_list:
                details = {
                    'validation': {
                        'valid_smiles': False,
                        'error': "No modified SMILES found"
                    },
                    'toxicity': {
                        'improved': False
                    },
                    'properties': {}
                }
                
                evaluation_results.append(EvaluationResult(
                    molecule_id=molecule_id,
                    task=task,
                    original_smiles=original_smiles,
                    modified_smiles="",
                    model=model,
                    success=False,
                    toxicity_endpoints={},
                    toxicity_deltas=[],
                    details=details,
                    message="Failed: No modified SMILES provided"
                ))
                continue
            
            valid_smiles_found = False
            any_repair_success = False
            molecule_results = []
            
            for modified_smiles in modified_smiles_list:
                details = {
                    'validation': {
                        'valid_smiles': False,
                        'error': None
                    },
                    'toxicity': {
                        'improved': False
                    },
                    'properties': {}
                }
                
                if not modified_smiles:
                    details['validation']['valid_smiles'] = False
                    details['validation']['error'] = "Empty SMILES string"
                    
                    molecule_results.append(EvaluationResult(
                        molecule_id=molecule_id,
                        task=task,
                        original_smiles=original_smiles,
                        modified_smiles=modified_smiles,
                        model=model,
                        success=False,
                        toxicity_endpoints={},
                        toxicity_deltas=[],
                        details=details,
                        message="Failed: Empty SMILES string"
                    ))
                    continue
                
                if not validate_smiles(modified_smiles):
                    details['validation']['valid_smiles'] = False
                    details['validation']['error'] = "Invalid SMILES string"
                    
                    molecule_results.append(EvaluationResult(
                        molecule_id=molecule_id,
                        task=task,
                        original_smiles=original_smiles,
                        modified_smiles=modified_smiles,
                        model=model,
                        success=False,
                        toxicity_endpoints={},
                        toxicity_deltas=[],
                        details=details,
                        message="Failed: Invalid SMILES structure"
                    ))
                    continue
                
                valid_smiles_found = True
                details['validation']['valid_smiles'] = True
                
                if full_evaluation:
                    repair_results = self._evaluate_repair_criteria(
                        task, 
                        original_smiles, 
                        modified_smiles
                    )
                    
                    details['properties'] = {
                        'qed': repair_results.qed,
                        'sas_score': repair_results.sas_score,
                        'lipinski_violations': repair_results.lipinski_violations,
                        'similarity': repair_results.similarity
                    }
                    
                    success = repair_results.passed_repair

                    if success:
                        any_repair_success = True
                    
                    message = "Success: Molecule meets all repair criteria" if success else f"Failed: {', '.join(repair_results.fails)}"
                    
                    toxicity_endpoints = repair_results.toxicity_endpoints
                    
                    toxicity_deltas = self._calculate_toxicity_deltas(repair_results.toxicity_endpoints)
                    
                    details['toxicity']['endpoints'] = {
                        endpoint: {
                                'value': endpoint_result.value, 
                                'probability': endpoint_result.probability
                            } for endpoint, endpoint_result in toxicity_endpoints.items()
                    }
                    details['toxicity']['improved'] = repair_results.toxicity_improved
                else:
                    success = True
                    message = "Basic validation passed"
                    toxicity_endpoints = {}
                    toxicity_deltas = []
                
                molecule_results.append(EvaluationResult(
                    molecule_id=molecule_id,
                    task=task,
                    original_smiles=original_smiles,
                    modified_smiles=modified_smiles,
                    model=model,
                    success=success,
                    toxicity_endpoints=toxicity_endpoints,
                    toxicity_deltas=toxicity_deltas,
                    details=details,
                    message=message
                ))
            
            evaluation_results.extend(molecule_results)
            
            if not valid_smiles_found:
                details = {
                    'validation': {
                        'valid_smiles': False,
                        'error': "No valid SMILES found among modifications"
                    },
                    'toxicity': {
                        'improved': False
                    },
                    'properties': {}
                }
                
                evaluation_results.append(EvaluationResult(
                    molecule_id=molecule_id,
                    task=task,
                    original_smiles=original_smiles,
                    modified_smiles="",
                    model=model,
                    success=False,
                    toxicity_endpoints={},
                    toxicity_deltas=[],
                    details=details,
                    message="Failed: No valid SMILES found among modifications"
                ))
        
        return evaluation_results
    
    def evaluate_all_results(
        self, 
        model: str,
        full_evaluation: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all results for a specific model across all tasks.
        
        Args:
            model: Model name
            full_evaluation: Whether to perform full evaluation
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping task names to their summaries
        """
        model_dir = self.results_dir / model
        task_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        
        all_summaries = {}
        for task_dir in task_dirs:
            task = task_dir.name
            results, summary = self.evaluate_task_results(model, task, full_evaluation)
            all_summaries[task] = summary
            
            if full_evaluation:
                results_by_molecule = {}
                for result in results:
                    if result.molecule_id not in results_by_molecule:
                        results_by_molecule[result.molecule_id] = []
                    results_by_molecule[result.molecule_id].append(result)
                
                toxicity_improved_count = 0
                for molecule_id, molecule_results in results_by_molecule.items():
                    if any(r.details.get('toxicity', {}).get('improved', False) for r in molecule_results):
                        toxicity_improved_count += 1
                
                summary['toxicity_improved_count'] = toxicity_improved_count
                summary['toxicity_improved_percentage'] = toxicity_improved_count / len(results_by_molecule) * 100 if results_by_molecule else 0
        
        original_molecule_count = sum(summary['original_molecule_count'] for summary in all_summaries.values())
        total_molecules = sum(summary['total_molecules'] for summary in all_summaries.values())
        valid_smiles_count = sum(summary['valid_smiles_count'] for summary in all_summaries.values())
        success_count = sum(summary['success_count'] for summary in all_summaries.values())
        
        overall = {
            'model': model,
            'original_molecule_count': original_molecule_count,
            'total_molecules': total_molecules,
            'valid_smiles_count': valid_smiles_count,
            'success_count': success_count,
            'valid_percentage': valid_smiles_count / total_molecules * 100 if total_molecules > 0 else 0,
            'success_percentage': success_count / original_molecule_count * 100 if original_molecule_count > 0 else 0,
            'tasks_completed': len(all_summaries)
        }
        
        if full_evaluation and all('toxicity_improved_count' in summary for summary in all_summaries.values()):
            toxicity_improved_count = sum(summary['toxicity_improved_count'] for summary in all_summaries.values())
            overall['toxicity_improved_count'] = toxicity_improved_count
            overall['toxicity_improved_percentage'] = toxicity_improved_count / original_molecule_count * 100 if original_molecule_count > 0 else 0
        
        all_summaries['overall'] = overall
        
        return all_summaries
    
    def save_evaluation_results(
        self, 
        model: str,
        all_summaries: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Save evaluation results to files.
        
        Args:
            model: Model name
            all_summaries: Dictionary of evaluation summaries
            
        Returns:
            str: Path to the summary file
        """ 
        base_output_dir = self.output_dir / model
        base_output_dir.mkdir(exist_ok=True, parents=True)
        
        is_single_task = len(all_summaries) == 1 and "overall" not in all_summaries
        
        if is_single_task:
            task_name = list(all_summaries.keys())[0]
            output_dir = base_output_dir / f"task_{task_name}"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            summary_file = output_dir / f"{task_name}_evaluation_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(all_summaries, f, indent=2)
            
            summary_data = []
            summary = all_summaries[task_name]
            row = {
                "Task": task_name,
                "Original Molecules": summary.get("original_molecule_count", summary.get("total_molecules", 0)),
                "Total Modified Molecules": summary["total_molecules"],
                "Valid SMILES": summary["valid_smiles_count"],
                "Successful Repairs": summary["success_count"],
                "Valid %": f"{summary['valid_percentage']:.2f}%",
                "Success %": f"{summary['success_percentage']:.2f}%",
            }
            
            if 'toxicity_improved_count' in summary:
                row["Toxicity Improved"] = summary["toxicity_improved_count"]
                row["Toxicity Improved %"] = f"{summary['toxicity_improved_percentage']:.2f}%"
            
            summary_data.append(row)
            
            df = pd.DataFrame(summary_data)
            csv_file = output_dir / f"{task_name}_evaluation_summary.csv"
            df.to_csv(csv_file, index=False)
            
            return str(summary_file)
        else:
            output_dir = base_output_dir / "all_tasks"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            summary_file = output_dir / "all_tasks_evaluation_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(all_summaries, f, indent=2)
            
            summary_data = []
            for task, summary in all_summaries.items():
                if task != "overall":
                    row = {
                        "Task": task,
                        "Original Molecules": summary.get("original_molecule_count", summary.get("total_molecules", 0)),  
                        "Total Modified Molecules": summary["total_molecules"],
                        "Valid SMILES": summary["valid_smiles_count"],
                        "Successful Repairs": summary["success_count"],
                        "Valid %": f"{summary['valid_percentage']:.2f}%",
                        "Success %": f"{summary['success_percentage']:.2f}%",
                    }
                    
                    if 'toxicity_improved_count' in summary:
                        row["Toxicity Improved"] = summary["toxicity_improved_count"]
                        row["Toxicity Improved %"] = f"{summary['toxicity_improved_percentage']:.2f}%"
                    
                    summary_data.append(row)

            if "overall" in all_summaries:
                overall = all_summaries["overall"]
                row = {
                    "Task": "OVERALL",
                    "Original Molecules": overall.get("original_molecule_count", overall.get("total_molecules", 0)),
                    "Total Modified Molecules": overall["total_molecules"],
                    "Valid SMILES": overall["valid_smiles_count"],
                    "Successful Repairs": overall["success_count"],
                    "Valid %": f"{overall['valid_percentage']:.2f}%",
                    "Success %": f"{overall['success_percentage']:.2f}%",
                }
                
                if 'toxicity_improved_count' in overall:
                    row["Toxicity Improved"] = overall["toxicity_improved_count"]
                    row["Toxicity Improved %"] = f"{overall['toxicity_improved_percentage']:.2f}%"
                    
                summary_data.append(row)
            
            df = pd.DataFrame(summary_data)
            csv_file = output_dir / "all_tasks_evaluation_summary.csv"
            df.to_csv(csv_file, index=False)
            
            if "tox21" in all_summaries or "toxcast" in all_summaries:
                self._generate_subtask_summaries(model, output_dir)
                
            return str(summary_file)
    
    def _generate_subtask_summaries(self, model: str, output_dir: Path) -> None:
        """
        Generate subtask evaluation summaries for tox21 and toxcast tasks.
        
        Args:
            model: Model name
            output_dir: Directory to save the summaries
        """
        if self._process_tox21_subtasks(model, output_dir):
            print(f"Generated tox21 subtask evaluation summary")
        
        if self._process_toxcast_subtasks(model, output_dir):
            print(f"Generated toxcast subtask evaluation summary")
    
    def _process_tox21_subtasks(self, model: str, output_dir: Path) -> bool:
        """
        Process tox21 subtasks and generate evaluation summary.
        
        Args:
            model: Model name
            output_dir: Directory to save the summary
            
        Returns:
            bool: True if processed successfully, False otherwise
        """
        try:
            evaluation_results, _ = self.evaluate_task_results(model, "tox21", True)
            
            subtasks = {}
            for result in evaluation_results:
                task = result.task
                if task not in subtasks:
                    subtasks[task] = []
                subtasks[task].append(result)
            
            subtask_summaries = {}
            for subtask, results in subtasks.items():
                results_by_molecule = {}
                for result in results:
                    if result.molecule_id not in results_by_molecule:
                        results_by_molecule[result.molecule_id] = []
                    results_by_molecule[result.molecule_id].append(result)
                
                original_molecule_count = len(results_by_molecule)
                
                molecules_per_original = 3 
                total_modified_molecules = original_molecule_count * molecules_per_original
                
                valid_smiles_count = sum(1 for r in results if r.details['validation']['valid_smiles'])
                
                success_count = 0
                for molecule_id, molecule_results in results_by_molecule.items():
                    if any(r.success for r in molecule_results):
                        success_count += 1
                
                subtask_summaries[subtask] = {
                    'task': subtask,
                    'model': model,
                    'original_molecule_count': original_molecule_count,
                    'total_molecules': total_modified_molecules,
                    'valid_smiles_count': valid_smiles_count,
                    'success_count': success_count,
                    'valid_percentage': valid_smiles_count / total_modified_molecules * 100 if total_modified_molecules > 0 else 0,
                    'success_percentage': success_count / original_molecule_count * 100 if original_molecule_count > 0 else 0
                }

                toxicity_improved_count = 0
                for molecule_id, molecule_results in results_by_molecule.items():
                    if any(r.details.get('toxicity', {}).get('improved', False) for r in molecule_results):
                        toxicity_improved_count += 1
                
                if toxicity_improved_count > 0:
                    subtask_summaries[subtask]['toxicity_improved_count'] = toxicity_improved_count
                    subtask_summaries[subtask]['toxicity_improved_percentage'] = toxicity_improved_count / original_molecule_count * 100 if original_molecule_count > 0 else 0
            
            json_file = output_dir / "tox21_subtasks_evaluation_summary.json"
            with open(json_file, 'w') as f:
                json.dump(subtask_summaries, f, indent=2)
            
            csv_data = []
            for subtask, summary in subtask_summaries.items():
                row = {
                    "Task": subtask,
                    "Original Molecules": summary["original_molecule_count"],
                    "Total Modified Molecules": summary["total_molecules"],
                    "Valid SMILES": summary["valid_smiles_count"],
                    "Successful Repairs": summary["success_count"],
                    "Valid %": f"{summary['valid_percentage']:.2f}%",
                    "Success %": f"{summary['success_percentage']:.2f}%",
                }
                
                if 'toxicity_improved_count' in summary:
                    row["Toxicity Improved"] = summary["toxicity_improved_count"]
                    row["Toxicity Improved %"] = f"{summary['toxicity_improved_percentage']:.2f}%"
                
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            csv_file = output_dir / "tox21_subtasks_evaluation_summary.csv"
            df.to_csv(csv_file, index=False)
            
            return True
        except Exception as e:
            print(f"Error processing tox21 subtasks: {e}")
            return False
    
    def _process_toxcast_subtasks(self, model: str, output_dir: Path) -> bool:
        """
        Process toxcast subtasks and generate evaluation summary.
        
        Args:
            model: Model name
            output_dir: Directory to save the summary
            
        Returns:
            bool: True if processed successfully, False otherwise
        """
        try:
            evaluation_results, _ = self.evaluate_task_results(model, "toxcast", True)
            
            subtasks = {}
            for result in evaluation_results:
                task = result.task
                if task not in subtasks:
                    subtasks[task] = []
                subtasks[task].append(result)
            
            subtask_summaries = {}
            for subtask, results in subtasks.items():
                results_by_molecule = {}
                for result in results:
                    if result.molecule_id not in results_by_molecule:
                        results_by_molecule[result.molecule_id] = []
                    results_by_molecule[result.molecule_id].append(result)
                
                original_molecule_count = len(results_by_molecule)
                
                molecules_per_original = 3 
                total_modified_molecules = original_molecule_count * molecules_per_original
                
                valid_smiles_count = sum(1 for r in results if r.details['validation']['valid_smiles'])
                
                success_count = 0
                for molecule_id, molecule_results in results_by_molecule.items():
                    if any(r.success for r in molecule_results):
                        success_count += 1
                
                subtask_summaries[subtask] = {
                    'task': subtask,
                    'model': model,
                    'original_molecule_count': original_molecule_count,
                    'total_molecules': total_modified_molecules,
                    'valid_smiles_count': valid_smiles_count,
                    'success_count': success_count,
                    'valid_percentage': valid_smiles_count / total_modified_molecules * 100 if total_modified_molecules > 0 else 0,
                    'success_percentage': success_count / original_molecule_count * 100 if original_molecule_count > 0 else 0
                }
                
                toxicity_improved_count = 0
                for molecule_id, molecule_results in results_by_molecule.items():
                    if any(r.details.get('toxicity', {}).get('improved', False) for r in molecule_results):
                        toxicity_improved_count += 1
                
                if toxicity_improved_count > 0:
                    subtask_summaries[subtask]['toxicity_improved_count'] = toxicity_improved_count
                    subtask_summaries[subtask]['toxicity_improved_percentage'] = toxicity_improved_count / original_molecule_count * 100 if original_molecule_count > 0 else 0

            json_file = output_dir / "toxcast_subtasks_evaluation_summary.json"
            with open(json_file, 'w') as f:
                json.dump(subtask_summaries, f, indent=2)
            
            csv_data = []
            for subtask, summary in subtask_summaries.items():
                row = {
                    "Task": subtask,
                    "Original Molecules": summary["original_molecule_count"],
                    "Total Modified Molecules": summary["total_molecules"],
                    "Valid SMILES": summary["valid_smiles_count"],
                    "Successful Repairs": summary["success_count"],
                    "Valid %": f"{summary['valid_percentage']:.2f}%",
                    "Success %": f"{summary['success_percentage']:.2f}%",
                }
                
                if 'toxicity_improved_count' in summary:
                    row["Toxicity Improved"] = summary["toxicity_improved_count"]
                    row["Toxicity Improved %"] = f"{summary['toxicity_improved_percentage']:.2f}%"
                
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            csv_file = output_dir / "toxcast_subtasks_evaluation_summary.csv"
            df.to_csv(csv_file, index=False)
            
            return True
        except Exception as e:
            print(f"Error processing toxcast subtasks: {e}")
            return False
    
    def _evaluate_repair_criteria(
        self,
        task_type: str,
        original_smiles: str,
        modified_smiles: str
    ) -> RepairResults:
        """
        Evaluate if a molecule meets the repair criteria.
        
        Criteria:
        1. Toxicity standard: Modified safety probability > 0.5
        2. Structure standards:
           - QED ≥ 0.5 
           - SAS score ≤ 6
           - Lipinski violations ≤ 1
           - Tanimoto similarity ≥ 0.4
        
        Args:
            task_type: Task type
            original_smiles: Original SMILES
            modified_smiles: Modified SMILES
            
        Returns:
            RepairResults: Results of repair evaluation
        """
        mapping = get_task_mapping(task_type)
        primary_endpoints = mapping.get("primary_endpoints", [])
        
        if not primary_endpoints:
            if task_type == "ames":
                primary_endpoints = ["AMES"]
            elif "herg" in task_type.lower():
                primary_endpoints = ["hERG"]
            else:
                primary_endpoints = ["AMES"]
        
        properties = calculate_properties(modified_smiles)
        qed = properties.get("qed", 0.0)
        sas_score = properties.get("sas_score", 10.0)
        lipinski_violations = properties.get("lipinski_violations", 0)
        
        similarity = calculate_similarity(original_smiles, modified_smiles)
        
        modified_toxicity_endpoints = {}
        for endpoint in primary_endpoints:
            value, probability = predict_toxicity(modified_smiles, endpoint)
            modified_toxicity_endpoints[endpoint] = ToxicityEndpointResult(
                endpoint=endpoint,
                value=value,
                probability=probability
            )
        
        original_toxicity_endpoints = {}
        for endpoint in primary_endpoints:
            if endpoint == "LD50_Zhu":
                original_toxicity_endpoints[endpoint] = ToxicityEndpointResult(
                    endpoint=endpoint,
                    value="100",
                    probability=0.1 
                )
            else:
                original_toxicity_endpoints[endpoint] = ToxicityEndpointResult(
                    endpoint=endpoint,
                    value="B",
                    probability=0.01  
                )

        toxicity_deltas = []
        toxicity_improved = False
        improvement_count = 0
        
        max_safety_prob = 0.0
        if modified_toxicity_endpoints:
            max_safety_prob = max(endpoint.probability for endpoint in modified_toxicity_endpoints.values())
        
        for endpoint in primary_endpoints:
            original = original_toxicity_endpoints.get(endpoint)
            modified = modified_toxicity_endpoints.get(endpoint)
            
            if original and modified:
                if original.value == modified.value and abs(original.probability - modified.probability) < 0.05:
                    changed = False
                    improved = False
                else:
                    changed = True
                    improved = modified.probability > original.probability
                
                toxicity_deltas.append(ToxicityDelta(
                    endpoint=endpoint,
                    original=original.value,
                    modified=modified.value,
                    original_prob=original.probability,
                    modified_prob=modified.probability,
                    changed=changed,
                    improved=improved
                ))
                
                if improved:
                    improvement_count += 1
        
        if improvement_count > 0:
            toxicity_improved = True
        
        fails = []
        
        if max_safety_prob <= 0.5:
            fails.append("Toxicity not reduced to safe level (safety probability ≤ 0.5)")
        
        if qed < 0.5:
            fails.append("Insufficient drug-likeness")
        
        if sas_score > 6:
            fails.append("High synthetic difficulty")
        
        if lipinski_violations > 1:
            fails.append("Too many Lipinski violations")
        
        if similarity < 0.4:
            fails.append("Low structural similarity")
        
        passed_repair = len(fails) == 0
        
        return RepairResults(
            toxicity_endpoints=modified_toxicity_endpoints,
            toxicity_improved=toxicity_improved,
            qed=qed,
            sas_score=sas_score,
            lipinski_violations=lipinski_violations,
            similarity=similarity,
            passed_repair=passed_repair,
            fails=fails
        )
    
    def _calculate_toxicity_deltas(
        self,
        modified_endpoints: Dict[str, ToxicityEndpointResult]
    ) -> List[ToxicityDelta]:
        """
        Calculate toxicity deltas between original and modified molecules.
        Original molecule's safety probability is set to 0.000001 or 0.01 (very toxic).
        
        Args:
            modified_endpoints: Toxicity endpoint results for modified molecule
            
        Returns:
            List[ToxicityDelta]: List of toxicity deltas
        """
        deltas = []
        
        for endpoint, modified in modified_endpoints.items():
            original_value = "B"
            original_prob = 0.01
            
            if endpoint == "LD50_Zhu":
                original_value = "100"
                original_prob = 0.1
            
            improved = modified.probability > original_prob
            
            deltas.append(ToxicityDelta(
                endpoint=endpoint,
                original=original_value,
                modified=modified.value,
                original_prob=original_prob,
                modified_prob=modified.probability,
                changed=True,
                improved=improved
            ))
        
        return deltas

def analyze_experiment_results(
    results_dir: str = "results/gpt",
    model: str = None,
    full_evaluation: bool = True,
    output_dir: str = "experiments/eval_results"
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze experiment results for all models or a specific model.
    
    Args:
        results_dir: Directory containing results
        model: Specific model to analyze (if None, analyze all models)
        full_evaluation: Whether to perform full property evaluation
        output_dir: Directory to save evaluation results
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping models to their summaries
    """
    evaluator = ResultEvaluator(results_dir, output_dir)

    results_path = Path(results_dir)
    if model:
        models = [model]
    else:
        models = [d.name for d in results_path.iterdir() if d.is_dir()]
    
    all_model_summaries = {}
    
    for model_name in models:
        try:
            summaries = evaluator.evaluate_all_results(model_name, full_evaluation)
            evaluator.save_evaluation_results(model_name, summaries)
            all_model_summaries[model_name] = summaries
            print(f"Evaluated model: {model_name}")
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
    
    return all_model_summaries 