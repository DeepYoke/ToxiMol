#!/usr/bin/env python3
"""
Evaluation script for toxicity repair experiments using TxGemma.

This script analyzes the results of toxicity repair experiments 
and produces evaluation metrics using TxGemma for toxicity prediction.
The TxGemma model predicts toxicity by extracting the first A or B from generated text.
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any

from result_evaluator import (
    ResultEvaluator,
    analyze_experiment_results
)
from molecule_utils import load_txgemma_model

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Evaluate toxicity repair experiment results with TxGemma")
    
    parser.add_argument(
        "--results-dir", 
        default="experiments/gpt/results",
        help="Directory containing experiment results (default: experiments/gpt/results)"
    )
    
    parser.add_argument(
        "--model", 
        default=None,
        help="Specific model to evaluate (default: evaluate all models)"
    )
    
    parser.add_argument(
        "--full", 
        action="store_true",
        help="Perform full evaluation with molecular property calculations"
    )
    
    parser.add_argument(
        "--task", 
        default=None,
        help="Evaluate a specific task (default: evaluate all tasks)"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="experiments/eval_results",
        help="Directory to save evaluation results (default: experiments/eval_results)"
    )
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' does not exist.")
        sys.exit(1)
    
    # Load TxGemma model for toxicity prediction
    print("Loading TxGemma model for toxicity prediction...")
    load_txgemma_model()
    print("TxGemma will classify toxicity by extracting the first 'A' or 'B' from generated text.")
    
    # If a specific task is provided, only evaluate that task
    if args.task and args.model:
        print(f"Evaluating task '{args.task}' for model '{args.model}'...")
        evaluator = ResultEvaluator(args.results_dir, args.output_dir)
        results, summary = evaluator.evaluate_task_results(args.model, args.task, args.full)
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Task: {args.task}")
        print(f"Model: {args.model}")
        print(f"Total molecules: {summary['total_molecules']}")
        print(f"Valid SMILES: {summary['valid_smiles_count']} ({summary['valid_percentage']:.2f}%)")
        print(f"Successful repairs: {summary['success_count']} ({summary['success_percentage']:.2f}%)")
        
        # Print toxicity improvements if available
        if 'toxicity_improved_count' in summary:
            print(f"Toxicity improved: {summary['toxicity_improved_count']} ({summary['toxicity_improved_percentage']:.2f}%)")
        
        # Save detailed results will be handled by evaluator
        
        # Create a dictionary with task summary
        task_summary = {args.task: summary}
        
        # Save summary
        evaluator.save_evaluation_results(args.model, task_summary)
        
    else:
        # Evaluate all models or a specific model
        print(f"Analyzing experiment results in '{args.results_dir}'...")
        if args.model:
            print(f"Focusing on model: {args.model}")
        
        all_model_summaries = analyze_experiment_results(
            args.results_dir,
            args.model,
            args.full,
            args.output_dir
        )
        
        # Print overall summary
        print("\nOverall Evaluation Summary:")
        for model_name, summaries in all_model_summaries.items():
            if 'overall' in summaries:
                overall = summaries['overall']
                print(f"\nModel: {model_name}")
                print(f"Total molecules: {overall['total_molecules']}")
                print(f"Valid SMILES: {overall['valid_smiles_count']} ({overall['valid_percentage']:.2f}%)")
                print(f"Successful repairs: {overall['success_count']} ({overall['success_percentage']:.2f}%)")
                
                # Print toxicity improvements if available
                if 'toxicity_improved_count' in overall:
                    print(f"Toxicity improved: {overall['toxicity_improved_count']} ({overall['toxicity_improved_percentage']:.2f}%)")
                
                print(f"Tasks completed: {overall['tasks_completed']}")
    
    print("\nEvaluation completed.")

if __name__ == "__main__":
    main() 