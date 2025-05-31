"""
Main entry point for the ToxiFixer system.
"""
import os
import asyncio
import argparse
from dotenv import load_dotenv

from src.agents.agent_manager import AgentManager
from src.utils.data_utils import load_data, create_task_datasets
from src.prompts import task_registry

# Load environment variables
load_dotenv()

def parse_args():
    """Parse command line arguments."""
    # Get all available tasks from registry
    all_tasks = task_registry.list_tasks()
    
    parser = argparse.ArgumentParser(description="ToxiFixer: Drug Toxicity Repair System")
    parser.add_argument(
        "--task", 
        type=str, 
        choices=all_tasks,
        help="Toxicity repair task to run"
    )
    parser.add_argument(
        "--smiles", 
        type=str, 
        help="SMILES string of the molecule to repair"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4.1",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--data_file", 
        type=str, 
        default="data/task_specific_samples.csv",
        help="Path to the input data file"
    )
    parser.add_argument(
        "--max_attempts", 
        type=int, 
        default=3,
        help="Maximum number of repair attempts"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=60,
        help="Timeout in seconds for API calls"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Path to save the results"
    )
    parser.add_argument(
        "--format", 
        type=str, 
        choices=['json', 'jsonl'],
        default='json',
        help="Format for saving results (default: json)"
    )
    
    return parser.parse_args()

async def main():
    """Main function."""
    args = parse_args()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it in .env file or as an environment variable.")
        return
    
    # Initialize agent manager
    agent_manager = AgentManager(
        manager_model=args.model,
        repair_model=args.model,
        eval_model=args.model
    )
    
    # If SMILES and task are provided, run the task directly
    if args.smiles and args.task:
        request_text = f"SMILES: {args.smiles}\nTask: {args.task}"
        
        print(f"Running task with input:\n{request_text}\n")
        
        result = await agent_manager.run_task(request_text, timeout=args.timeout)
        
        print("\n=== Task Result ===")
        print(f"Original SMILES: {result.original_smiles}")
        print(f"Success: {result.success}")
        print(f"Attempts: {result.attempts}")
        print(f"Feedback: {result.feedback}")
        
        if result.success and result.best_modification:
            print(f"Best modification: {result.best_modification}")
            
            if result.evaluation_details:
                print("\n=== Evaluation Details ===")
                best_mod = result.evaluation_details.get("best_modification", {})
                print(f"QED: {best_mod.get('qed', 0):.3f}")
                print(f"LogP: {best_mod.get('logp', 0):.3f}")
                print(f"Lipinski violations: {best_mod.get('lipinski_violations', 0)}")
                print(f"Potentially mutagenic: {best_mod.get('potentially_mutagenic', False)}")
        
        if hasattr(args, 'output') and args.output:
            output_format = getattr(args, 'format', 'json')
            print(f"\nSaving results to {args.output} in {output_format} format...")
            agent_manager.save_result(result, args.output, format=output_format)
            print("Results saved successfully.")
    else:
        print("No SMILES or task provided. Use --smiles and --task to specify a molecule and task.")
        print("Available task categories:")
        
        for category in task_registry.list_categories():
            tasks = task_registry.list_tasks(category)
            description = task_registry.get_category_description(category)
            print(f"  - {category.upper()} ({len(tasks)} tasks): {description}")
            print(f"    Example: {tasks[0] if tasks else 'none'}")
        
        print("\nUse --task parameter to specify a task")

if __name__ == "__main__":
    asyncio.run(main()) 