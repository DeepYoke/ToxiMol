#!/usr/bin/env python3
"""
Toxicity Repair Experiment Runner using OpenAI GPT models.

This script runs toxicity repair experiments on molecules using OpenAI's models.
It can process single tasks or run batch experiments across multiple toxicity datasets.
"""

import os
import json
import base64
import argparse
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import openai
from openai import OpenAI
import logging
import base64
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("toxicity_repair")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

DEFAULT_API_KEY = "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"

AVAILABLE_TASKS = [
    "ames", "carcinogens_lagunin", "clintox", "dili", "herg", 
    "herg_central", "herg_karim", "ld50_zhu", "skin_reaction", 
    "tox21", "toxcast"
]
def save_base64_image(base64_str, output_path):
    img_data = base64.b64decode(base64_str)

    with open(output_path, 'wb') as f:
        f.write(img_data)
    print(f"Img is saved to tmp dir: {output_path}")

def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64 string for API submission.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_json_file(file_path: str) -> Dict:
    """
    Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data as dictionary
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        raise

def load_task_data(task_name: str) -> Tuple[List[Dict], Dict]:
    """
    Load molecule data and task prompt for a specific task.
    
    Args:
        task_name: Name of the task (e.g., "ames")
        
    Returns:
        Tuple of (molecules list, task prompt dictionary)
    """
    try:
        molecules_hf = load_dataset("DeepYoke/ToxiMol-benchmark", data_dir=task_name, split="train", trust_remote_code=True)
        molecules = molecules_hf.to_pandas().to_dict(orient='records')  
        
        prompt_path = BASE_DIR / "annotation" / f"{task_name}_prompt.json"
        task_prompt = load_json_file(prompt_path)
        
        return molecules, task_prompt
    except Exception as e:
        logger.error(f"Error loading data for task {task_name}: {e}")
        raise

def load_repair_prompt() -> Dict:
    """
    Load the main repair prompt.
    
    Returns:
        Repair prompt dictionary
    """
    repair_prompt_path = BASE_DIR / "annotation" / "repair_prompt.json"
    return load_json_file(repair_prompt_path)

def get_specific_prompt(task_name: str, molecule: Dict, task_prompt: Dict) -> str:
    """
    Generate the specific prompt for a molecule based on its task.
    
    Args:
        task_name: Name of the task
        molecule: Molecule data dictionary
        task_prompt: Task prompt dictionary
        
    Returns:
        Formatted prompt string for the molecule
    """
    if task_name == "tox21" and "subtasks" in task_prompt:
        subtask = molecule["task"].replace("tox21_", "")
        if subtask in task_prompt["subtasks"]:
            instruction = task_prompt["subtasks"][subtask]["instruction"]
        else:
            instruction = task_prompt["subtasks"]["default"]["instruction"]
    elif task_name == "toxcast" and "subtasks" in task_prompt:
        subtask = molecule["task"].replace("toxcast_", "")
        if subtask in task_prompt["subtasks"]:
            instruction = task_prompt["subtasks"][subtask]["instruction"]
        else:
            instruction = task_prompt["subtasks"]["default"]["instruction"]
    else:
        instruction = task_prompt["instruction"]
    
    return instruction.replace("{{ smiles }}", molecule["smiles"])

def create_repair_request(
    molecule: Dict, 
    task_name: str,
    task_prompt: Dict,
    repair_prompt: Dict,
    image_path: str
) -> List[Dict]:
    """
    Create the full prompt for the OpenAI API request.
    
    Args:
        molecule: Molecule data dictionary
        task_name: Name of the task
        task_prompt: Task prompt dictionary
        repair_prompt: Main repair prompt dictionary
        image_path: Path to the molecule image
        
    Returns:
        List of message dictionaries for the API request
    """
    specific_instruction = get_specific_prompt(task_name, molecule, task_prompt)
    
    encoded_image = encode_image(image_path)
    
    system_content = (
        f"You are a {repair_prompt['agent_role']}. "
        f"{repair_prompt['task_overview']} "
        f"Follow these guidelines for working with the molecular structure image: "
        f"{'; '.join(repair_prompt['image_integration']['usage_guidelines'])}. "
        f"Your output must strictly follow this format: {repair_prompt['output_format']['structure']}"
    )
    
    user_content = [
        {
            "type": "text",
            "text": (
                f"Task: Modify the following molecule to reduce its {task_prompt['task_description']} while "
                f"maintaining its therapeutic properties.\n\n"
                f"SMILES: {molecule['smiles']}\n\n"
                f"{specific_instruction}\n\n"
                f"IMPORTANT: Provide 1-3 modified versions of this molecule following EXACTLY the format: "
                f"'MODIFIED_SMILES: smiles1;smiles2;smiles3'. Use semicolons to separate multiple SMILES. "
                f"No explanations or other text should be included."
            )
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encoded_image}"
            }
        }
    ]
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    return messages

def call_openai_api(
    client: OpenAI, 
    messages: List[Dict], 
    model: str, 
    max_retries: int = 3, 
    retry_delay: int = 5
) -> str:
    """
    Call the OpenAI API with retry logic.
    
    Args:
        client: OpenAI client instance
        messages: List of message dictionaries
        model: Model name to use
        max_retries: Maximum number of retries on error
        retry_delay: Delay between retries in seconds
        
    Returns:
        API response content
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"Failed after {max_retries} attempts")
                raise

def extract_results(response_text: str) -> Dict:
    """
    Parse the API response to extract the modified SMILES.
    
    Args:
        response_text: Raw text response from the API
        
    Returns:
        Dictionary containing the parsed results
    """
    results = {
        "modified_smiles": [],
        "raw_response": response_text
    }
    
    if "MODIFIED_SMILES:" in response_text:
        smiles_part = response_text.split("MODIFIED_SMILES:")[1].strip()
        if ";" in smiles_part:
            smiles_candidates = [s.strip() for s in smiles_part.split(";") if s.strip()]
            results["modified_smiles"] = smiles_candidates
        else:
            if smiles_part.lower().strip() == "none":
                results["modified_smiles"] = []
            else:
                results["modified_smiles"] = [smiles_part.strip()]
    
    return results

def process_molecule(
    client: OpenAI,
    molecule: Dict,
    task_name: str,
    task_prompt: Dict,
    repair_prompt: Dict,
    model: str
) -> Dict:
    """
    Process a single molecule and return the results.
    
    Args:
        client: OpenAI client instance
        molecule: Molecule data dictionary
        task_name: Name of the task
        task_prompt: Task prompt dictionary
        repair_prompt: Main repair prompt dictionary
        model: Model name to use
        
    Returns:
        Dictionary with the results and metadata
    """
    molecule_id = molecule["id"]
    logger.info(f"Processing {task_name} molecule ID: {molecule_id}")
    task = molecule['task']
    image_binary = molecule["image"]
    tmp_dir = f'~/toximol_tmp_images/{task}'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    image_path = os.path.join(tmp_dir, f'{molecule_id}.png')
    save_base64_image(image_binary, image_path)
    
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        raise FileNotFoundError(f"Image not found: {image_path}")

    messages = create_repair_request(
        molecule, 
        task_name, 
        task_prompt, 
        repair_prompt, 
        str(image_path)
    )
    
    try:
        response = call_openai_api(client, messages, model)
        
        results = extract_results(response)
        
        results["task"] = molecule["task"] if "task" in molecule else task_name
        results["molecule_id"] = molecule_id
        results["original_smiles"] = molecule["smiles"]
        results["model"] = model
        
        logger.info(f"Successfully processed molecule {molecule_id}")
        return results
    
    except Exception as e:
        logger.error(f"Error processing molecule {molecule_id}: {e}")
        error_result = {
            "task": molecule["task"] if "task" in molecule else task_name,
            "molecule_id": molecule_id,
            "original_smiles": molecule["smiles"],
            "model": model,
            "error": str(e),
            "modified_smiles": [],
            "raw_response": ""
        }
        
        return error_result

def cleanup_old_results(output_dir: Path, task_name: str):
    """
    Clean up old individual result files if they exist.
    
    Args:
        output_dir: Directory containing results
        task_name: Name of the task
    """
    try:
        individual_files = list(output_dir.glob(f"{task_name}_*.json"))
        if individual_files:
            logger.info(f"Cleaning up {len(individual_files)} old individual result files")
            for file in individual_files:
                if file.name == f"{task_name}_results.json":
                    continue
                try:
                    os.remove(file)
                except Exception as e:
                    logger.warning(f"Failed to remove file {file}: {e}")
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")

def run_task(
    task_name: str,
    model: str,
    api_key: str,
    molecule_limit: Optional[int] = None,
    molecules_ids: Optional[List[int]] = None
) -> List[Dict]:
    """
    Run the experiment for a specific task.
    
    Args:
        task_name: Name of the task to run
        model: Model name to use
        api_key: OpenAI API key
        molecule_limit: Maximum number of molecules to process (optional)
        molecules_ids: Specific molecule IDs to process (optional)
        
    Returns:
        List of result dictionaries
    """
    client = OpenAI(api_key=api_key)
    
    output_dir = RESULTS_DIR / model / task_name
    os.makedirs(output_dir, exist_ok=True)
    
    cleanup_old_results(output_dir, task_name)
    
    molecules, task_prompt = load_task_data(task_name)
    repair_prompt = load_repair_prompt()
    
    if molecules_ids:
        molecules = [m for m in molecules if m["id"] in molecules_ids]
    
    if molecule_limit and molecule_limit > 0:
        molecules = molecules[:molecule_limit]
    
    logger.info(f"Running task {task_name} with {len(molecules)} molecules")
    
    all_results = []
    for molecule in molecules:
        result = process_molecule(
            client,
            molecule,
            task_name,
            task_prompt,
            repair_prompt,
            model
        )
        all_results.append(result)
        
        time.sleep(1)
    
    combined_results = {
        "task_name": task_name,
        "model": model,
        "total_molecules": len(molecules),
        "success_count": sum(1 for r in all_results if "error" not in r),
        "error_count": sum(1 for r in all_results if "error" in r),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results
    }

    output_file = output_dir / f"{task_name}_results.json"
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    logger.info(f"Saved combined results for {task_name} to {output_file}")
    
    return all_results

def run_all_tasks(
    model: str,
    api_key: str,
    molecule_limit: Optional[int] = None
) -> Dict[str, List[Dict]]:
    """
    Run the experiment for all available tasks.
    
    Args:
        model: Model name to use
        api_key: OpenAI API key
        molecule_limit: Maximum number of molecules to process per task (optional)
        
    Returns:
        Dictionary mapping task names to lists of result dictionaries
    """
    all_results = {}
    
    for task_name in AVAILABLE_TASKS:
        logger.info(f"Starting task: {task_name}")
        task_results = run_task(task_name, model, api_key, molecule_limit)
        all_results[task_name] = task_results
    
    overall_summary = {
        "model": model,
        "tasks_completed": len(AVAILABLE_TASKS),
        "total_molecules": sum(len(results) for results in all_results.values()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_dir = RESULTS_DIR / model
    os.makedirs(summary_dir, exist_ok=True)
    
    with open(summary_dir / "overall_summary.json", 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    return all_results

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run toxicity repair experiments using OpenAI models")
    
    parser.add_argument(
        "--task", 
        choices=AVAILABLE_TASKS + ["all"], 
        default="all",
        help="Task to run (default: all)"
    )
    
    parser.add_argument(
        "--model", 
        default="gpt-4.1",
        help="OpenAI model to use (default: gpt-4.1)"
    )
    
    parser.add_argument(
        "--api-key", 
        default=DEFAULT_API_KEY,
        help="OpenAI API key (default: use environment variable)"
    )
    
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit the number of molecules per task (default: no limit)"
    )
    
    parser.add_argument(
        "--molecule-ids", 
        type=int, 
        nargs="+",
        help="Specific molecule IDs to process (default: all molecules)"
    )
    
    args = parser.parse_args()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    logger.info(f"Starting experiment with model: {args.model}")
    
    if args.task == "all":
        run_all_tasks(args.model, args.api_key, args.limit)
    else:
        run_task(args.task, args.model, args.api_key, args.limit, args.molecule_ids)
    
    logger.info("Experiment completed")

if __name__ == "__main__":
    main() 