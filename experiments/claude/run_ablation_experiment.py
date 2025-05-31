#!/usr/bin/env python3
"""
Toxicity Repair Ablation Experiment Runner using Claude models via 智增增 API.

This script runs ablation experiments on molecule generation, varying the number of
molecules to generate from 1 to 9. It uses the Claude models through 智增增 API service.
"""

import os
import json
import base64
import argparse
import time
import requests
import shutil
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("claude_ablation_experiment")

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "Experimental_dataset"
RESULTS_DIR = Path(__file__).resolve().parent / "results_ablation"

# 智增增 API settings
ZZZ_API_KEY = "sk-zk2bf6cbabe8c8f27ccbb928b5e3bde3cb767cab83731789"
ZZZ_BASE_URL = "https://api.zhizengzeng.com/v1"

# All available Claude models
AVAILABLE_MODELS = [
    "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-thinking",
    "claude-3-opus-20240229"
]

# All available tasks
AVAILABLE_TASKS = [
    "ames", "carcinogens_lagunin", "clintox", "dili", "herg", 
    "herg_central", "herg_karim", "ld50_zhu", "skin_reaction", 
    "tox21", "toxcast"
]

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
        # Load molecules data
        molecules_path = DATA_DIR / task_name / f"{task_name}.json"
        molecules = load_json_file(molecules_path)
        
        # Load task prompt
        prompt_path = DATA_DIR / task_name / f"{task_name}_prompt.json"
        task_prompt = load_json_file(prompt_path)
        
        return molecules, task_prompt
    except Exception as e:
        logger.error(f"Error loading data for task {task_name}: {e}")
        raise

def load_repair_prompt() -> Dict:
    """
    Load the ablation repair prompt.
    
    Returns:
        Repair prompt dictionary
    """
    repair_prompt_path = DATA_DIR / "repair_prompt_ablation.json"
    return load_json_file(repair_prompt_path)

def replace_num_molecules_placeholder(repair_prompt: Dict, num_molecules: int) -> Dict:
    """
    Replace the {{num_molecules}} placeholder in the repair prompt with the actual number.
    
    Args:
        repair_prompt: The repair prompt dictionary
        num_molecules: Number of molecules to generate
        
    Returns:
        Updated repair prompt dictionary
    """
    # Make a deep copy to avoid modifying the original
    import copy
    updated_prompt = copy.deepcopy(repair_prompt)
    
    # Update the structure format
    structure = updated_prompt["output_format"]["structure"]
    updated_prompt["output_format"]["structure"] = structure.replace("{{num_molecules}}", str(num_molecules))
    
    # Update each note
    for i, note in enumerate(updated_prompt["output_format"]["notes"]):
        updated_prompt["output_format"]["notes"][i] = note.replace("{{num_molecules}}", str(num_molecules))
    
    return updated_prompt

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
    # Handle special cases for tox21 and toxcast with subtasks
    if task_name == "tox21" and "subtasks" in task_prompt:
        # Extract actual subtask name from the molecule task field (e.g., "tox21_SR_ARE" -> "SR_ARE")
        subtask = molecule["task"].replace("tox21_", "")
        if subtask in task_prompt["subtasks"]:
            instruction = task_prompt["subtasks"][subtask]["instruction"]
        else:
            instruction = task_prompt["subtasks"]["default"]["instruction"]
    elif task_name == "toxcast" and "subtasks" in task_prompt:
        # Extract actual subtask name from the molecule task field (e.g., "toxcast_APR_HepG2_MitoMass_24h_dn" -> "APR_HepG2_MitoMass_24h_dn")
        subtask = molecule["task"].replace("toxcast_", "")
        if subtask in task_prompt["subtasks"]:
            instruction = task_prompt["subtasks"][subtask]["instruction"]
        else:
            instruction = task_prompt["subtasks"]["default"]["instruction"]
    else:
        # Regular tasks
        instruction = task_prompt["instruction"]
    
    # Return the instruction, replacing any {{ smiles }} placeholders with the actual SMILES
    return instruction.replace("{{ smiles }}", molecule["smiles"])

def create_repair_request(
    molecule: Dict, 
    task_name: str,
    task_prompt: Dict,
    repair_prompt: Dict,
    image_path: str,
    model: str,
    num_molecules: int
) -> List[Dict]:
    """
    Create the messages for the API request.
    
    Args:
        molecule: Molecule data dictionary
        task_name: Name of the task
        task_prompt: Task prompt dictionary
        repair_prompt: Main repair prompt dictionary
        image_path: Path to the molecule image
        model: Model name to use
        num_molecules: Number of molecules to generate
        
    Returns:
        List of message dictionaries for the API request
    """
    # Update repair prompt with the specific number of molecules
    updated_repair_prompt = replace_num_molecules_placeholder(repair_prompt, num_molecules)
    
    # Generate specific instruction for the molecule
    specific_instruction = get_specific_prompt(task_name, molecule, task_prompt)
    
    # Encode the molecule image
    encoded_image = encode_image(image_path)
    
    # Create the system message from the repair prompt
    system_content = (
        f"You are a {updated_repair_prompt['agent_role']}. "
        f"{updated_repair_prompt['task_overview']} "
        f"Follow these guidelines for working with the molecular structure image: "
        f"{'; '.join(updated_repair_prompt['image_integration']['usage_guidelines'])}. "
        f"Your output must strictly follow this format: {updated_repair_prompt['output_format']['structure']}"
    )
    
    # Enhanced prompt to get better responses
    user_content = (
        f"IMPORTANT TASK: Modify the following molecule to reduce its {task_prompt['task_description']} while "
        f"maintaining its therapeutic properties.\n\n"
        f"SMILES: {molecule['smiles']}\n\n"
        f"{specific_instruction}\n\n"
        f"YOU MUST PROVIDE EXACTLY {num_molecules} MODIFIED VERSION(S). DO NOT RETURN EMPTY RESPONSES.\n\n"
        f"RESPOND EXACTLY IN THIS FORMAT AND NOTHING ELSE:\n"
        f"MODIFIED_SMILES: [modified_smiles_1];[modified_smiles_2];...;[modified_smiles_{num_molecules}]\n\n"
        f"Use semicolons to separate multiple SMILES. Provide exactly {num_molecules} modification(s)."
    )
    
    # Claude模型支持多模态输入，使用和o4-mini相同的格式
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": [
            {"type": "text", "text": user_content},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}"
                }
            }
        ]}
    ]
    
    return messages

def call_zhizengzeng_api(
    messages: List[Dict], 
    model: str, 
    api_key: str = ZZZ_API_KEY,
    max_retries: int = 3, 
    retry_delay: int = 5
) -> str:
    """
    Call the 智增增 API to access Claude models.
    
    Args:
        messages: List of message dictionaries
        model: Model name to use
        api_key: 智增增 API key
        max_retries: Maximum number of retries on error
        retry_delay: Delay between retries in seconds
        
    Returns:
        API response content
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 基础参数设置
    params = {
        "model": model,
        "messages": messages,
        "temperature": 0.7  # Claude模型支持temperature参数
    }
    
    # 对于thinking模型，添加thinking参数
    if "thinking" in model:
        params["thinking"] = True
    
    # Debug log
    logger.info(f"Calling API with model: {model}")
    logger.info(f"API parameters: {json.dumps({k: v for k, v in params.items() if k != 'messages'})}")
    
    # API URL
    url = f"{ZZZ_BASE_URL}/chat/completions"
    
    for attempt in range(max_retries):
        try:
            # Make the API call
            response = requests.post(url, json=params, headers=headers)
            
            # Debug the raw response
            logger.info(f"API raw response: {response.text}")
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse JSON response
            response_data = response.json()
            
            # Check if the API call was successful
            if response_data.get("code") != 0:
                error_msg = response_data.get("msg", "Unknown error")
                raise Exception(f"API error: {error_msg}")
            
            # Check for refusal or empty content
            if "choices" in response_data and len(response_data["choices"]) > 0:
                if "message" in response_data["choices"][0]:
                    message = response_data["choices"][0]["message"]
                    
                    # Check for refusal
                    if "refusal" in message and message["refusal"]:
                        logger.warning(f"Model refused to respond: {message.get('refusal')}")
                    
                    # Check for empty content
                    if "content" in message:
                        content = message["content"]
                        if not content or content.strip() == "":
                            logger.warning("Model returned empty content")
                        return content
                    else:
                        logger.error("Response message missing content field")
                else:
                    logger.error("Response choice missing message field")
            else:
                logger.error("Response missing choices or empty choices array")
            
            # If we reached here without returning, something unusual happened
            logger.error(f"Unexpected response structure: {response_data}")
            return "MODIFIED_SMILES: N/A"  # 提供默认响应以避免空结果
            
        except Exception as e:
            logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Increase delay for next retry (exponential backoff)
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
    # Initialize results
    results = {
        "modified_smiles": [],
        "raw_response": response_text
    }
    
    # Handle empty responses
    if not response_text or response_text.strip() == "":
        logger.warning("Empty response received, no SMILES to extract")
        return results
    
    # Extract modified SMILES using the strict format
    if "MODIFIED_SMILES:" in response_text:
        # Get everything after MODIFIED_SMILES:
        smiles_part = response_text.split("MODIFIED_SMILES:")[1].strip()
        # Split by semicolons and clean up
        if ";" in smiles_part:
            smiles_candidates = [s.strip() for s in smiles_part.split(";") if s.strip()]
            results["modified_smiles"] = smiles_candidates
        else:
            # Handle case with only one SMILES or 'none'
            if smiles_part.lower().strip() == "none" or smiles_part.lower().strip() == "n/a":
                results["modified_smiles"] = []
            else:
                results["modified_smiles"] = [smiles_part.strip()]
    else:
        logger.warning(f"Response does not contain expected 'MODIFIED_SMILES:' format: {response_text[:100]}...")
    
    # 如果是思考模型，尝试提取思考过程
    if "THINKING:" in response_text and "MODIFIED_SMILES:" in response_text:
        thinking_part = response_text.split("THINKING:")[1].split("MODIFIED_SMILES:")[0].strip()
        results["thinking_process"] = thinking_part
    
    return results

def process_molecule(
    molecule: Dict,
    task_name: str,
    task_prompt: Dict,
    repair_prompt: Dict,
    model: str,
    num_molecules: int,
    api_key: str = ZZZ_API_KEY
) -> Dict:
    """
    Process a single molecule and return the results.
    
    Args:
        molecule: Molecule data dictionary
        task_name: Name of the task
        task_prompt: Task prompt dictionary
        repair_prompt: Main repair prompt dictionary
        model: Model name to use
        num_molecules: Number of molecules to generate
        api_key: 智增增 API key
        
    Returns:
        Dictionary with the results and metadata
    """
    molecule_id = molecule["id"]
    logger.info(f"Processing {task_name} molecule ID: {molecule_id} with {num_molecules} molecule(s)")
    
    # Get the image path
    image_path = DATA_DIR / task_name / "image" / f"{molecule_id}.png"
    
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Create the API request
    messages = create_repair_request(
        molecule, 
        task_name, 
        task_prompt, 
        repair_prompt, 
        str(image_path),
        model,
        num_molecules
    )
    
    # Call the API
    try:
        response = call_zhizengzeng_api(messages, model, api_key)
        
        # Parse the results
        results = extract_results(response)
        
        # Add metadata
        results["task"] = molecule["task"] if "task" in molecule else task_name
        results["molecule_id"] = molecule_id
        results["original_smiles"] = molecule["smiles"]
        results["model"] = model
        results["num_molecules_requested"] = num_molecules
        results["num_molecules_generated"] = len(results["modified_smiles"])
        
        logger.info(f"Successfully processed molecule {molecule_id}")
        return results
    
    except Exception as e:
        logger.error(f"Error processing molecule {molecule_id}: {e}")
        # Create error result
        error_result = {
            "task": molecule["task"] if "task" in molecule else task_name,
            "molecule_id": molecule_id,
            "original_smiles": molecule["smiles"],
            "model": model,
            "num_molecules_requested": num_molecules,
            "num_molecules_generated": 0,
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
        # Find all individual result files
        individual_files = list(output_dir.glob(f"{task_name}_*.json"))
        if individual_files:
            logger.info(f"Cleaning up {len(individual_files)} old individual result files")
            for file in individual_files:
                # Skip the combined results file
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
    num_molecules: int,
    api_key: str = ZZZ_API_KEY,
    molecule_limit: Optional[int] = None,
    molecules_ids: Optional[List[int]] = None
) -> List[Dict]:
    """
    Run the experiment for a specific task with a specific number of molecules to generate.
    
    Args:
        task_name: Name of the task to run
        model: Model name to use
        num_molecules: Number of molecules to generate
        api_key: 智增增 API key
        molecule_limit: Maximum number of molecules to process (optional)
        molecules_ids: Specific molecule IDs to process (optional)
        
    Returns:
        List of result dictionaries
    """
    # Create output directory
    output_dir = RESULTS_DIR / f"molecules_{num_molecules}" / model / task_name
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean up old individual result files if they exist
    cleanup_old_results(output_dir, task_name)
    
    # Load task data
    molecules, task_prompt = load_task_data(task_name)
    repair_prompt = load_repair_prompt()
    
    # Filter molecules if needed
    if molecules_ids:
        molecules = [m for m in molecules if m["id"] in molecules_ids]
    
    if molecule_limit and molecule_limit > 0:
        molecules = molecules[:molecule_limit]
    
    logger.info(f"Running task {task_name} with {len(molecules)} molecules, generating {num_molecules} per molecule")
    
    # Process each molecule
    all_results = []
    for molecule in molecules:
        result = process_molecule(
            molecule,
            task_name,
            task_prompt,
            repair_prompt,
            model,
            num_molecules,
            api_key
        )
        all_results.append(result)
        
        # Add a short delay to avoid rate limits
        time.sleep(1)
    
    # Create combined results
    combined_results = {
        "task_name": task_name,
        "model": model,
        "num_molecules_to_generate": num_molecules,
        "total_molecules": len(molecules),
        "success_count": sum(1 for r in all_results if "error" not in r),
        "error_count": sum(1 for r in all_results if "error" in r),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results
    }
    
    # Save combined results to a single file
    output_file = output_dir / f"{task_name}_results.json"
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    logger.info(f"Saved combined results for {task_name} to {output_file}")
    
    return all_results

def run_all_tasks(
    model: str,
    num_molecules: int,
    api_key: str = ZZZ_API_KEY,
    molecule_limit: Optional[int] = None
) -> Dict[str, List[Dict]]:
    """
    Run the experiment for all available tasks with a specific number of molecules to generate.
    
    Args:
        model: Model name to use
        num_molecules: Number of molecules to generate
        api_key: 智增增 API key
        molecule_limit: Maximum number of molecules to process per task (optional)
        
    Returns:
        Dictionary mapping task names to lists of result dictionaries
    """
    all_results = {}
    
    for task_name in AVAILABLE_TASKS:
        logger.info(f"Starting task: {task_name} with {num_molecules} molecules to generate")
        task_results = run_task(task_name, model, num_molecules, api_key, molecule_limit)
        all_results[task_name] = task_results
    
    # Save overall summary
    overall_summary = {
        "model": model,
        "num_molecules_to_generate": num_molecules,
        "tasks_completed": len(AVAILABLE_TASKS),
        "total_molecules": sum(len(results) for results in all_results.values()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_dir = RESULTS_DIR / f"molecules_{num_molecules}" / model
    os.makedirs(summary_dir, exist_ok=True)
    
    with open(summary_dir / "overall_summary.json", 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    return all_results

def run_ablation_experiment(
    model: str,
    api_key: str = ZZZ_API_KEY,
    start_num: int = 1,
    end_num: int = 9,
    molecule_limit: Optional[int] = None,
    task: str = "all"
):
    """
    Run the full ablation experiment, varying the number of molecules to generate.
    
    Args:
        model: Model name to use
        api_key: 智增增 API key
        start_num: Starting number of molecules to generate
        end_num: Ending number of molecules to generate
        molecule_limit: Maximum number of molecules to process per task (optional)
        task: Specific task to run, or "all" for all tasks
    """
    logger.info(f"Starting ablation experiment with model: {model}")
    logger.info(f"Will run experiments for {start_num} to {end_num} molecules to generate")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run for each number of molecules
    for num_molecules in range(start_num, end_num + 1):
        logger.info(f"=== Running experiment with {num_molecules} molecules to generate ===")
        
        if task == "all":
            run_all_tasks(model, num_molecules, api_key, molecule_limit)
        else:
            if task in AVAILABLE_TASKS:
                run_task(task, model, num_molecules, api_key, molecule_limit)
            else:
                logger.error(f"Invalid task: {task}. Must be one of {AVAILABLE_TASKS} or 'all'")
                return
        
        logger.info(f"Completed experiment for {num_molecules} molecules to generate")
    
    logger.info("Ablation experiment completed successfully")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run toxicity repair ablation experiments using Claude models via 智增增 API")
    
    parser.add_argument(
        "--task", 
        choices=AVAILABLE_TASKS + ["all"], 
        default="all",
        help="Task to run (default: all)"
    )
    
    parser.add_argument(
        "--model", 
        choices=AVAILABLE_MODELS,
        default="claude-3-7-sonnet-20250219",
        help="Claude model to use (default: claude-3-7-sonnet-20250219)"
    )
    
    parser.add_argument(
        "--api-key", 
        default=ZZZ_API_KEY,
        help="智增增 API key (default: use predefined key)"
    )
    
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit the number of molecules per task (default: no limit)"
    )
    
    parser.add_argument(
        "--start-num", 
        type=int, 
        default=1,
        help="Starting number of molecules to generate (default: 1)"
    )
    
    parser.add_argument(
        "--end-num", 
        type=int, 
        default=9,
        help="Ending number of molecules to generate (default: 9)"
    )
    
    parser.add_argument(
        "--single-num", 
        type=int, 
        help="Run for a single specific number of molecules (overrides start-num and end-num)"
    )
    
    parser.add_argument(
        "--molecule-ids", 
        type=int, 
        nargs="+",
        help="Specific molecule IDs to process (default: all molecules)"
    )
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    logger.info(f"Starting ablation experiment with model: {args.model} via 智增增 API")
    
    if args.single_num:
        # Run for a single number of molecules
        if args.task == "all":
            run_all_tasks(args.model, args.single_num, args.api_key, args.limit)
        else:
            run_task(args.task, args.model, args.single_num, args.api_key, args.limit, args.molecule_ids)
    else:
        # Run the full ablation experiment
        run_ablation_experiment(
            args.model, 
            args.api_key, 
            args.start_num,
            args.end_num,
            args.limit,
            args.task
        )
    
    logger.info("Experiment completed")

if __name__ == "__main__":
    main() 