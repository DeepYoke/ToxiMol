"""
Modified load_task_data function for run_toxicity_repair.py
This function replaces the original load_task_data to use Hugging Face datasets library.
"""

from datasets import load_dataset
from typing import List, Dict, Tuple
from pathlib import Path
import json
import os

# Update this to use your local data directory for prompts
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "Experimental_dataset"

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
    Load molecule data and task prompt for a specific task using Hugging Face datasets.
    
    Args:
        task_name: Name of the task (e.g., "ames")
        
    Returns:
        Tuple of (molecules list, task prompt dictionary)
    """
    try:
        # Load molecules data using Hugging Face datasets
        dataset = load_dataset("treasurels/ToxiMol-benchmark", task_name)
        
        # Convert dataset to list of dictionaries with all required fields
        molecules = []
        for item in dataset["train"]:
            molecule = {
                "task": item["task"],
                "id": item["id"],
                "smiles": item["smiles"],
                "image_path": item["image_path"]  # This provides the relative path to the image
            }
            molecules.append(molecule)
        
        # Load task prompt from local file (still needed for prompts)
        prompt_path = DATA_DIR / task_name / f"{task_name}_prompt.json"
        task_prompt = load_json_file(prompt_path)
        
        return molecules, task_prompt
    except Exception as e:
        logger.error(f"Error loading data for task {task_name}: {e}")
        raise

def process_molecule_modified(
    client,  # OpenAI or other API client
    molecule: Dict,
    task_name: str,
    task_prompt: Dict,
    repair_prompt: Dict,
    model: str
) -> Dict:
    """
    Modified process_molecule function that works with the new data structure.
    
    Args:
        client: API client instance
        molecule: Molecule data dictionary (now includes image_path)
        task_name: Name of the task
        task_prompt: Task prompt dictionary
        repair_prompt: Main repair prompt dictionary
        model: Model name to use
        
    Returns:
        Dictionary with the results and metadata
    """
    molecule_id = molecule["id"]
    logger.info(f"Processing {task_name} molecule ID: {molecule_id}")
    
    # For local dataset, construct the full image path
    # If you're using the Hugging Face dataset directly, you might need to download the image
    # For now, assuming you still have local access to images:
    image_path = DATA_DIR / molecule["image_path"]
    
    # Alternative: If you want to use the image from the Hugging Face dataset directly:
    # You would need to modify this to download/access the image from the dataset
    
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Rest of the function remains the same as your original
    # Create the API request
    messages = create_repair_request(
        molecule, 
        task_name, 
        task_prompt, 
        repair_prompt, 
        str(image_path)
    )
    
    # Call the API and process results...
    # (rest of your original process_molecule logic)
    
    # The rest of your original process_molecule function would go here
    pass  # Remove this and add your original logic 