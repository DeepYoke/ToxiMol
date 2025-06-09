 #!/usr/bin/env python3
"""
Toxicity Repair Experiment Runner using InternVL3-8B model.

This script runs toxicity repair experiments on molecules using InternVL3-8B model.
It can process single tasks or run batch experiments across multiple toxicity datasets.
"""

import os
import json
import base64
import argparse
import time
import math
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, AutoConfig, TextIteratorStreamer
from threading import Thread
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("toxicity_repair")

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "Experimental_dataset"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# InternVL3 model constants
DEFAULT_MODEL_PATH = "OpenGVLab/InternVL3-38B"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# All available tasks
AVAILABLE_TASKS = [
    "ames", "clintox", "carcinogens_lagunin", "dili", "herg", 
    "herg_central", "herg_karim", "ld50_zhu", "skin_reaction", 
    "tox21", "toxcast"
]

class InternVL3Agent:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.device_map = self._split_model(model_path)
        
        logger.info(f"Loading InternVL3 model from {model_path}")
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=self.device_map
        ).eval()
        
        logger.info("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            use_fast=False
        )
        
        self.transform = self._build_transform(input_size=448)
        logger.info("Model and tokenizer loaded successfully")

    def _split_model(self, model_path):
        """Split model across available GPUs."""
        device_map = {}
        world_size = torch.cuda.device_count()
        # world_size = 8
        
        if world_size <= 0:
            logger.warning("No CUDA devices found, using CPU only (this will be slow)")
            return "cpu"
            
        logger.info(f"Found {world_size} CUDA devices")
        
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            num_layers = config.llm_config.num_hidden_layers
            
            # Since the first GPU will be used for ViT, treat it as half a GPU
            num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
            num_layers_per_gpu = [num_layers_per_gpu] * world_size
            num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
            
            layer_cnt = 0
            for i, num_layer in enumerate(num_layers_per_gpu):
                for j in range(num_layer):
                    device_map[f'language_model.model.layers.{layer_cnt}'] = i
                    layer_cnt += 1
                    
            device_map['vision_model'] = 0
            device_map['mlp1'] = 0
            device_map['language_model.model.tok_embeddings'] = 0
            device_map['language_model.model.embed_tokens'] = 0
            device_map['language_model.output'] = 0
            device_map['language_model.model.norm'] = 0
            device_map['language_model.model.rotary_emb'] = 0
            device_map['language_model.lm_head'] = 0
            device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
            
            return device_map
        except Exception as e:
            logger.error(f"Error splitting model: {e}")
            logger.info("Falling back to automatic device mapping")
            return "auto"

    def _build_transform(self, input_size=448):
        """Build image transformation pipeline."""
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Find the closest aspect ratio from target_ratios to the input aspect ratio."""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """Preprocess image into multiple tiles according to aspect ratio."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find the closest aspect ratio to the target
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # Calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # Split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
            
        return processed_images

    def load_image(self, image_path, input_size=448, max_num=12):
        """Load and preprocess an image file for model input."""
        try:
            image = Image.open(image_path).convert('RGB')
            images = self._dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [self.transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values.to(torch.bfloat16).cuda() if torch.cuda.is_available() else pixel_values.to(torch.bfloat16)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

    def generate_completion(self, system_prompt, user_prompt, image_path=None, generation_config=None):
        """Generate a completion using the InternVL3 model."""
        if generation_config is None:
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            
        try:
            # Load image if provided
            pixel_values = None
            if image_path:
                pixel_values = self.load_image(image_path)
                
            # Add image placeholder to user prompt if image is provided
            if pixel_values is not None:
                if '<image>' not in user_prompt:
                    user_prompt = '<image>\n' + user_prompt
                    
            history = None
            if system_prompt:
                history = [{"role": "system", "content": system_prompt}]
                
            # Get the response
            response = self.model.chat(
                self.tokenizer, 
                pixel_values, 
                user_prompt, 
                generation_config,
                history=history
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            raise

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
    Load the main repair prompt.
    
    Returns:
        Repair prompt dictionary
    """
    repair_prompt_path = DATA_DIR / "repair_prompt.json"
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

def create_user_prompt(
    molecule: Dict, 
    task_name: str,
    task_prompt: Dict
) -> str:
    """
    Create the user prompt for the model.
    
    Args:
        molecule: Molecule data dictionary
        task_name: Name of the task
        task_prompt: Task prompt dictionary
        
    Returns:
        Formatted user prompt string
    """
    # Generate specific instruction for the molecule
    specific_instruction = get_specific_prompt(task_name, molecule, task_prompt)
    
    # Create the user prompt
    user_prompt = (
        f"Task: Modify the following molecule to reduce its {task_prompt['task_description']} while "
        f"maintaining its therapeutic properties.\n\n"
        f"SMILES: {molecule['smiles']}\n\n"
        f"{specific_instruction}\n\n"
        f"IMPORTANT: Provide 1-3 modified versions of this molecule following EXACTLY the format: "
        f"'MODIFIED_SMILES: smiles1;smiles2;smiles3'. Use semicolons to separate multiple SMILES. "
        f"No explanations or other text should be included."
    )
    
    return user_prompt

def create_system_prompt(repair_prompt: Dict) -> str:
    """
    Create the system prompt for the model.
    
    Args:
        repair_prompt: Main repair prompt dictionary
        
    Returns:
        Formatted system prompt string
    """
    # Create the system message from the repair prompt
    system_prompt = (
        f"You are a {repair_prompt['agent_role']}. "
        f"{repair_prompt['task_overview']} "
        f"Follow these guidelines for working with the molecular structure image: "
        f"{'; '.join(repair_prompt['image_integration']['usage_guidelines'])}. "
        f"Your output must strictly follow this format: {repair_prompt['output_format']['structure']}"
    )
    
    return system_prompt

def extract_results(response_text: str) -> Dict:
    """
    Parse the model response to extract the modified SMILES.
    
    Args:
        response_text: Raw text response from the model
        
    Returns:
        Dictionary containing the parsed results
    """
    # Initialize results
    results = {
        "modified_smiles": [],
        "raw_response": response_text
    }
    
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
            if smiles_part.lower().strip() == "none":
                results["modified_smiles"] = []
            else:
                results["modified_smiles"] = [smiles_part.strip()]
    
    return results

def process_molecule(
    agent: InternVL3Agent,
    molecule: Dict,
    task_name: str,
    task_prompt: Dict,
    repair_prompt: Dict,
    model: str,
    generation_config: Dict = None
) -> Dict:
    """
    Process a single molecule and return the results.
    
    Args:
        agent: InternVL3Agent instance
        molecule: Molecule data dictionary
        task_name: Name of the task
        task_prompt: Task prompt dictionary
        repair_prompt: Main repair prompt dictionary
        model: Model name to use
        generation_config: Generation configuration
        
    Returns:
        Dictionary with the results and metadata
    """
    molecule_id = molecule["id"]
    logger.info(f"Processing {task_name} molecule ID: {molecule_id}")
    
    # Get the image path
    image_path = DATA_DIR / task_name / "image" / f"{molecule_id}.png"
    
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Create the system and user prompts
    system_prompt = create_system_prompt(repair_prompt)
    user_prompt = create_user_prompt(molecule, task_name, task_prompt)
    
    # Set default generation config if not provided
    if generation_config is None:
        generation_config = dict(max_new_tokens=1024, do_sample=True, temperature=0.5)
    
    # Call the model
    try:
        response = agent.generate_completion(system_prompt, user_prompt, str(image_path), generation_config)
        
        # Parse the results
        results = extract_results(response)
        
        # Add metadata
        results["task"] = molecule["task"] if "task" in molecule else task_name
        results["molecule_id"] = molecule_id
        results["original_smiles"] = molecule["smiles"]
        results["model"] = model
        
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
    model_path: str,
    molecule_limit: Optional[int] = None,
    molecules_ids: Optional[List[int]] = None,
    generation_config: Dict = None
) -> List[Dict]:
    """
    Run the experiment for a specific task.
    
    Args:
        task_name: Name of the task to run
        model_path: Path to the model
        molecule_limit: Maximum number of molecules to process (optional)
        molecules_ids: Specific molecule IDs to process (optional)
        generation_config: Generation configuration
        
    Returns:
        List of result dictionaries
    """
    # Create model agent
    agent = InternVL3Agent(model_path)
    
    # Extract model name from path
    model_name = model_path.split('/')[-1]
    
    # Create output directory
    output_dir = RESULTS_DIR / model_name / task_name
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
    
    logger.info(f"Running task {task_name} with {len(molecules)} molecules")
    
    # Process each molecule
    all_results = []
    for molecule in molecules:
        result = process_molecule(
            agent,
            molecule,
            task_name,
            task_prompt,
            repair_prompt,
            model_name,
            generation_config
        )
        all_results.append(result)
        
        # Add a short delay to avoid potential issues
        time.sleep(1)
    
    # Create combined results
    combined_results = {
        "task_name": task_name,
        "model": model_name,
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
    model_path: str,
    molecule_limit: Optional[int] = None,
    generation_config: Dict = None
) -> Dict[str, List[Dict]]:
    """
    Run the experiment for all available tasks.
    
    Args:
        model_path: Path to the model
        molecule_limit: Maximum number of molecules to process per task (optional)
        generation_config: Generation configuration
        
    Returns:
        Dictionary mapping task names to lists of result dictionaries
    """
    all_results = {}
    
    # Extract model name from path
    model_name = model_path.split('/')[-1]
    
    for task_name in AVAILABLE_TASKS:
        logger.info(f"Starting task: {task_name}")
        task_results = run_task(task_name, model_path, molecule_limit, None, generation_config)
        all_results[task_name] = task_results
        # 清理GPU显存
        if torch.cuda.is_available():
            logger.info("Clearing GPU memory after task completion")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            logger.info(f"GPU memory cleared. Current memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    
    # Save overall summary
    overall_summary = {
        "model": model_name,
        "tasks_completed": len(AVAILABLE_TASKS),
        "total_molecules": sum(len(results) for results in all_results.values()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_dir = RESULTS_DIR / model_name
    os.makedirs(summary_dir, exist_ok=True)
    
    with open(summary_dir / "overall_summary.json", 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    return all_results

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run toxicity repair experiments using InternVL3 model")
    
    parser.add_argument(
        "--task", 
        choices=AVAILABLE_TASKS + ["all"], 
        default="all",
        help="Task to run (default: all)"
    )
    
    parser.add_argument(
        "--model-path", 
        default=DEFAULT_MODEL_PATH,
        help=f"Path to InternVL3 model (default: {DEFAULT_MODEL_PATH})"
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
    
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.5,
        help="Temperature for text generation (default: 0.5)"
    )
    
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=1024,
        help="Maximum number of tokens to generate (default: 1024)"
    )
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Set generation config
    generation_config = {
        "max_new_tokens": args.max_tokens,
        "do_sample": True,
        "temperature": args.temperature
    }
    
    logger.info(f"Starting experiment with model: {args.model_path}")
    
    if args.task == "all":
        run_all_tasks(args.model_path, args.limit, generation_config)
    else:
        run_task(args.task, args.model_path, args.limit, args.molecule_ids, generation_config)
    
    logger.info("Experiment completed")

if __name__ == "__main__":
    main()