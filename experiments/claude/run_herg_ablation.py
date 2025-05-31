#!/usr/bin/env python3
"""
Toxicity Repair 消融实验脚本：使用统一的hERG Prompt

本脚本进行hERG相关任务的消融实验，让herg、herg_central和herg_karim三个任务
都使用herg_central_prompt.json中的提示（更通用的prompt），而其它逻辑保持不变。
通过这种方式评估统一提示对毒性修复效果的影响。
"""

import os
import json
import base64
import argparse
import time
import requests
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("claude_herg_ablation")

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "Experimental_dataset"
# 为消融实验创建专门的结果目录
RESULTS_DIR = Path(__file__).resolve().parent / "ablation_results" / "unified_herg_prompt"

# 智增增 API settings
ZZZ_API_KEY = "sk-zk2bf6cbabe8c8f27ccbb928b5e3bde3cb767cab83731789"
ZZZ_BASE_URL = "https://api.zhizengzeng.com/v1"

# 消融实验中使用的Claude模型
AVAILABLE_MODELS = [
    "claude-3-7-sonnet-20250219"
]

# 消融实验仅针对以下hERG相关任务
HERG_TASKS = ["herg", "herg_central", "herg_karim"]

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
    为消融实验加载分子数据和任务提示。
    所有hERG相关任务都使用herg_central_prompt.json中的提示。
    
    Args:
        task_name: Name of the task (e.g., "herg")
        
    Returns:
        Tuple of (molecules list, task prompt dictionary)
    """
    try:
        # 加载分子数据，使用原始任务的分子数据
        molecules_path = DATA_DIR / task_name / f"{task_name}.json"
        molecules = load_json_file(molecules_path)
        
        # 消融实验：对所有hERG相关任务，都使用herg_central的提示
        if task_name in HERG_TASKS:
            # 统一使用herg_central的提示
            prompt_path = DATA_DIR / "herg_central" / "herg_central_prompt.json"
            logger.info(f"使用统一的herg_central提示为任务: {task_name}")
        else:
            # 其他任务使用原始提示
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

def create_repair_request(
    molecule: Dict, 
    task_name: str,
    task_prompt: Dict,
    repair_prompt: Dict,
    image_path: str,
    model: str
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
        
    Returns:
        List of message dictionaries for the API request
    """
    # Generate specific instruction for the molecule
    specific_instruction = get_specific_prompt(task_name, molecule, task_prompt)
    
    # Encode the molecule image
    encoded_image = encode_image(image_path)
    
    # Create the system message from the repair prompt
    system_content = (
        f"You are a {repair_prompt['agent_role']}. "
        f"{repair_prompt['task_overview']} "
        f"Follow these guidelines for working with the molecular structure image: "
        f"{'; '.join(repair_prompt['image_integration']['usage_guidelines'])}. "
        f"Your output must strictly follow this format: {repair_prompt['output_format']['structure']}"
    )
    
    # Enhanced prompt to get better responses
    user_content = (
        f"IMPORTANT TASK: Modify the following molecule to reduce its {task_prompt['task_description']} while "
        f"maintaining its therapeutic properties.\n\n"
        f"SMILES: {molecule['smiles']}\n\n"
        f"{specific_instruction}\n\n"
        f"YOU MUST PROVIDE AT LEAST ONE MODIFIED VERSION. DO NOT RETURN EMPTY RESPONSES.\n\n"
        f"RESPOND EXACTLY IN THIS FORMAT AND NOTHING ELSE:\n"
        f"MODIFIED_SMILES: [modified_smiles_1];[modified_smiles_2];[modified_smiles_3]\n\n"
        f"Use semicolons to separate multiple SMILES. Provide 1-3 modifications."
    )
    
    # Claude模型支持多模态输入
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
        api_key: 智增增 API key
        
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
    
    # Create the API request
    messages = create_repair_request(
        molecule, 
        task_name, 
        task_prompt, 
        repair_prompt, 
        str(image_path),
        model
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
    model: str,
    api_key: str = ZZZ_API_KEY,
    molecule_limit: Optional[int] = None,
    molecules_ids: Optional[List[int]] = None
) -> List[Dict]:
    """
    Run the experiment for a specific task.
    
    Args:
        task_name: Name of the task to run
        model: Model name to use
        api_key: 智增增 API key
        molecule_limit: Maximum number of molecules to process (optional)
        molecules_ids: Specific molecule IDs to process (optional)
        
    Returns:
        List of result dictionaries
    """
    # 创建特定任务的输出目录
    output_dir = RESULTS_DIR / model / task_name
    os.makedirs(output_dir, exist_ok=True)
    
    # 清理旧的结果文件
    cleanup_old_results(output_dir, task_name)
    
    # 加载任务数据（这里是消融实验的关键，通过load_task_data函数实现统一prompt）
    molecules, task_prompt = load_task_data(task_name)
    repair_prompt = load_repair_prompt()
    
    # 如果是hERG相关任务，记录使用了统一的提示
    if task_name in HERG_TASKS:
        logger.info(f"消融实验: 任务 {task_name} 使用统一的herg_central提示")
    
    # 根据需要过滤分子
    if molecules_ids:
        molecules = [m for m in molecules if m["id"] in molecules_ids]
    
    if molecule_limit and molecule_limit > 0:
        molecules = molecules[:molecule_limit]
    
    logger.info(f"运行任务 {task_name} 共有 {len(molecules)} 个分子")
    
    # 处理每个分子
    all_results = []
    for molecule in molecules:
        result = process_molecule(
            molecule,
            task_name,
            task_prompt,
            repair_prompt,
            model,
            api_key
        )
        all_results.append(result)
        
        # 添加短暂延迟避免速率限制
        time.sleep(1)
    
    # 创建组合结果
    combined_results = {
        "task_name": task_name,
        "model": model,
        "total_molecules": len(molecules),
        "success_count": sum(1 for r in all_results if "error" not in r),
        "error_count": sum(1 for r in all_results if "error" in r),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "using_unified_herg_prompt": task_name in HERG_TASKS,  # 标记是否使用了统一提示
        "results": all_results
    }
    
    # 保存组合结果到单个文件
    output_file = output_dir / f"{task_name}_results.json"
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    logger.info(f"保存 {task_name} 的组合结果到 {output_file}")
    
    return all_results

def run_herg_ablation(
    model: str,
    api_key: str = ZZZ_API_KEY,
    molecule_limit: Optional[int] = None
) -> Dict[str, List[Dict]]:
    """
    运行hERG相关任务的消融实验
    
    Args:
        model: 要使用的模型名称
        api_key: 智增增 API key
        molecule_limit: 每个任务处理的分子数量上限（可选）
        
    Returns:
        Dict[str, List[Dict]]: 任务名称到结果列表的映射
    """
    all_results = {}
    
    # 仅运行hERG相关任务
    for task_name in HERG_TASKS:
        logger.info(f"开始hERG消融实验任务: {task_name} (使用统一的herg_central提示)")
        task_results = run_task(task_name, model, api_key, molecule_limit)
        all_results[task_name] = task_results
    
    # 保存总体摘要
    overall_summary = {
        "model": model,
        "experiment_type": "unified_herg_prompt_ablation",
        "tasks_completed": len(HERG_TASKS),
        "total_molecules": sum(len(results) for results in all_results.values()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": "消融实验：所有hERG相关任务使用统一的herg_central提示"
    }
    
    summary_dir = RESULTS_DIR / model
    os.makedirs(summary_dir, exist_ok=True)
    
    with open(summary_dir / "ablation_summary.json", 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    return all_results

def main():
    """脚本的主入口点"""
    parser = argparse.ArgumentParser(description="运行hERG任务的消融实验（使用统一的herg_central提示）")
    
    parser.add_argument(
        "--task", 
        choices=HERG_TASKS + ["all"], 
        default="all",
        help="要运行的任务 (默认: 所有hERG相关任务)"
    )
    
    parser.add_argument(
        "--model", 
        choices=AVAILABLE_MODELS,
        default="claude-3-7-sonnet-20250219",
        help="要使用的Claude模型 (默认: claude-3-7-sonnet-20250219)"
    )
    
    parser.add_argument(
        "--api-key", 
        default=ZZZ_API_KEY,
        help="智增增 API key (默认: 使用预定义的密钥)"
    )
    
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="限制每个任务的分子数量 (默认: 无限制)"
    )
    
    parser.add_argument(
        "--molecule-ids", 
        type=int, 
        nargs="+",
        help="要处理的特定分子ID (默认: 所有分子)"
    )
    
    args = parser.parse_args()
    
    # 确保结果目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    logger.info(f"开始hERG消融实验，使用模型: {args.model} 通过智增增 API")
    logger.info("消融实验: 所有hERG相关任务使用统一的herg_central提示")
    
    if args.task == "all":
        run_herg_ablation(args.model, args.api_key, args.limit)
    else:
        run_task(args.task, args.model, args.api_key, args.limit, args.molecule_ids)
    
    logger.info("消融实验完成")

if __name__ == "__main__":
    main() 