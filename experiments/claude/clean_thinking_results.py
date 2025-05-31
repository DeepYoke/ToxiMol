#!/usr/bin/env python3
"""
清洗Claude-3-7-Sonnet-Thinking模型的结果文件，移除思考过程，使格式与GPT-4.1保持一致。

此脚本将处理experiments/claude/results/claude-3-7-sonnet-thinking目录下的所有任务结果。
"""

import os
import json
import re
from pathlib import Path
import shutil
import sys

# 设置源目录和输出目录
SOURCE_DIR = Path('experiments/claude/results/claude-3-7-sonnet-thinking')
OUTPUT_DIR = Path('experiments/claude/results/claude-3-7-sonnet-thinking-cleaned')

def clean_raw_response(raw_response):
    """
    清洗原始响应，删除思考过程，保留MODIFIED_SMILES部分。
    
    Args:
        raw_response: 原始响应字符串
        
    Returns:
        str: 清洗后的响应字符串
    """
    # 如果没有思考标签，则直接返回
    if "<think>" not in raw_response:
        return raw_response
    
    # 提取MODIFIED_SMILES部分
    if "MODIFIED_SMILES:" in raw_response:
        # 找到MODIFIED_SMILES:后面的内容
        matches = re.search(r'MODIFIED_SMILES:(.*?)($|</think>)', raw_response, re.DOTALL)
        if matches:
            smiles_part = matches.group(1).strip()
            # 移除可能的思考标签
            smiles_part = smiles_part.replace("</think>", "").strip()
            # 检查是否有占位符，如[modified_smiles_1]
            if re.search(r'\[modified_smiles_\d+\]', smiles_part):
                # 返回一个空的SMILES字符串
                return "MODIFIED_SMILES: "
            return f"MODIFIED_SMILES: {smiles_part}"
    
    # 如果没有找到MODIFIED_SMILES，则提取所有非思考部分
    parts = re.split(r'<think>.*?</think>', raw_response, flags=re.DOTALL)
    cleaned = ' '.join(part.strip() for part in parts if part.strip())
    
    # 如果清洗后的结果包含MODIFIED_SMILES，则只保留这一部分
    if "MODIFIED_SMILES:" in cleaned:
        matches = re.search(r'MODIFIED_SMILES:(.*?)($)', cleaned, re.DOTALL)
        if matches:
            return f"MODIFIED_SMILES: {matches.group(1).strip()}"
    
    return cleaned or "MODIFIED_SMILES: "

def clean_modified_smiles(modified_smiles_list):
    """
    清洗修改后的SMILES列表，移除无效项。
    
    Args:
        modified_smiles_list: SMILES字符串列表
        
    Returns:
        list: 清洗后的SMILES列表
    """
    cleaned_list = []
    for smiles in modified_smiles_list:
        # 移除占位符和明显无效的SMILES
        if not re.search(r'\[modified_smiles_\d+\]', smiles) and not smiles.endswith("</think>"):
            # 移除可能的思考标签
            smiles = smiles.replace("</think>", "").strip()
            if smiles:
                cleaned_list.append(smiles)
    
    return cleaned_list or [""]  # 如果没有有效SMILES，返回空字符串列表

def clean_result_file(file_path, output_path=None):
    """
    清洗单个结果文件。
    
    Args:
        file_path: 输入文件路径
        output_path: 输出文件路径，如果为None则覆盖原文件
        
    Returns:
        bool: 是否成功处理
    """
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确保数据格式正确
        if "results" not in data:
            print(f"警告: {file_path} 中没有找到results字段")
            return False
        
        # 处理每个结果项
        for result in data["results"]:
            # 清洗raw_response
            if "raw_response" in result:
                result["raw_response"] = clean_raw_response(result["raw_response"])
            
            # 清洗modified_smiles
            if "modified_smiles" in result:
                result["modified_smiles"] = clean_modified_smiles(result["modified_smiles"])
        
        # 保存到输出文件
        output_file = output_path or file_path
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"成功处理: {file_path}")
        return True
    
    except Exception as e:
        print(f"处理 {file_path} 时出错: {str(e)}")
        return False

def process_all_files():
    """
    处理所有任务文件夹中的结果文件。
    """
    # 检查源目录是否存在
    if not SOURCE_DIR.exists():
        print(f"错误: 源目录 {SOURCE_DIR} 不存在")
        return
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # 处理overall_summary.json
    if (SOURCE_DIR / "overall_summary.json").exists():
        shutil.copy(SOURCE_DIR / "overall_summary.json", OUTPUT_DIR / "overall_summary.json")
        print("已复制 overall_summary.json")
    
    # 获取所有任务文件夹
    task_dirs = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]
    
    success_count = 0
    error_count = 0
    
    for task_dir in task_dirs:
        task_name = task_dir.name
        print(f"\n处理任务: {task_name}")
        
        # 创建对应的输出任务目录
        output_task_dir = OUTPUT_DIR / task_name
        output_task_dir.mkdir(exist_ok=True, parents=True)
        
        # 查找结果文件
        result_files = list(task_dir.glob("*_results.json"))
        
        for result_file in result_files:
            output_file = output_task_dir / result_file.name
            if clean_result_file(result_file, output_file):
                success_count += 1
            else:
                error_count += 1
    
    print(f"\n处理完成: 成功 {success_count} 文件, 失败 {error_count} 文件")
    print(f"清洗后的结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    # 添加命令行参数处理
    if len(sys.argv) > 1 and sys.argv[1] == "--overwrite":
        # 覆盖原始文件
        print("警告: 将覆盖原始文件!")
        confirm = input("确定要继续吗? (y/n): ")
        if confirm.lower() == 'y':
            for task_dir in [d for d in SOURCE_DIR.iterdir() if d.is_dir()]:
                for result_file in task_dir.glob("*_results.json"):
                    clean_result_file(result_file)
            print("覆盖完成!")
        else:
            print("已取消操作")
    else:
        # 创建新的清洗后文件
        process_all_files() 