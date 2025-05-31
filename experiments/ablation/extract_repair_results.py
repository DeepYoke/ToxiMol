#!/usr/bin/env python3
"""
提取毒性修复结果分子脚本

此脚本分析评估结果，提取成功修复和失败修复的分子SMILES，
并按照任务分类保存到CSV文件中。

用法:
    python -m experiments.txgemma_evaluation.extract_repair_results \
        --results-dir experiments/gpt/results \
        --model claude-3-7-sonnet-20250219 \
        --output-dir experiments/gpt/repair_analysis

会生成:
    {output_dir}/{model_name}/{task_name}/
        - successful_repairs.json
        - failed_repairs.json
"""
import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from experiments.evaluation.result_evaluator import (
    ResultEvaluator,
    analyze_experiment_results
)
from experiments.evaluation.molecule_utils import load_txgemma_model
from experiments.evaluation.evaluation_models import EvaluationResult


def extract_repair_results(
    model: str,
    results_dir: str,
    output_dir: str,
    use_existing_eval: bool = True
) -> None:
    """
    提取并保存成功和失败的修复分子数据

    Args:
        model: 模型名称
        results_dir: 结果目录
        output_dir: 输出目录
        use_existing_eval: 是否使用现有评估结果
    """
    print(f"分析模型 {model} 的修复结果...")
    
    # 加载TxGemma模型进行毒性预测
    print("加载TxGemma模型...")
    load_txgemma_model()
    
    # 创建评估器
    evaluator = ResultEvaluator(results_dir)
    
    # 获取模型目录下的所有任务
    model_dir = Path(results_dir) / model
    task_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    
    # 创建基础输出目录
    base_output_dir = Path(output_dir) / model
    base_output_dir.mkdir(exist_ok=True, parents=True)
    
    # 对每个任务处理结果
    for task_dir in task_dirs:
        task = task_dir.name
        print(f"处理任务: {task}")
        
        # 评估任务结果
        results, summary = evaluator.evaluate_task_results(model, task, True)
        
        # 计算期望的总修复分子数（原始分子数 × 3）
        expected_count = summary.get("original_molecule_count", 0) * 3
        
        # 创建任务输出目录
        task_output_dir = base_output_dir / task
        task_output_dir.mkdir(exist_ok=True, parents=True)
        
        # 提取并组织结果
        successful_repairs, failed_repairs, invalid_repairs = organize_results(results)
        
        # 计算实际总数
        actual_count = len(successful_repairs) + len(failed_repairs) + len(invalid_repairs)
        
        # 保存成功修复的分子
        successful_json = task_output_dir / "successful_repairs.json"
        save_repair_results_json(successful_repairs, successful_json)
        print(f"- 成功修复分子 {len(successful_repairs)} 个，已保存至 {successful_json}")
        
        # 保存失败修复的分子
        failed_json = task_output_dir / "failed_repairs.json"
        save_repair_results_json(failed_repairs, failed_json)
        print(f"- 失败修复分子 {len(failed_repairs)} 个，已保存至 {failed_json}")
        
        # 保存无效SMILES分子
        invalid_json = task_output_dir / "invalid_repairs.json"
        save_repair_results_json(invalid_repairs, invalid_json)
        print(f"- 无效SMILES分子 {len(invalid_repairs)} 个，已保存至 {invalid_json}")
        
        # 保存摘要信息
        summary_file = task_output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump({
                "task": task,
                "model": model,
                "original_molecules": summary.get("original_molecule_count", 0),
                "expected_repairs": expected_count,  # 期望的修复分子总数（原始分子 × 3）
                "actual_repairs": actual_count,      # 实际处理的修复分子总数
                "successful_count": len(successful_repairs),  # 成功的修复分子数
                "failed_count": len(failed_repairs),          # 失败的修复分子数 
                "invalid_count": len(invalid_repairs),        # 无效SMILES的分子数
                "repair_success_rate": len(successful_repairs) / actual_count * 100 if actual_count > 0 else 0,  # 按修复分子计算的成功率
                "original_success_rate": summary.get('success_percentage', 0)  # 按原始分子计算的成功率（至少有一个修复成功）
            }, f, indent=2)


def organize_results(results: List[EvaluationResult]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    组织评估结果为成功和失败的修复列表

    Args:
        results: 评估结果列表

    Returns:
        Tuple[List[Dict], List[Dict], List[Dict]]: 成功、失败和无效的修复结果
    """
    # 按分子ID分组结果
    results_by_molecule = {}
    for result in results:
        if result.molecule_id not in results_by_molecule:
            results_by_molecule[result.molecule_id] = []
        results_by_molecule[result.molecule_id].append(result)
    
    successful_repairs = []
    failed_repairs = []
    invalid_repairs = []
    
    total_expected = len(results_by_molecule) * 3  # 每个原始分子应该有3个修复方案
    
    # 遍历每个原始分子的所有修复结果
    for molecule_id, molecule_results in results_by_molecule.items():
        # 提取原始SMILES (所有结果中的原始SMILES应该相同)
        original_smiles = molecule_results[0].original_smiles if molecule_results else ""
        
        # 处理每个修复结果，包括无效SMILES
        for result in molecule_results:
            modified_smiles = result.modified_smiles
            
            # 构建结果数据
            repair_data = {
                "molecule_id": molecule_id,
                "task": result.task,
                "original_smiles": original_smiles,
                "modified_smiles": modified_smiles,
                "message": result.message
            }
            
            # 检查是否有效SMILES
            is_valid_smiles = result.details['validation']['valid_smiles']
            
            if not is_valid_smiles:
                # 无效SMILES
                repair_data["valid_smiles"] = False
                repair_data["error"] = result.details['validation'].get('error', "未知错误")
                invalid_repairs.append(repair_data)
                continue
                
            # 添加有效SMILES的详细属性
            repair_data["valid_smiles"] = True
            repair_data["success"] = result.success
            repair_data["qed"] = result.details.get("properties", {}).get("qed", 0)
            repair_data["sas_score"] = result.details.get("properties", {}).get("sas_score", 0)
            repair_data["lipinski_violations"] = result.details.get("properties", {}).get("lipinski_violations", 0)
            repair_data["similarity"] = result.details.get("properties", {}).get("similarity", 0)
            repair_data["toxicity_improved"] = result.details.get("toxicity", {}).get("improved", False)
            
            # 获取毒性端点结果
            toxicity_data = {}
            for endpoint, data in result.toxicity_endpoints.items():
                toxicity_data[endpoint] = {
                    "value": data.value,
                    "probability": data.probability
                }
            repair_data["toxicity_endpoints"] = toxicity_data
            
            # 根据成功/失败添加到相应列表
            if result.success:
                successful_repairs.append(repair_data)
            else:
                failed_repairs.append(repair_data)
    
    # 验证总数
    total_actual = len(successful_repairs) + len(failed_repairs) + len(invalid_repairs)
    if total_actual < total_expected:
        print(f"警告: 实际修复分子总数({total_actual})少于预期({total_expected})，可能有缺失的数据")
    
    return successful_repairs, failed_repairs, invalid_repairs


def save_repair_results_json(repair_data: List[Dict], output_file: Path) -> None:
    """
    将修复结果保存为JSON文件

    Args:
        repair_data: 修复结果数据
        output_file: 输出文件路径
    """
    if not repair_data:
        # 如果没有数据，创建一个空的JSON文件
        with open(output_file, "w") as f:
            json.dump({"message": "没有数据", "results": []}, f, ensure_ascii=False, indent=2)
        return
    
    # 处理结果数据，确保JSON可序列化
    processed_data = []
    
    for data in repair_data:
        # 复制数据，并展开toxicity_endpoints
        processed_item = data.copy()
        
        # 处理toxicity_endpoints
        toxicity_endpoints = processed_item.pop("toxicity_endpoints", {})
        processed_item["toxicity"] = {}
        
        for endpoint, endpoint_data in toxicity_endpoints.items():
            processed_item["toxicity"][endpoint] = endpoint_data
            
        processed_data.append(processed_item)
    
    # 保存为JSON
    with open(output_file, "w") as f:
        json.dump({
            "count": len(processed_data),
            "results": processed_data
        }, f, ensure_ascii=False, indent=2)


def main():
    """主入口点"""
    parser = argparse.ArgumentParser(description="提取和保存毒性修复分子结果")
    
    parser.add_argument(
        "--results-dir", 
        default="experiments/gpt/results",
        help="包含实验结果的目录 (默认: experiments/gpt/results)"
    )
    
    parser.add_argument(
        "--model", 
        required=True,
        help="要分析的模型名称"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="experiments/gpt/repair_analysis",
        help="输出目录，将保存JSON格式的结果 (默认: experiments/gpt/repair_analysis)"
    )
    
    parser.add_argument(
        "--use-existing", 
        action="store_true",
        help="使用现有评估结果而不是重新评估"
    )
    
    args = parser.parse_args()
    
    # 检查结果目录是否存在
    if not os.path.exists(args.results_dir):
        print(f"错误: 结果目录 '{args.results_dir}' 不存在")
        sys.exit(1)
    
    # 检查模型目录是否存在
    model_dir = Path(args.results_dir) / args.model
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录 '{model_dir}' 不存在")
        sys.exit(1)
    
    # 提取修复结果
    extract_repair_results(
        args.model,
        args.results_dir,
        args.output_dir,
        args.use_existing
    )
    
    print("\n提取完成！结果已保存到 {}/{}".format(args.output_dir, args.model))


if __name__ == "__main__":
    main() 