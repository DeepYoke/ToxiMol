#!/usr/bin/env python3
"""
评价指标消融实验脚本。

此脚本评估不同评价指标组合对分子毒性修复成功率的影响。
评价指标包括:
1. 毒性端点概率 (必选)
2. QED (药物相似性)
3. SAS (合成可及性)
4. Lipinski违反项数 (药物规则)
5. 相似度 (与原始分子的结构相似度)

脚本将测试所有可能的指标组合(共16种)并保存评估结果。
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import pandas as pd
import copy

# 导入评估模块
from experiments.evaluation.result_evaluator import ResultEvaluator
from experiments.evaluation.evaluation_models import (
    RepairResults,
    ToxicityEndpointResult,
    ToxicityDelta
)
from experiments.evaluation.molecule_utils import (
    validate_smiles,
    calculate_properties,
    calculate_similarity,
    predict_toxicity,
    load_txgemma_model
)
from experiments.evaluation.task_mappings import get_task_mapping

# 评价指标的名称和描述
CRITERIA_NAMES = {
    1: "toxicity_probability",  # 毒性端点概率
    2: "qed",                   # 药物相似性
    3: "sas",                   # 合成可及性
    4: "lipinski",              # 药物规则
    5: "similarity"             # 结构相似度
}

# 评价指标组合
CRITERIA_COMBINATIONS = [
    # 单指标
    {1},
    
    # 两个指标的组合
    {1, 2},
    {1, 3},
    {1, 4},
    {1, 5},
    
    # 三个指标的组合
    {1, 2, 3},
    {1, 2, 4},
    {1, 2, 5},
    {1, 3, 4},
    {1, 3, 5},
    {1, 4, 5},
    
    # 四个指标的组合
    {1, 2, 3, 4},
    {1, 2, 3, 5},
    {1, 2, 4, 5},
    {1, 3, 4, 5},
    
    # 完整的五个指标
    {1, 2, 3, 4, 5}
]

class CriteriaAblationEvaluator(ResultEvaluator):
    """评价指标消融实验评估器"""
    
    def __init__(self, results_dir: str = "results/gpt", criteria_set: Set[int] = None):
        """
        初始化评估器。
        
        Args:
            results_dir: 实验结果目录
            criteria_set: 要评估的指标集合(默认为全部指标)
        """
        super().__init__(results_dir)
        self.criteria_set = criteria_set if criteria_set is not None else {1, 2, 3, 4, 5}
        
        # 创建指标集描述字符串(用于输出)
        criteria_names = [CRITERIA_NAMES[i] for i in sorted(self.criteria_set)]
        self.criteria_description = "+".join(criteria_names)
    
    def _evaluate_repair_criteria(
        self,
        task_type: str,
        original_smiles: str,
        modified_smiles: str
    ) -> RepairResults:
        """
        根据选定的指标组合评估分子是否满足修复标准。
        
        Args:
            task_type: 任务类型
            original_smiles: 原始SMILES
            modified_smiles: 修改后的SMILES
            
        Returns:
            RepairResults: 修复评估结果
        """
        # 获取任务映射和主要端点
        mapping = get_task_mapping(task_type)
        primary_endpoints = mapping.get("primary_endpoints", [])
        
        # 若没有指定端点，使用默认端点
        if not primary_endpoints:
            if task_type == "ames":
                primary_endpoints = ["AMES"]
            elif "herg" in task_type.lower():
                primary_endpoints = ["hERG"]
            else:
                # 通用端点
                primary_endpoints = ["AMES"]
        
        # 计算分子性质
        properties = calculate_properties(modified_smiles)
        qed = properties.get("qed", 0.0)
        sas_score = properties.get("sas_score", 10.0)
        lipinski_violations = properties.get("lipinski_violations", 0)
        
        # 计算相似度
        similarity = calculate_similarity(original_smiles, modified_smiles)
        
        # 预测修改后分子的毒性
        modified_toxicity_endpoints = {}
        for endpoint in primary_endpoints:
            value, probability = predict_toxicity(modified_smiles, endpoint)
            modified_toxicity_endpoints[endpoint] = ToxicityEndpointResult(
                endpoint=endpoint,
                value=value,
                probability=probability
            )
        
        # 对原始分子，假设安全概率很低(0.01)
        original_toxicity_endpoints = {}
        for endpoint in primary_endpoints:
            if endpoint == "LD50_Zhu":
                # 对于LD50，使用低值(100)表示高毒性
                original_toxicity_endpoints[endpoint] = ToxicityEndpointResult(
                    endpoint=endpoint,
                    value="100",
                    probability=0.1  # 略高但仍有毒
                )
            else:
                # 对于分类端点，使用"B"(有毒)和很低的安全概率
                original_toxicity_endpoints[endpoint] = ToxicityEndpointResult(
                    endpoint=endpoint,
                    value="B",
                    probability=0.01  # 非常低的安全概率
                )
        
        # 计算毒性变化
        toxicity_deltas = []
        toxicity_improved = False
        improvement_count = 0
        
        # 获取所有端点的最大安全概率
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
                    # 安全概率增加=改进，无论多小
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
        
        # 按原始框架：任何端点有改进，则认为毒性有改进
        if improvement_count > 0:
            toxicity_improved = True
        
        # 检查是否通过所有标准
        fails = []
        
        # 根据选定的指标组合检查
        
        # 1. 毒性标准 - 安全概率必须 > 0.5 (必选指标)
        if max_safety_prob <= 0.5:
            fails.append("毒性未降至安全水平(安全概率 ≤ 0.5)")
        
        # 2. 药物相似性(QED)
        if 2 in self.criteria_set and qed < 0.5:
            fails.append("药物相似性不足(QED < 0.5)")
        
        # 3. 合成可及性(SAS)
        if 3 in self.criteria_set and sas_score > 6:
            fails.append("合成难度过高(SAS > 6)")
        
        # 4. Lipinski规则(药物规则)
        if 4 in self.criteria_set and lipinski_violations > 1:
            fails.append("Lipinski违反项过多(> 1)")
        
        # 5. 结构相似度
        if 5 in self.criteria_set and similarity < 0.4:
            fails.append("结构相似度过低(< 0.4)")
        
        # 无失败即通过
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

def generate_criteria_description(criteria_set: Set[int]) -> str:
    """
    生成指标组合的描述字符串。
    
    Args:
        criteria_set: 指标集合
        
    Returns:
        str: 描述字符串
    """
    criteria_names = [CRITERIA_NAMES[i] for i in sorted(criteria_set)]
    return "+".join(criteria_names)

def run_criteria_ablation(
    model: str = "claude-3-7-sonnet-20250219",
    results_dir: str = "experiments/gpt/results",
    output_base_dir: str = "experiments/ablation/results/criteria_ablation"
) -> Dict[str, Dict[str, Any]]:
    """
    运行评价指标消融实验。
    
    Args:
        model: 模型名称
        results_dir: 结果目录
        output_base_dir: 输出目录
        
    Returns:
        Dict[str, Dict[str, Any]]: 所有指标组合的评估结果
    """
    # 确保输出目录存在
    output_dir = Path(output_base_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    # 对每个指标组合运行评估
    for criteria_set in CRITERIA_COMBINATIONS:
        criteria_desc = generate_criteria_description(criteria_set)
        print(f"\n评估指标组合: {criteria_desc}")
        
        # 创建评估器
        evaluator = CriteriaAblationEvaluator(results_dir, criteria_set)
        
        # 评估所有任务
        try:
            summaries = evaluator.evaluate_all_results(model, full_evaluation=True)
            
            # 保存评估结果
            criteria_output_dir = output_dir / criteria_desc.replace("+", "_")
            os.makedirs(criteria_output_dir, exist_ok=True)
            
            # 保存详细结果
            result_file = criteria_output_dir / f"{model}_evaluation.json"
            with open(result_file, 'w') as f:
                json.dump(summaries, f, indent=2)
            
            # 创建数据帧并保存CSV
            summary_data = []
            for task, summary in summaries.items():
                if task != "overall":
                    row = {
                        "指标组合": criteria_desc,
                        "任务": task,
                        "原始分子数": summary.get("original_molecule_count", 0),
                        "总修改分子数": summary.get("total_molecules", 0),
                        "有效SMILES数": summary.get("valid_smiles_count", 0),
                        "成功修复数": summary.get("success_count", 0),
                        "有效百分比": f"{summary.get('valid_percentage', 0):.2f}%",
                        "成功百分比": f"{summary.get('success_percentage', 0):.2f}%",
                    }
                    
                    # 添加毒性改进(如果有)
                    if 'toxicity_improved_count' in summary:
                        row["毒性改进数"] = summary["toxicity_improved_count"]
                        row["毒性改进百分比"] = f"{summary['toxicity_improved_percentage']:.2f}%"
                    
                    summary_data.append(row)
            
            # 添加总体行
            if "overall" in summaries:
                overall = summaries["overall"]
                row = {
                    "指标组合": criteria_desc,
                    "任务": "总体",
                    "原始分子数": overall.get("original_molecule_count", 0),
                    "总修改分子数": overall.get("total_molecules", 0),
                    "有效SMILES数": overall.get("valid_smiles_count", 0),
                    "成功修复数": overall.get("success_count", 0),
                    "有效百分比": f"{overall.get('valid_percentage', 0):.2f}%",
                    "成功百分比": f"{overall.get('success_percentage', 0):.2f}%",
                }
                
                # 添加毒性改进(如果有)
                if 'toxicity_improved_count' in overall:
                    row["毒性改进数"] = overall["toxicity_improved_count"]
                    row["毒性改进百分比"] = f"{overall['toxicity_improved_percentage']:.2f}%"
                    
                summary_data.append(row)
            
            # 保存CSV
            df = pd.DataFrame(summary_data)
            csv_file = criteria_output_dir / f"{model}_evaluation.csv"
            df.to_csv(csv_file, index=False)
            
            all_results[criteria_desc] = copy.deepcopy(summaries["overall"])
            
            print(f"评估完成: {criteria_desc}")
            print(f"总体成功率: {summaries['overall']['success_percentage']:.2f}%")
            
        except Exception as e:
            print(f"评估失败: {criteria_desc}: {e}")
    
    # 创建比较报告
    create_comparison_report(all_results, output_dir / f"{model}_comparison.csv")
    
    return all_results

def create_comparison_report(all_results: Dict[str, Dict[str, Any]], output_file: str):
    """
    创建不同指标组合的比较报告。
    
    Args:
        all_results: 所有指标组合的结果
        output_file: 输出文件路径
    """
    # 创建比较数据
    comparison_data = []
    
    for criteria_desc, results in all_results.items():
        row = {
            "指标组合": criteria_desc,
            "原始分子数": results.get("original_molecule_count", 0),
            "总修改分子数": results.get("total_molecules", 0),
            "有效SMILES数": results.get("valid_smiles_count", 0),
            "成功修复数": results.get("success_count", 0),
            "有效百分比": f"{results.get('valid_percentage', 0):.2f}%",
            "成功百分比": f"{results.get('success_percentage', 0):.2f}%",
        }
        
        # 添加毒性改进(如果有)
        if 'toxicity_improved_count' in results:
            row["毒性改进数"] = results["toxicity_improved_count"]
            row["毒性改进百分比"] = f"{results['toxicity_improved_percentage']:.2f}%"
        
        comparison_data.append(row)
    
    # 按指标数量排序(从少到多)
    comparison_data.sort(key=lambda x: len(x["指标组合"].split("+")))
    
    # 保存为CSV
    df = pd.DataFrame(comparison_data)
    df.to_csv(output_file, index=False)
    
    print(f"\n比较报告已保存到: {output_file}")

def main():
    """脚本主入口"""
    # 加载TxGemma模型进行毒性预测
    print("加载TxGemma模型进行毒性预测...")
    load_txgemma_model()
    
    # 运行评价指标消融实验
    print("\n开始进行评价指标消融实验...")
    run_criteria_ablation()
    
    print("\n评价指标消融实验已完成。")

if __name__ == "__main__":
    main() 