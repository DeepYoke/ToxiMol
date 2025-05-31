#!/usr/bin/env python3
"""
消融实验：分析其他指标都达标但毒性端点预测结果概率小于等于0.5而判定为修复失败的分子

本脚本分析以下指标：
1. 其他指标都达标但毒性修复不达标的Modified Molecules数量及占比
2. 所有修复分子其他指标均达标但毒性均未修复的Original Molecules数量及占比
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Tuple

# 导入评估相关模块
from experiments.evaluation.result_evaluator import ResultEvaluator
from experiments.evaluation.molecule_utils import load_txgemma_model

class NonToxicityThresholdAnalyzer:
    """分析非毒性指标与毒性指标影响的消融实验分析器"""
    
    def __init__(self, results_dir: str = "experiments/gpt/results"):
        """
        初始化分析器
        
        Args:
            results_dir: 包含实验结果的目录
        """
        self.results_dir = Path(results_dir)
        self.evaluator = ResultEvaluator(results_dir)
        
    def analyze_model(self, model: str, full_evaluation: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        分析特定模型的所有任务结果
        
        Args:
            model: 模型名称
            full_evaluation: 是否执行完整评估（包括分子属性计算）
            
        Returns:
            Dict[str, Dict[str, Any]]: 任务名称到分析结果的映射
        """
        # 获取模型目录下的所有任务
        model_dir = self.results_dir / model
        if not model_dir.exists():
            print(f"错误: 模型目录 '{model_dir}' 不存在。")
            return {}
            
        task_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        
        # 分析每个任务
        all_analyses = {}
        for task_dir in task_dirs:
            task = task_dir.name
            print(f"分析任务: {task}")
            task_analysis = self.analyze_task(model, task, full_evaluation)
            all_analyses[task] = task_analysis
        
        # 创建整体分析
        original_molecule_count = sum(analysis.get('original_molecule_count', 0) for analysis in all_analyses.values())
        total_modified_molecules = sum(analysis.get('total_modified_molecules', 0) for analysis in all_analyses.values())
        success_count = sum(analysis.get('success_count', 0) for analysis in all_analyses.values())
        
        nontoxic_fail_modified = sum(analysis.get('nontoxic_fail_modified_count', 0) for analysis in all_analyses.values())
        nontoxic_fail_original = sum(analysis.get('nontoxic_fail_original_count', 0) for analysis in all_analyses.values())
        
        overall = {
            'model': model,
            'original_molecule_count': original_molecule_count,
            'total_modified_molecules': total_modified_molecules,
            'success_count': success_count,
            'success_percentage': (success_count / original_molecule_count * 100) if original_molecule_count > 0 else 0,
            'nontoxic_fail_modified_count': nontoxic_fail_modified,
            'nontoxic_fail_modified_percentage': (nontoxic_fail_modified / total_modified_molecules * 100) if total_modified_molecules > 0 else 0,
            'nontoxic_fail_original_count': nontoxic_fail_original,
            'nontoxic_fail_original_percentage': (nontoxic_fail_original / original_molecule_count * 100) if original_molecule_count > 0 else 0,
            'tasks_completed': len(all_analyses)
        }
        
        all_analyses['overall'] = overall
        
        return all_analyses
    
    def analyze_task(self, model: str, task: str, full_evaluation: bool = True) -> Dict[str, Any]:
        """
        分析特定任务的结果
        
        Args:
            model: 模型名称
            task: 任务名称
            full_evaluation: 是否执行完整评估
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 评估任务结果
        evaluation_results, summary = self.evaluator.evaluate_task_results(model, task, full_evaluation)
        
        # 按原始分子ID分组结果
        results_by_molecule = {}
        for result in evaluation_results:
            if result.molecule_id not in results_by_molecule:
                results_by_molecule[result.molecule_id] = []
            results_by_molecule[result.molecule_id].append(result)
        
        # 计数：其他指标都达标但毒性修复不达标的Modified Molecules
        nontoxic_fail_modified_count = 0
        
        # 计数：所有修复分子其他指标均达标但毒性均未修复的Original Molecules
        nontoxic_fail_original_count = 0
        
        # 分析每个原始分子的修复结果
        for molecule_id, molecule_results in results_by_molecule.items():
            # 筛选有效的SMILES结果
            valid_results = [r for r in molecule_results if r.details['validation']['valid_smiles']]
            if not valid_results:
                continue
                
            # 检查每个Modified Molecule
            nontoxic_success_but_toxic_fail = []
            
            for result in valid_results:
                # 判断其他指标是否全部达标
                other_criteria_passed = True
                fail_reasons = []
                
                if 'properties' in result.details:
                    properties = result.details['properties']
                    # QED < 0.5
                    if 'qed' in properties and properties['qed'] < 0.5:
                        other_criteria_passed = False
                        fail_reasons.append("QED < 0.5")
                    
                    # SAS > 6
                    if 'sas_score' in properties and properties['sas_score'] > 6:
                        other_criteria_passed = False
                        fail_reasons.append("SAS > 6")
                    
                    # Lipinski violations > 1
                    if 'lipinski_violations' in properties and properties['lipinski_violations'] > 1:
                        other_criteria_passed = False
                        fail_reasons.append("Lipinski violations > 1")
                    
                    # Similarity < 0.4
                    if 'similarity' in properties and properties['similarity'] < 0.4:
                        other_criteria_passed = False
                        fail_reasons.append("Similarity < 0.4")
                
                # 获取毒性安全概率
                max_safety_prob = 0.0
                if result.toxicity_endpoints:
                    max_safety_prob = max(endpoint.probability for endpoint in result.toxicity_endpoints.values())
                
                # 检查是否是其他指标都达标但毒性未改善导致的失败
                if other_criteria_passed and max_safety_prob <= 0.5 and not result.success:
                    nontoxic_fail_modified_count += 1
                    nontoxic_success_but_toxic_fail.append(result)
            
            # 检查原始分子的所有Modified Molecules是否都是其他指标达标但毒性未改善
            if len(nontoxic_success_but_toxic_fail) == len(valid_results) and len(valid_results) > 0:
                nontoxic_fail_original_count += 1
        
        # 创建分析结果
        analysis = {
            'task': task,
            'model': model,
            'original_molecule_count': summary['original_molecule_count'],
            'total_modified_molecules': summary['total_molecules'],
            'valid_smiles_count': summary['valid_smiles_count'],
            'success_count': summary['success_count'],
            'success_percentage': (summary['success_count'] / summary['original_molecule_count'] * 100) if summary['original_molecule_count'] > 0 else 0,
            'nontoxic_fail_modified_count': nontoxic_fail_modified_count,
            'nontoxic_fail_modified_percentage': (nontoxic_fail_modified_count / summary['total_molecules'] * 100) if summary['total_molecules'] > 0 else 0,
            'nontoxic_fail_original_count': nontoxic_fail_original_count,
            'nontoxic_fail_original_percentage': (nontoxic_fail_original_count / summary['original_molecule_count'] * 100) if summary['original_molecule_count'] > 0 else 0
        }
        
        return analysis
    
    def save_analysis_results(self, model: str, analyses: Dict[str, Dict[str, Any]]) -> str:
        """
        保存分析结果到文件
        
        Args:
            model: 模型名称
            analyses: 分析结果
            
        Returns:
            str: 保存的文件路径
        """
        # 创建输出目录
        output_dir = Path("experiments/ablation/results") / model
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 保存JSON结果
        json_file = output_dir / "nontoxicity_threshold_analysis.json"
        with open(json_file, 'w') as f:
            json.dump(analyses, f, indent=2)
        
        # 创建CSV数据
        csv_data = []
        for task, analysis in analyses.items():
            if task != "overall":
                row = {
                    "Task": task,
                    "Original Molecules": analysis["original_molecule_count"],
                    "Total Modified Molecules": analysis["total_modified_molecules"],
                    "Valid SMILES": analysis.get("valid_smiles_count", 0),
                    "Successful Repairs": analysis.get("success_count", 0),
                    "Success %": f"{analysis.get('success_percentage', 0):.2f}%",
                    "Other Fixed But Toxic Failed Modified": analysis["nontoxic_fail_modified_count"],
                    "Other Fixed But Toxic Failed Modified %": f"{analysis['nontoxic_fail_modified_percentage']:.2f}%",
                    "Other Fixed But Toxic Failed Original": analysis["nontoxic_fail_original_count"],
                    "Other Fixed But Toxic Failed Original %": f"{analysis['nontoxic_fail_original_percentage']:.2f}%"
                }
                csv_data.append(row)
        
        # 添加整体行
        if "overall" in analyses:
            overall = analyses["overall"]
            row = {
                "Task": "OVERALL",
                "Original Molecules": overall["original_molecule_count"],
                "Total Modified Molecules": overall["total_modified_molecules"],
                "Valid SMILES": sum(analyses[task].get("valid_smiles_count", 0) for task in analyses if task != "overall"),
                "Successful Repairs": overall["success_count"],
                "Success %": f"{overall['success_percentage']:.2f}%",
                "Other Fixed But Toxic Failed Modified": overall["nontoxic_fail_modified_count"],
                "Other Fixed But Toxic Failed Modified %": f"{overall['nontoxic_fail_modified_percentage']:.2f}%",
                "Other Fixed But Toxic Failed Original": overall["nontoxic_fail_original_count"],
                "Other Fixed But Toxic Failed Original %": f"{overall['nontoxic_fail_original_percentage']:.2f}%"
            }
            csv_data.append(row)
        
        # 保存CSV
        csv_file = output_dir / "nontoxicity_threshold_analysis.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        
        print(f"分析结果已保存到: {output_dir}")
        return str(json_file)

def main():
    """脚本主入口点"""
    parser = argparse.ArgumentParser(description="消融实验：分析其他指标都达标但毒性修复不达标的分子")
    
    parser.add_argument(
        "--results-dir", 
        default="experiments/gpt/results",
        help="包含实验结果的目录 (默认: experiments/gpt/results)"
    )
    
    parser.add_argument(
        "--model", 
        default="claude-3-7-sonnet-20250219",
        help="要分析的模型名称 (默认: claude-3-7-sonnet-20250219)"
    )
    
    args = parser.parse_args()
    
    # 检查结果目录是否存在
    if not os.path.exists(args.results_dir):
        print(f"错误: 结果目录 '{args.results_dir}' 不存在。")
        sys.exit(1)
    
    # 加载TxGemma模型用于毒性预测
    print("加载TxGemma模型用于毒性预测...")
    load_txgemma_model()
    
    # 创建分析器并分析结果
    analyzer = NonToxicityThresholdAnalyzer(args.results_dir)
    analyses = analyzer.analyze_model(args.model, full_evaluation=True)
    
    if not analyses:
        print(f"错误: 找不到模型 '{args.model}' 的结果。")
        sys.exit(1)
    
    # 保存分析结果
    analyzer.save_analysis_results(args.model, analyses)
    
    # 打印整体摘要
    if "overall" in analyses:
        overall = analyses["overall"]
        print("\n消融实验分析摘要:")
        print(f"模型: {args.model}")
        print(f"原始分子数量: {overall['original_molecule_count']}")
        print(f"修复分子总数: {overall['total_modified_molecules']}")
        print(f"其他指标都达标但毒性修复不达标的修复分子数量: {overall['nontoxic_fail_modified_count']} ({overall['nontoxic_fail_modified_percentage']:.2f}%)")
        print(f"所有修复分子其他指标均达标但毒性均未修复的原始分子数量: {overall['nontoxic_fail_original_count']} ({overall['nontoxic_fail_original_percentage']:.2f}%)")
        print(f"完成的任务数: {overall['tasks_completed']}")
    
    print("\n分析完成。")

if __name__ == "__main__":
    main() 