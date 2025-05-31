#!/usr/bin/env python3
"""
消融实验：分析不同毒性端点安全概率阈值对修复成功率的影响

本脚本分析以下指标：
1. 使用不同毒性安全概率阈值(0.6, 0.7, 0.8, 0.9)时的修复成功率
2. 在各个阈值下的详细修复统计信息
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
from experiments.evaluation.evaluation_models import RepairResults, ToxicityEndpointResult

class ToxicityProbabilityThresholdAnalyzer:
    """分析不同毒性安全概率阈值的消融实验分析器"""
    
    def __init__(self, results_dir: str = "experiments/gpt/results"):
        """
        初始化分析器
        
        Args:
            results_dir: 包含实验结果的目录
        """
        self.results_dir = Path(results_dir)
        self.evaluator = ResultEvaluator(results_dir)
        
    def analyze_model_with_threshold(self, model: str, toxicity_threshold: float = 0.6, full_evaluation: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        使用指定的毒性安全概率阈值分析特定模型的所有任务结果
        
        Args:
            model: 模型名称
            toxicity_threshold: 毒性安全概率阈值，默认为0.6
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
            print(f"分析任务: {task} (阈值: {toxicity_threshold})")
            task_analysis = self.analyze_task_with_threshold(model, task, toxicity_threshold, full_evaluation)
            all_analyses[task] = task_analysis
        
        # 创建整体分析
        original_molecule_count = sum(analysis.get('original_molecule_count', 0) for analysis in all_analyses.values())
        total_modified_molecules = sum(analysis.get('total_modified_molecules', 0) for analysis in all_analyses.values())
        success_count = sum(analysis.get('success_count', 0) for analysis in all_analyses.values())
        valid_count = sum(analysis.get('valid_smiles_count', 0) for analysis in all_analyses.values())
        toxicity_improved_count = sum(analysis.get('toxicity_improved_count', 0) for analysis in all_analyses.values())
        
        overall = {
            'model': model,
            'toxicity_threshold': toxicity_threshold,
            'original_molecule_count': original_molecule_count,
            'total_modified_molecules': total_modified_molecules,
            'valid_smiles_count': valid_count,
            'success_count': success_count,
            'success_percentage': (success_count / original_molecule_count * 100) if original_molecule_count > 0 else 0,
            'toxicity_improved_count': toxicity_improved_count,
            'toxicity_improved_percentage': (toxicity_improved_count / original_molecule_count * 100) if original_molecule_count > 0 else 0,
            'tasks_completed': len(all_analyses)
        }
        
        all_analyses['overall'] = overall
        
        return all_analyses
    
    def analyze_task_with_threshold(self, model: str, task: str, toxicity_threshold: float = 0.6, full_evaluation: bool = True) -> Dict[str, Any]:
        """
        使用指定的毒性安全概率阈值分析特定任务的结果
        
        Args:
            model: 模型名称
            task: 任务名称
            toxicity_threshold: 毒性安全概率阈值，默认为0.6
            full_evaluation: 是否执行完整评估
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 评估任务结果 - 获取原始评估结果
        evaluation_results_original, summary = self.evaluator.evaluate_task_results(model, task, full_evaluation)
        
        # 按原始分子ID分组结果
        results_by_molecule = {}
        for result in evaluation_results_original:
            if result.molecule_id not in results_by_molecule:
                results_by_molecule[result.molecule_id] = []
            results_by_molecule[result.molecule_id].append(result)
        
        # 新的成功计数
        success_count = 0
        
        # 毒性改善的分子计数
        toxicity_improved_count = 0
        
        # 重新评估每个原始分子的修复结果
        for molecule_id, molecule_results in results_by_molecule.items():
            # 筛选有效的SMILES结果
            valid_results = [r for r in molecule_results if r.details['validation']['valid_smiles']]
            if not valid_results:
                continue
                
            # 检查每个Modified Molecule，使用新的毒性阈值
            molecule_success = False
            molecule_toxicity_improved = False
            
            for result in valid_results:
                # 重新评估毒性改善状态
                new_toxicity_improved = False
                
                # 获取最大安全概率
                max_safety_prob = 0.0
                if result.toxicity_endpoints:
                    max_safety_prob = max(endpoint.probability for endpoint in result.toxicity_endpoints.values())
                
                # 使用新阈值评估毒性是否改善
                if max_safety_prob > toxicity_threshold:
                    new_toxicity_improved = True
                
                # 评估其他指标
                other_criteria_passed = True
                if 'properties' in result.details:
                    properties = result.details['properties']
                    # QED < 0.5
                    if 'qed' in properties and properties['qed'] < 0.5:
                        other_criteria_passed = False
                    
                    # SAS > 6
                    if 'sas_score' in properties and properties['sas_score'] > 6:
                        other_criteria_passed = False
                    
                    # Lipinski violations > 1
                    if 'lipinski_violations' in properties and properties['lipinski_violations'] > 1:
                        other_criteria_passed = False
                    
                    # Similarity < 0.4
                    if 'similarity' in properties and properties['similarity'] < 0.4:
                        other_criteria_passed = False
                
                # 判断是否成功修复 - 使用新阈值
                new_success = new_toxicity_improved and other_criteria_passed
                
                # 更新分子级别的成功状态
                if new_success:
                    molecule_success = True
                
                # 更新分子级别的毒性改善状态
                if new_toxicity_improved:
                    molecule_toxicity_improved = True
            
            # 统计成功修复的原始分子数量
            if molecule_success:
                success_count += 1
            
            # 统计毒性改善的原始分子数量
            if molecule_toxicity_improved:
                toxicity_improved_count += 1
        
        # 创建分析结果
        analysis = {
            'task': task,
            'model': model,
            'toxicity_threshold': toxicity_threshold,
            'original_molecule_count': summary['original_molecule_count'],
            'total_modified_molecules': summary['total_molecules'],
            'valid_smiles_count': summary['valid_smiles_count'],
            'success_count': success_count,
            'success_percentage': (success_count / summary['original_molecule_count'] * 100) if summary['original_molecule_count'] > 0 else 0,
            'toxicity_improved_count': toxicity_improved_count,
            'toxicity_improved_percentage': (toxicity_improved_count / summary['original_molecule_count'] * 100) if summary['original_molecule_count'] > 0 else 0
        }
        
        return analysis
    
    def save_analysis_results(self, model: str, analyses: Dict[str, Dict[str, Any]], toxicity_threshold: float) -> str:
        """
        保存分析结果到文件
        
        Args:
            model: 模型名称
            analyses: 分析结果
            toxicity_threshold: 毒性安全概率阈值
            
        Returns:
            str: 保存的文件路径
        """
        # 创建输出目录
        output_dir = Path("experiments/ablation/results") / model
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 保存JSON结果
        threshold_str = str(toxicity_threshold).replace(".", "_")
        json_file = output_dir / f"toxicity_threshold_{threshold_str}_analysis.json"
        with open(json_file, 'w') as f:
            json.dump(analyses, f, indent=2)
        
        # 创建CSV数据
        csv_data = []
        for task, analysis in analyses.items():
            if task != "overall":
                row = {
                    "Task": task,
                    "Toxicity Threshold": analysis["toxicity_threshold"],
                    "Original Molecules": analysis["original_molecule_count"],
                    "Total Modified Molecules": analysis["total_modified_molecules"],
                    "Valid SMILES": analysis.get("valid_smiles_count", 0),
                    "Successful Repairs": analysis.get("success_count", 0),
                    "Success %": f"{analysis.get('success_percentage', 0):.2f}%",
                    "Toxicity Improved": analysis.get("toxicity_improved_count", 0),
                    "Toxicity Improved %": f"{analysis.get('toxicity_improved_percentage', 0):.2f}%"
                }
                csv_data.append(row)
        
        # 添加整体行
        if "overall" in analyses:
            overall = analyses["overall"]
            row = {
                "Task": "OVERALL",
                "Toxicity Threshold": overall["toxicity_threshold"],
                "Original Molecules": overall["original_molecule_count"],
                "Total Modified Molecules": overall["total_modified_molecules"],
                "Valid SMILES": overall.get("valid_smiles_count", 0),
                "Successful Repairs": overall["success_count"],
                "Success %": f"{overall['success_percentage']:.2f}%",
                "Toxicity Improved": overall["toxicity_improved_count"],
                "Toxicity Improved %": f"{overall['toxicity_improved_percentage']:.2f}%"
            }
            csv_data.append(row)
        
        # 保存CSV
        csv_file = output_dir / f"toxicity_threshold_{threshold_str}_analysis.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        
        print(f"阈值 {toxicity_threshold} 的分析结果已保存到: {output_dir}")
        return str(json_file)

def main():
    """脚本主入口点"""
    parser = argparse.ArgumentParser(description="消融实验：分析不同毒性安全概率阈值对修复成功率的影响")
    
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
    
    parser.add_argument(
        "--threshold", 
        type=float,
        default=0.6,
        help="毒性安全概率阈值 (默认: 0.6)"
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
    analyzer = ToxicityProbabilityThresholdAnalyzer(args.results_dir)
    analyses = analyzer.analyze_model_with_threshold(args.model, args.threshold, full_evaluation=True)
    
    if not analyses:
        print(f"错误: 找不到模型 '{args.model}' 的结果。")
        sys.exit(1)
    
    # 保存分析结果
    analyzer.save_analysis_results(args.model, analyses, args.threshold)
    
    # 打印整体摘要
    if "overall" in analyses:
        overall = analyses["overall"]
        print("\n消融实验分析摘要:")
        print(f"模型: {args.model}")
        print(f"毒性安全概率阈值: {args.threshold}")
        print(f"原始分子数量: {overall['original_molecule_count']}")
        print(f"修复分子总数: {overall['total_modified_molecules']}")
        print(f"成功修复的分子数量: {overall['success_count']} ({overall['success_percentage']:.2f}%)")
        print(f"毒性改善的分子数量: {overall['toxicity_improved_count']} ({overall['toxicity_improved_percentage']:.2f}%)")
        print(f"完成的任务数: {overall['tasks_completed']}")
    
    print("\n分析完成。")

if __name__ == "__main__":
    main() 