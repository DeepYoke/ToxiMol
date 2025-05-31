#!/usr/bin/env python3
"""
评价指标消融实验命令行脚本。

此脚本提供命令行接口来运行评价指标消融实验。
"""
import os
import sys
import argparse
from pathlib import Path

from experiments.ablation.run_criteria_ablation import run_criteria_ablation
from experiments.evaluation.molecule_utils import load_txgemma_model

def main():
    """脚本主入口"""
    parser = argparse.ArgumentParser(description="运行评价指标消融实验")
    
    parser.add_argument(
        "--model", 
        default="claude-3-7-sonnet-20250219",
        help="要评估的模型名称 (默认: claude-3-7-sonnet-20250219)"
    )
    
    parser.add_argument(
        "--results-dir", 
        default="experiments/gpt/results",
        help="包含实验结果的目录 (默认: experiments/gpt/results)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="experiments/ablation/results/criteria_ablation",
        help="结果输出目录 (默认: experiments/ablation/results/criteria_ablation)"
    )
    
    args = parser.parse_args()
    
    # 确保结果目录存在
    if not os.path.exists(args.results_dir):
        print(f"错误: 结果目录 '{args.results_dir}' 不存在。")
        sys.exit(1)
    
    # 确保模型结果存在
    model_dir = Path(args.results_dir) / args.model
    if not os.path.exists(model_dir):
        print(f"错误: 模型结果目录 '{model_dir}' 不存在。")
        sys.exit(1)
    
    # 加载TxGemma模型
    print("加载TxGemma模型进行毒性预测...")
    load_txgemma_model()
    
    # 运行评价指标消融实验
    print(f"\n开始进行评价指标消融实验: 模型 = {args.model}")
    
    run_criteria_ablation(
        model=args.model,
        results_dir=args.results_dir,
        output_base_dir=args.output_dir
    )
    
    print(f"\n评价指标消融实验已完成。结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 