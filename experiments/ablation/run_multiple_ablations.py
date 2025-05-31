#!/usr/bin/env python3
"""
批量执行多个模型的消融实验分析

这个脚本依次对多个模型运行毒性阈值消融分析，分析毒性端点预测结果概率大于0.5
但因其他指标不达标而判定为修复失败的分子情况。
"""

import os
import sys
import subprocess
import time
import argparse
from datetime import datetime
from pathlib import Path

# 要分析的模型列表
DEFAULT_MODELS = [
    "claude-3-7-sonnet-20250219",
    "o4-mini",
    "qwen2.5-vl-72b-instruct",
    "InternVL3-78B"
]

def run_ablation_analysis(model, results_dir):
    """运行指定模型的消融分析"""
    print(f"\n{'='*80}")
    print(f"开始分析模型: {model}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # 构建分析命令
    cmd = f"python -m experiments.ablation.toxicity_threshold_analysis --model {model} --results-dir {results_dir}"
    
    try:
        # 执行命令并实时输出结果
        process = subprocess.Popen(
            cmd, 
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # 实时输出日志
        for line in process.stdout:
            print(line, end='')
        
        # 等待进程完成
        process.wait()
        
        if process.returncode == 0:
            print(f"\n{'-'*80}")
            print(f"模型 {model} 分析完成!")
            print(f"{'-'*80}\n")
            return True
        else:
            print(f"\n{'-'*80}")
            print(f"模型 {model} 分析失败，返回代码: {process.returncode}")
            print(f"{'-'*80}\n")
            return False
            
    except Exception as e:
        print(f"\n{'-'*80}")
        print(f"运行模型 {model} 分析时出错: {e}")
        print(f"{'-'*80}\n")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量运行多个模型的消融实验分析")
    
    parser.add_argument(
        "--results-dir", 
        default="experiments/gpt/results",
        help="包含实验结果的目录 (默认: experiments/gpt/results)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"要分析的模型列表 (默认: {', '.join(DEFAULT_MODELS)})"
    )
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    Path("experiments/ablation/results").mkdir(exist_ok=True, parents=True)
    
    print("开始批量运行消融实验分析...")
    print(f"将依次分析以下模型: {', '.join(args.models)}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 记录成功和失败的模型
    successful_models = []
    failed_models = []
    
    # 依次分析每个模型
    for model in args.models:
        success = run_ablation_analysis(model, args.results_dir)
        if success:
            successful_models.append(model)
        else:
            failed_models.append(model)
        
        # 模型之间暂停几秒，确保资源释放
        time.sleep(5)
    
    # 计算总耗时
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    # 打印摘要
    print("\n" + "="*80)
    print("消融实验分析完成摘要")
    print("="*80)
    print(f"总耗时: {hours}小时 {minutes}分钟 {seconds}秒")
    print(f"成功模型 ({len(successful_models)}): {', '.join(successful_models)}")
    print(f"失败模型 ({len(failed_models)}): {', '.join(failed_models)}")
    print("="*80)
    
    # 生成汇总报告（如果所有模型分析都成功）
    if failed_models:
        print("由于存在分析失败的模型，跳过生成汇总报告")
    else:
        # 这里可以添加生成汇总报告的代码
        # 暂未实现，可根据需要扩展
        pass

if __name__ == "__main__":
    main() 