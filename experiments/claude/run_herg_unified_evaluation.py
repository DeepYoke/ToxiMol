#!/usr/bin/env python3
"""
统一评估hERG消融实验脚本

这个脚本一次性对三个hERG相关任务(herg, herg_central, herg_karim)进行评估，
并将结果整合在 experiments/claude/ablation_results/unified_herg_prompt/evaluation 目录下。
"""

import os
import sys
import json
import shutil
import subprocess
import time
from pathlib import Path
from datetime import datetime
import argparse

# hERG相关任务
HERG_TASKS = ["herg", "herg_central", "herg_karim"]

# 常量定义
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "ablation_results" / "unified_herg_prompt"
OUTPUT_DIR = RESULTS_DIR / "evaluation"

def run_task_evaluation(task, model, results_dir, full_evaluation=True):
    """
    针对单个任务运行评估
    
    Args:
        task: 任务名称
        model: 模型名称
        results_dir: 结果目录
        full_evaluation: 是否执行完整评估
        
    Returns:
        bool: 是否成功
    """
    print(f"\n{'='*80}")
    print(f"评估任务: {task}")
    print(f"模型: {model}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    cmd = [
        "python", "-m", "experiments.evaluation.run_evaluation",
        "--results-dir", str(results_dir),
        "--model", model,
        "--task", task
    ]
    
    if full_evaluation:
        cmd.append("--full")
    
    try:
        # 运行评估命令并实时输出结果
        process = subprocess.Popen(
            cmd,
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
            print(f"任务 {task} 评估成功!")
            print(f"{'-'*80}\n")
            return True
        else:
            print(f"\n{'-'*80}")
            print(f"任务 {task} 评估失败，返回代码: {process.returncode}")
            print(f"{'-'*80}\n")
            return False
    
    except Exception as e:
        print(f"\n{'-'*80}")
        print(f"运行任务 {task} 评估时出错: {e}")
        print(f"{'-'*80}\n")
        return False

def run_overall_evaluation(model, results_dir, full_evaluation=True):
    """
    运行整体评估（所有任务）
    
    Args:
        model: 模型名称
        results_dir: 结果目录
        full_evaluation: 是否执行完整评估
        
    Returns:
        bool: 是否成功
    """
    print(f"\n{'='*80}")
    print(f"运行整体评估")
    print(f"模型: {model}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    cmd = [
        "python", "-m", "experiments.evaluation.run_evaluation",
        "--results-dir", str(results_dir),
        "--model", model
    ]
    
    if full_evaluation:
        cmd.append("--full")
    
    try:
        # 运行评估命令并实时输出结果
        process = subprocess.Popen(
            cmd,
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
            print(f"整体评估成功!")
            print(f"{'-'*80}\n")
            return True
        else:
            print(f"\n{'-'*80}")
            print(f"整体评估失败，返回代码: {process.returncode}")
            print(f"{'-'*80}\n")
            return False
    
    except Exception as e:
        print(f"\n{'-'*80}")
        print(f"运行整体评估时出错: {e}")
        print(f"{'-'*80}\n")
        return False

def collect_evaluation_results(model, output_dir):
    """
    收集并整合评估结果
    
    Args:
        model: 模型名称
        output_dir: 输出目录
        
    Returns:
        bool: 是否成功
    """
    try:
        # 创建输出目录
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 源文件目录
        source_dir = RESULTS_DIR / model / "evaluation"
        
        if not source_dir.exists():
            print(f"错误: 找不到评估结果目录 {source_dir}")
            return False
        
        # 复制评估结果文件
        for file in source_dir.glob("*.json"):
            target_path = output_dir / file.name
            shutil.copy(file, target_path)
            print(f"复制文件: {file} 到 {target_path}")
        
        # 创建一个总结文件，包含每个任务的主要评估指标
        try:
            # 加载整体评估摘要
            with open(source_dir / "evaluation_summary.json", "r") as f:
                overall_summary = json.load(f)
            
            # 加载任务评估结果
            with open(source_dir / "task_evaluations.json", "r") as f:
                task_evaluations = json.load(f)
            
            # 创建HERG比较摘要
            herg_comparison = {
                "model": model,
                "experiment_type": "unified_herg_prompt_ablation",
                "tasks": {}
            }
            
            # 为每个hERG任务提取关键指标
            for task in HERG_TASKS:
                if task in task_evaluations:
                    task_summary = task_evaluations[task]
                    herg_comparison["tasks"][task] = {
                        "total_molecules": task_summary.get("total_molecules", 0),
                        "valid_smiles_count": task_summary.get("valid_smiles_count", 0),
                        "valid_percentage": task_summary.get("valid_percentage", 0),
                        "success_count": task_summary.get("success_count", 0),
                        "success_percentage": task_summary.get("success_percentage", 0),
                        "toxicity_improved_count": task_summary.get("toxicity_improved_count", 0),
                        "toxicity_improved_percentage": task_summary.get("toxicity_improved_percentage", 0)
                    }
            
            # 添加整体结果
            herg_comparison["overall"] = {
                "total_molecules": sum(task_data.get("total_molecules", 0) for task_data in herg_comparison["tasks"].values()),
                "valid_smiles_count": sum(task_data.get("valid_smiles_count", 0) for task_data in herg_comparison["tasks"].values()),
                "success_count": sum(task_data.get("success_count", 0) for task_data in herg_comparison["tasks"].values()),
                "toxicity_improved_count": sum(task_data.get("toxicity_improved_count", 0) for task_data in herg_comparison["tasks"].values())
            }
            
            # 计算百分比
            if herg_comparison["overall"]["total_molecules"] > 0:
                herg_comparison["overall"]["valid_percentage"] = (
                    herg_comparison["overall"]["valid_smiles_count"] / 
                    herg_comparison["overall"]["total_molecules"] * 100
                )
                herg_comparison["overall"]["success_percentage"] = (
                    herg_comparison["overall"]["success_count"] / 
                    herg_comparison["overall"]["total_molecules"] * 100
                )
                herg_comparison["overall"]["toxicity_improved_percentage"] = (
                    herg_comparison["overall"]["toxicity_improved_count"] / 
                    herg_comparison["overall"]["total_molecules"] * 100
                )
            
            # 保存HERG比较摘要
            with open(output_dir / "herg_comparison_summary.json", "w") as f:
                json.dump(herg_comparison, f, indent=2)
            
            print(f"已创建HERG比较摘要: {output_dir / 'herg_comparison_summary.json'}")
            
            return True
        
        except Exception as e:
            print(f"创建HERG比较摘要时出错: {e}")
            return False
        
    except Exception as e:
        print(f"收集评估结果时出错: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="统一评估hERG消融实验结果")
    
    parser.add_argument(
        "--model", 
        default="claude-3-7-sonnet-20250219",
        help="要评估的模型名称 (默认: claude-3-7-sonnet-20250219)"
    )
    
    parser.add_argument(
        "--skip-task-eval", 
        action="store_true",
        help="跳过单独任务评估，仅运行整体评估"
    )
    
    parser.add_argument(
        "--skip-overall-eval", 
        action="store_true",
        help="跳过整体评估，仅运行单独任务评估"
    )
    
    args = parser.parse_args()
    
    # 确保结果目录存在
    if not RESULTS_DIR.exists() or not (RESULTS_DIR / args.model).exists():
        print(f"错误: 找不到结果目录 {RESULTS_DIR / args.model}")
        sys.exit(1)
    
    # 开始计时
    start_time = time.time()
    
    print("\n开始统一评估hERG消融实验...")
    print(f"模型: {args.model}")
    print(f"结果目录: {RESULTS_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 创建评估目录
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    task_success_count = 0
    
    # 评估各个任务
    if not args.skip_task_eval:
        for task in HERG_TASKS:
            success = run_task_evaluation(task, args.model, RESULTS_DIR, full_evaluation=True)
            if success:
                task_success_count += 1
    
    # 运行整体评估
    overall_success = False
    if not args.skip_overall_eval:
        overall_success = run_overall_evaluation(args.model, RESULTS_DIR, full_evaluation=True)
    
    # 收集评估结果
    results_collected = collect_evaluation_results(args.model, OUTPUT_DIR)
    
    # 计算总耗时
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    # 打印摘要
    print("\n" + "="*80)
    print("统一评估完成摘要")
    print("="*80)
    print(f"总耗时: {hours}小时 {minutes}分钟 {seconds}秒")
    
    if not args.skip_task_eval:
        print(f"任务评估成功: {task_success_count}/{len(HERG_TASKS)}")
    
    if not args.skip_overall_eval:
        print(f"整体评估: {'成功' if overall_success else '失败'}")
    
    print(f"结果收集: {'成功' if results_collected else '失败'}")
    print("="*80)
    
    if results_collected:
        print(f"\n统一评估结果保存在: {OUTPUT_DIR}")
    
if __name__ == "__main__":
    main() 