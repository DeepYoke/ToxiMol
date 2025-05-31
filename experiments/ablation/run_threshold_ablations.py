#!/usr/bin/env python3
"""
批量运行不同毒性安全概率阈值的消融实验分析

这个脚本使用不同毒性安全概率阈值(0.6, 0.7, 0.8, 0.9)依次对claude-3-7-sonnet-20250219模型
进行消融实验分析，检查阈值变化对修复成功率的影响。
"""

import os
import sys
import subprocess
import time
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# 要分析的毒性安全概率阈值
THRESHOLDS = [0.6, 0.7, 0.8, 0.9]

def run_threshold_analysis(threshold, model="claude-3-7-sonnet-20250219"):
    """运行特定阈值的消融分析"""
    print(f"\n{'='*80}")
    print(f"开始分析模型 {model} 使用阈值: {threshold}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # 构建分析命令
    cmd = f"python -m experiments.ablation.toxicity_probability_threshold_analysis --model {model} --threshold {threshold}"
    
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
            print(f"阈值 {threshold} 的分析完成!")
            print(f"{'-'*80}\n")
            return True
        else:
            print(f"\n{'-'*80}")
            print(f"阈值 {threshold} 的分析失败，返回代码: {process.returncode}")
            print(f"{'-'*80}\n")
            return False
            
    except Exception as e:
        print(f"\n{'-'*80}")
        print(f"运行阈值 {threshold} 的分析时出错: {e}")
        print(f"{'-'*80}\n")
        return False

def generate_summary_report(model="claude-3-7-sonnet-20250219"):
    """
    生成所有阈值结果的汇总报告
    
    Args:
        model: 模型名称
    """
    print(f"\n{'='*80}")
    print(f"生成汇总报告...")
    print(f"{'='*80}\n")
    
    # 输出目录
    output_dir = Path("experiments/ablation/results") / model
    
    # 收集所有阈值的分析结果
    all_threshold_results = {}
    
    for threshold in THRESHOLDS:
        threshold_str = str(threshold).replace(".", "_")
        json_file = output_dir / f"toxicity_threshold_{threshold_str}_analysis.json"
        
        if not json_file.exists():
            print(f"警告: 阈值 {threshold} 的结果文件不存在: {json_file}")
            continue
        
        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
                if "overall" in results:
                    all_threshold_results[threshold] = results
        except Exception as e:
            print(f"读取阈值 {threshold} 的结果时出错: {e}")
    
    if not all_threshold_results:
        print("错误: 没有找到任何有效的阈值分析结果!")
        return False
    
    # 创建汇总CSV - 比较不同阈值下总体成功率
    summary_data = []
    
    for threshold, results in all_threshold_results.items():
        overall = results["overall"]
        row = {
            "Toxicity Threshold": threshold,
            "Original Molecules": overall["original_molecule_count"],
            "Total Modified Molecules": overall["total_modified_molecules"],
            "Successful Repairs": overall["success_count"],
            "Success %": f"{overall['success_percentage']:.2f}%",
            "Success % (numeric)": overall['success_percentage'],
            "Toxicity Improved": overall["toxicity_improved_count"],
            "Toxicity Improved %": f"{overall['toxicity_improved_percentage']:.2f}%", 
            "Toxicity Improved % (numeric)": overall['toxicity_improved_percentage']
        }
        summary_data.append(row)
    
    # 按阈值排序
    summary_data.sort(key=lambda x: x["Toxicity Threshold"])
    
    # 创建任务级汇总数据
    task_summary_data = []
    
    # 获取所有任务
    all_tasks = set()
    for results in all_threshold_results.values():
        all_tasks.update([task for task in results.keys() if task != "overall"])
    
    # 对每个任务比较不同阈值下的成功率
    for task in sorted(all_tasks):
        for threshold, results in all_threshold_results.items():
            if task in results:
                task_results = results[task]
                row = {
                    "Task": task,
                    "Toxicity Threshold": threshold,
                    "Original Molecules": task_results["original_molecule_count"],
                    "Success %": f"{task_results['success_percentage']:.2f}%",
                    "Success % (numeric)": task_results['success_percentage'],
                    "Toxicity Improved %": f"{task_results['toxicity_improved_percentage']:.2f}%",
                    "Toxicity Improved % (numeric)": task_results['toxicity_improved_percentage']
                }
                task_summary_data.append(row)
    
    # 保存汇总CSV
    summary_df = pd.DataFrame(summary_data)
    task_summary_df = pd.DataFrame(task_summary_data)
    
    # 去除数值列，以提高可读性
    summary_df_readable = summary_df.drop(columns=["Success % (numeric)", "Toxicity Improved % (numeric)"])
    
    summary_file = output_dir / "threshold_comparison_summary.csv"
    task_summary_file = output_dir / "threshold_comparison_by_task.csv"
    
    summary_df_readable.to_csv(summary_file, index=False)
    task_summary_df.to_csv(task_summary_file, index=False)
    
    print(f"汇总报告已保存到:")
    print(f"- 总体比较: {summary_file}")
    print(f"- 任务级比较: {task_summary_file}")
    
    # 打印摘要结果
    print("\n不同阈值的修复成功率对比:")
    for row in summary_data:
        print(f"阈值 {row['Toxicity Threshold']}: 成功率 {row['Success %']}, 毒性改善率 {row['Toxicity Improved %']}")
    
    return True

def main():
    """主函数"""
    model = "claude-3-7-sonnet-20250219"
    
    # 确保输出目录存在
    Path("experiments/ablation/results").mkdir(exist_ok=True, parents=True)
    output_dir = Path("experiments/ablation/results") / model
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("开始运行多阈值毒性消融实验分析...")
    print(f"将依次分析以下阈值: {', '.join(str(t) for t in THRESHOLDS)}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 记录成功和失败的阈值
    successful_thresholds = []
    failed_thresholds = []
    
    # 依次分析每个阈值
    for threshold in THRESHOLDS:
        success = run_threshold_analysis(threshold, model)
        if success:
            successful_thresholds.append(threshold)
        else:
            failed_thresholds.append(threshold)
        
        # 阈值之间暂停几秒，确保资源释放
        time.sleep(5)
    
    # 生成汇总报告
    if successful_thresholds:
        generate_summary_report(model)
    
    # 计算总耗时
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    # 打印摘要
    print("\n" + "="*80)
    print("多阈值消融实验分析完成摘要")
    print("="*80)
    print(f"总耗时: {hours}小时 {minutes}分钟 {seconds}秒")
    print(f"成功阈值 ({len(successful_thresholds)}): {', '.join(str(t) for t in successful_thresholds)}")
    print(f"失败阈值 ({len(failed_thresholds)}): {', '.join(str(t) for t in failed_thresholds)}")
    print("="*80)
    
    if successful_thresholds:
        print(f"\n结果保存在: {output_dir}")
        print(f"汇总报告: {output_dir}/threshold_comparison_summary.csv")

if __name__ == "__main__":
    main() 