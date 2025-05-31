#!/usr/bin/env python3
"""
运行其他指标都达标但毒性修复不达标的消融实验分析

这个脚本对claude-3-7-sonnet-20250219模型运行非毒性阈值消融分析，
分析其他指标都达标但由于毒性端点预测结果概率小于等于0.5而判定为修复失败的分子情况。
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

def run_analysis():
    """运行非毒性阈值分析"""
    print(f"\n{'='*80}")
    print(f"开始分析模型: claude-3-7-sonnet-20250219")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # 构建分析命令
    cmd = f"python -m experiments.ablation.nontoxicity_threshold_analysis --model claude-3-7-sonnet-20250219"
    
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
            print(f"分析完成!")
            print(f"{'-'*80}\n")
            return True
        else:
            print(f"\n{'-'*80}")
            print(f"分析失败，返回代码: {process.returncode}")
            print(f"{'-'*80}\n")
            return False
            
    except Exception as e:
        print(f"\n{'-'*80}")
        print(f"运行分析时出错: {e}")
        print(f"{'-'*80}\n")
        return False

def main():
    """主函数"""
    # 确保输出目录存在
    Path("experiments/ablation/results/claude-3-7-sonnet-20250219").mkdir(exist_ok=True, parents=True)
    
    print("开始运行非毒性阈值消融实验分析...")
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行分析
    success = run_analysis()
    
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
    print(f"状态: {'成功' if success else '失败'}")
    print("="*80)
    
    if success:
        print("\n结果保存在: experiments/ablation/results/claude-3-7-sonnet-20250219/nontoxicity_threshold_analysis.csv")
    
if __name__ == "__main__":
    main() 