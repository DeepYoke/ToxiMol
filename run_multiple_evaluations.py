#!/usr/bin/env python3
"""
批量执行多个模型的评估

这个脚本依次对多个模型运行毒性修复评估，使用python -m 格式调用以避免导入问题。
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# 要评估的模型列表 - 已移除完成的InternVL3-8B
MODELS = [
    "moonshot-v1-128k-vision-preview",
    "o1",
    "o3",
    "o4-mini",
    "Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-72b-instruct"
]

def run_evaluation(model):
    """运行指定模型的评估"""
    print(f"\n{'='*80}")
    print(f"开始评估模型: {model}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # 构建评估命令
    cmd = f"python -m experiments.evaluation.run_evaluation --model {model} --full"
    
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
            print(f"模型 {model} 评估完成!")
            print(f"{'-'*80}\n")
            return True
        else:
            print(f"\n{'-'*80}")
            print(f"模型 {model} 评估失败，返回代码: {process.returncode}")
            print(f"{'-'*80}\n")
            return False
            
    except Exception as e:
        print(f"\n{'-'*80}")
        print(f"运行模型 {model} 评估时出错: {e}")
        print(f"{'-'*80}\n")
        return False

def main():
    """主函数"""
    print("开始批量评估多个模型...")
    print(f"将依次评估以下模型: {', '.join(MODELS)}")
    print(f"注意: InternVL3-8B 已完成评估，从 moonshot-v1-128k-vision-preview 开始")
    
    # 记录开始时间
    start_time = time.time()
    
    # 记录成功和失败的模型
    successful_models = []
    failed_models = []
    
    # 依次评估每个模型
    for model in MODELS:
        success = run_evaluation(model)
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
    print("评估完成摘要")
    print("="*80)
    print(f"总耗时: {hours}小时 {minutes}分钟 {seconds}秒")
    print(f"成功模型 ({len(successful_models)}): {', '.join(successful_models)}")
    print(f"失败模型 ({len(failed_models)}): {', '.join(failed_models)}")
    print("="*80)

if __name__ == "__main__":
    main() 