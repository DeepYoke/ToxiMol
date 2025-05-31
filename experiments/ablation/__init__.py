"""
分子毒性修复框架的消融实验包。

此包包含用于评估框架不同组件的消融实验。
"""

# 毒性阈值和非毒性阈值分析
from experiments.ablation.toxicity_threshold_analysis import ToxicityThresholdAnalyzer
from experiments.ablation.nontoxicity_threshold_analysis import NonToxicityThresholdAnalyzer

# 毒性概率阈值分析
from experiments.ablation.toxicity_probability_threshold_analysis import ToxicityProbabilityThresholdAnalyzer

# 评价指标消融实验
from experiments.ablation.run_criteria_ablation import (
    CriteriaAblationEvaluator,
    run_criteria_ablation,
    create_comparison_report,
    CRITERIA_COMBINATIONS,
    CRITERIA_NAMES
)

__all__ = [
    # 毒性阈值分析
    "ToxicityThresholdAnalyzer",
    "NonToxicityThresholdAnalyzer",
    
    # 毒性概率阈值分析
    "ToxicityProbabilityThresholdAnalyzer",
    
    # 评价指标消融实验
    "CriteriaAblationEvaluator",
    "run_criteria_ablation",
    "create_comparison_report",
    "CRITERIA_COMBINATIONS",
    "CRITERIA_NAMES"
] 