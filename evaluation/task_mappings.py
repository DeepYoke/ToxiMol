"""
Task to toxicity endpoint mappings.

This module maps task names to their corresponding toxicity endpoints.
"""
from typing import Dict, List, Any, Optional

# Task to toxicity endpoint mappings
# primary_endpoints: Main evaluation metrics
TASK_MAPPINGS = {
    # Ames mutagenicity
    "ames": {
        "primary_endpoints": ["AMES"]
    },
    
    # Carcinogens (Lagunin dataset)
    "carcinogens_lagunin": {
        "primary_endpoints": ["Carcinogens_Lagunin"]
    },
    
    # Clinical toxicity
    "clintox": {
        "primary_endpoints": ["ClinTox"]
    },
    
    # Drug-induced liver injury
    "dili": {
        "primary_endpoints": ["DILI"]
    },
    
    # hERG channel inhibition
    "herg": {
        "primary_endpoints": ["hERG"]
    },
    
    # hERG central
    "herg_central": {
        "primary_endpoints": ["herg_central"]
    },
    
    # hERG Karim dataset
    "herg_karim": {
        "primary_endpoints": ["hERG_Karim"]
    },
    
    # Acute toxicity (LD50)
    "ld50_zhu": {
        "primary_endpoints": ["LD50_Zhu"]
    },
    
    # Skin reaction
    "skin_reaction": {
        "primary_endpoints": ["Skin_Reaction"]
    },
    
    # Tox21 assays - Each subtask maps to its corresponding endpoint
    "tox21": {
        "primary_endpoints": [
            "Tox21_SR_ARE",
            "Tox21_SR_p53",
            "Tox21_NR_Aromatase",
            "Tox21_SR_ATAD5",
            "Tox21_NR_ER_LBD",
            "Tox21_SR_HSE",
            "Tox21_NR_AR",
            "Tox21_NR_PPAR_gamma",
            "Tox21_NR_ER",
            "Tox21_SR_MMP",
            "Tox21_NR_AhR",
            "Tox21_NR_AR_LBD"
        ],
        "subtask_mapping": {
            "SR_ARE": "Tox21_SR_ARE",
            "SR_p53": "Tox21_SR_p53",
            "NR_Aromatase": "Tox21_NR_Aromatase",
            "SR_ATAD5": "Tox21_SR_ATAD5",
            "NR_ER_LBD": "Tox21_NR_ER_LBD",
            "SR_HSE": "Tox21_SR_HSE",
            "NR_AR": "Tox21_NR_AR",
            "NR_PPAR_gamma": "Tox21_NR_PPAR_gamma",
            "NR_ER": "Tox21_NR_ER",
            "SR_MMP": "Tox21_SR_MMP",
            "NR_AhR": "Tox21_NR_AhR",
            "NR_AR_LBD": "Tox21_NR_AR_LBD"
        }
    },
    
    # ToxCast assays - Each subtask maps to its corresponding endpoint
    "toxcast": {
        "primary_endpoints": [
            "ToxCast_APR_HepG2_MitoMass_24h_dn",
            "ToxCast_BSK_3C_ICAM1_down",
            "ToxCast_Tanguay_ZF_120hpf_TR_up",
            "ToxCast_ATG_M_32_CIS_dn",
            "ToxCast_ATG_RORg_TRANS_up",
            "ToxCast_TOX21_p53_BLA_p3_ch2",
            "ToxCast_BSK_3C_MCP1_down",
            "ToxCast_ATG_MRE_CIS_up",
            "ToxCast_NVS_GPCR_gLTB4",
            "ToxCast_NVS_ENZ_hAChE"
        ],
        "subtask_mapping": {
            "APR_HepG2_MitoMass_24h_dn": "ToxCast_APR_HepG2_MitoMass_24h_dn",
            "BSK_3C_ICAM1_down": "ToxCast_BSK_3C_ICAM1_down",
            "Tanguay_ZF_120hpf_TR_up": "ToxCast_Tanguay_ZF_120hpf_TR_up",
            "ATG_M_32_CIS_dn": "ToxCast_ATG_M_32_CIS_dn",
            "ATG_RORg_TRANS_up": "ToxCast_ATG_RORg_TRANS_up",
            "TOX21_p53_BLA_p3_ch2": "ToxCast_TOX21_p53_BLA_p3_ch2",
            "BSK_3C_MCP1_down": "ToxCast_BSK_3C_MCP1_down",
            "ATG_MRE_CIS_up": "ToxCast_ATG_MRE_CIS_up",
            "NVS_GPCR_gLTB4": "ToxCast_NVS_GPCR_gLTB4",
            "NVS_ENZ_hAChE": "ToxCast_NVS_ENZ_hAChE"
        }
    }
}

def get_task_mapping(task_name: str) -> Dict[str, Any]:
    """
    Get the toxicity endpoint mapping for a specific task.
    
    Args:
        task_name: Name of the task
        
    Returns:
        Dict[str, Any]: Mapping information for the task
    """
    return TASK_MAPPINGS.get(task_name, {}) 