#!/usr/bin/env python3
"""
Toxicity Repair Experiment Runner using DeepseekVLV2 model.

This script runs toxicity repair experiments on molecules using DeepseekVLV2 model.
It can process single tasks or run batch experiments across multiple toxicity datasets.
"""

import os
import json
import base64
import argparse
import time
import math
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, AutoConfig, TextIteratorStreamer, AutoModelForCausalLM
from threading import Thread
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from DeepSeek.deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from DeepSeek.deepseek_vl2.utils.io import load_pil_images

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("toxicity_repair")

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# InternVL3 model constants
DEFAULT_MODEL_PATH = "deepseek-ai/deepseek-vl2"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# All available tasks
AVAILABLE_TASKS = [
    "clintox", "ames", "carcinogens_lagunin", "dili", "herg", 
    "herg_central", "herg_karim", "ld50_zhu", "skin_reaction", 
    "tox21", "toxcast"
]

class DeepseekVLV2Agent:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.device_map = self._split_model(model_path)
        
        logger.info(f"Loading DeepseekVLV2 model from {model_path}")
        self.processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=self.device_map,
            # cache_dir=CACHE_DIR
        ).eval()
        
        logger.info("Model and tokenizer loaded successfully")

    def _split_model(self, model_path):
        """Split model across available GPUs."""
        device_map = {}
        world_size = torch.cuda.device_count()
        
        if world_size <= 0:
            logger.warning("No CUDA devices found, using CPU only (this will be slow)")
            return "cpu"
            
        logger.info(f"Found {world_size} CUDA devices")
        
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            num_layers = config.num_hidden_layers
            
            # Since the first GPU will be used for ViT, treat it as half a GPU
            num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
            num_layers_per_gpu = [num_layers_per_gpu] * world_size
            num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
            
            layer_cnt = 0
            for i, num_layer in enumerate(num_layers_per_gpu):
                for j in range(num_layer):
                    device_map[f'model.layers.{layer_cnt}'] = i
                    layer_cnt += 1
                    
            device_map['vision_model'] = 0
            device_map['model.embed_tokens'] = 0
            device_map['model.norm'] = 0
            device_map['lm_head'] = 0
            device_map[f'model.layers.{num_layers - 1}'] = 0
            
            return device_map
        except Exception as e:
            logger.error(f"Error splitting model: {e}")
            logger.info("Falling back to automatic device mapping")
            return "auto"

    def load_image(self, image_path):
        """Load and preprocess an image file for model input."""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

    def generate_completion(self, system_prompt, user_prompt, image_path=None, generation_config=None):
        """Generate a completion using the DeepseekVLV2 model."""
        if generation_config is None:
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            
        try:
            # Prepare conversation
            conversation = [
                {
                    "role": "<|User|>",
                    "content": user_prompt,
                    "images": [image_path] if image_path else []
                },
                {"role": "<|Assistant|>", "content": ""}
            ]
            
            # Load images and prepare inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = self.processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt=system_prompt
            )
            
            device = next(self.model.parameters()).device
            
            input_dict = {
                'input_ids': prepare_inputs.input_ids.to(device),
                'attention_mask': prepare_inputs.attention_mask.to(device),
                'images': prepare_inputs.images.to(device) if hasattr(prepare_inputs, 'images') else None,
                'images_seq_mask': prepare_inputs.images_seq_mask.to(device) if hasattr(prepare_inputs, 'images_seq_mask') else None,
                'images_spatial_crop': prepare_inputs.images_spatial_crop.to(device) if hasattr(prepare_inputs, 'images_spatial_crop') else None
            }
            
            input_dict = {k: v for k, v in input_dict.items() if v is not None}
            
            # Run image encoder to get the image embeddings
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                inputs_embeds = self.model.prepare_inputs_embeds(**input_dict)
                
                if torch.is_tensor(inputs_embeds):
                    inputs_embeds = inputs_embeds.to(device)
                
                # Run the model to get the response
                outputs = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=input_dict['attention_mask'],
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=generation_config.get("max_new_tokens", 1024),
                    do_sample=generation_config.get("do_sample", True),
                    temperature=generation_config.get("temperature", 0.5),
                    use_cache=True
                )
            
            outputs = outputs.cpu()
            response = self.tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return response
            
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
