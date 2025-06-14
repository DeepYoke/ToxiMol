 #!/usr/bin/env python3
"""
Toxicity Repair Experiment Runner using InternVL3-8B model.

This script runs toxicity repair experiments on molecules using InternVL3-8B model.
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
from transformers import AutoTokenizer, AutoModel, AutoConfig, TextIteratorStreamer
from threading import Thread
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging

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
DEFAULT_MODEL_PATH = "OpenGVLab/InternVL3-38B"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# All available tasks
AVAILABLE_TASKS = [
    "ames", "clintox", "carcinogens_lagunin", "dili", "herg", 
    "herg_central", "herg_karim", "ld50_zhu", "skin_reaction", 
    "tox21", "toxcast"
]

class InternVL3Agent:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.device_map = self._split_model(model_path)
        
        logger.info(f"Loading InternVL3 model from {model_path}")
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=self.device_map
        ).eval()
        
        logger.info("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            use_fast=False,
        )
        
        self.transform = self._build_transform(input_size=448)
        logger.info("Model and tokenizer loaded successfully")

    def _split_model(self, model_path):
        """Split model across available GPUs."""
        device_map = {}
        world_size = torch.cuda.device_count()
        # world_size = 8
        
        if world_size <= 0:
            logger.warning("No CUDA devices found, using CPU only (this will be slow)")
            return "cpu"
            
        logger.info(f"Found {world_size} CUDA devices")
        
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            num_layers = config.llm_config.num_hidden_layers
            
            # Since the first GPU will be used for ViT, treat it as half a GPU
            num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
            num_layers_per_gpu = [num_layers_per_gpu] * world_size
            num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
            
            layer_cnt = 0
            for i, num_layer in enumerate(num_layers_per_gpu):
                for j in range(num_layer):
                    device_map[f'language_model.model.layers.{layer_cnt}'] = i
                    layer_cnt += 1
                    
            device_map['vision_model'] = 0
            device_map['mlp1'] = 0
            device_map['language_model.model.tok_embeddings'] = 0
            device_map['language_model.model.embed_tokens'] = 0
            device_map['language_model.output'] = 0
            device_map['language_model.model.norm'] = 0
            device_map['language_model.model.rotary_emb'] = 0
            device_map['language_model.lm_head'] = 0
            device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
            
            return device_map
        except Exception as e:
            logger.error(f"Error splitting model: {e}")
            logger.info("Falling back to automatic device mapping")
            return "auto"

    def _build_transform(self, input_size=448):
        """Build image transformation pipeline."""
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Find the closest aspect ratio from target_ratios to the input aspect ratio."""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """Preprocess image into multiple tiles according to aspect ratio."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find the closest aspect ratio to the target
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # Calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # Split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
            
        return processed_images

    def load_image(self, image_path, input_size=448, max_num=12):
        """Load and preprocess an image file for model input."""
        try:
            image = Image.open(image_path).convert('RGB')
            images = self._dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [self.transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values.to(torch.bfloat16).cuda() if torch.cuda.is_available() else pixel_values.to(torch.bfloat16)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

    def generate_completion(self, system_prompt, user_prompt, image_path=None, generation_config=None):
        """Generate a completion using the InternVL3 model."""
        if generation_config is None:
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            
        try:
            # Load image if provided
            pixel_values = None
            if image_path:
                pixel_values = self.load_image(image_path)
                
            # Add image placeholder to user prompt if image is provided
            if pixel_values is not None:
                if '<image>' not in user_prompt:
                    user_prompt = '<image>\n' + user_prompt
                    
            history = None
            if system_prompt:
                history = [{"role": "system", "content": system_prompt}]
                
            # Get the response
            response = self.model.chat(
                self.tokenizer, 
                pixel_values, 
                user_prompt, 
                generation_config,
                history=history
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            raise
