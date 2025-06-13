#!/usr/bin/env python3
"""
Llava-One-Vision Experiment Runner.

This script runs experiments using the Llava-One-Vision model.
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


from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import warnings
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llava_one_vision")

# Constants
DEFAULT_MODEL_PATH = ""  # 替换为你使用的实际模型路径

class LlavaOneVisionAgent:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_name = "llava_qwen"
        self.devide = "cuda"
        self.device_map = "auto"
        llava_model_args = {
            "multimodal": True,
            "attn_implementation": "sdpa",
        }
        logger.info(f"Loading Llava-One-Vision model from {model_path}")

        self.tokenizer, self.model, self.image_processor, max_length = load_pretrained_model(self.model_path, None, self.model_name, device_map=self.device_map,  **llava_model_args)
        self.model.eval()
        logger.info("Model and tokenizer loaded successfully")

    def generate_completion(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        image_path: Optional[str] = None,
        generation_config=None
    ) -> str:
        """Generate a completion using the Llava-One-Vision model."""

        path = image_path
        image = Image.open(path)
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]

        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + system_prompt + user_prompt
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_sizes = [image.size]

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        # print(text_outputs)
        return text_outputs