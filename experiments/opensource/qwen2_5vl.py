#!/usr/bin/env python3
"""
Qwen2.5-VL Agent compatible with InternVL3Agent-style interfaces.

This agent supports multimodal input (image + text) and exposes unified methods such as `load_image`
and `generate_completion` for easy integration across models.
"""

import os
import torch
from PIL import Image
import requests
from io import BytesIO
from typing import List, Optional, Union, Dict, Any
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - QwenVLAgent - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen2vl")

class QwenVLAgent:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype: torch.dtype = torch.float16
    ):
        self.model_path = model_path
        logger.info(f"Loading Qwen2.5-VL model from {model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation="sdpa"
        ).eval()

        logger.info("Loading processor")
        self.processor = AutoProcessor.from_pretrained(model_path)

    def load_image(self, image_path: Union[str, Image.Image]) -> Image.Image:
        try:
            if isinstance(image_path, Image.Image):
                return image_path.convert("RGB")
            elif image_path.startswith("http://") or image_path.startswith("https://"):
                response = requests.get(image_path)
                return Image.open(BytesIO(response.content)).convert("RGB")
            elif os.path.isfile(image_path):
                return Image.open(image_path).convert("RGB")
            else:
                raise FileNotFoundError(f"Image path not found or invalid: {image_path}")
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

    def generate_completion(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        image_path: Optional[Union[str, Image.Image]] = None,
        generation_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response from the model using a prompt and optional image.
        
        Args:
            system_prompt: Optional system message.
            user_prompt: Main user query.
            image_path: Optional image (local path, URL, or PIL.Image).
            generation_config: Dict of generation parameters (e.g., max_new_tokens, do_sample).
        
        Returns:
            Model's response text.
        """
        try:
            if generation_config is None:
                generation_config = {"max_new_tokens": 1024, "do_sample": True}
            
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            message_content = []

            if image_path:
                image = self.load_image(image_path)
                message_content.append({"type": "image", "image": image})
            
            message_content.append({"type": "text", "text": user_prompt})
            messages.append({"role": "user", "content": message_content})

            logger.info("Tokenizing messages for Qwen2.5-VL")
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)

            logger.info("Generating response")
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    **generation_config
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]

            decoded = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            return decoded[0] if decoded else ""
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
