import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """模型配置"""
    general_model_path: str = "./models/Qwen2.5-VL-7B-Instruct"
    general_model_name: str = "Qwen2.5-VL-7B-Instruct"

    specialized_model_path: str = "./models/Qwen2-VL-2B-Instruct"
    lora_path: str = "./output/Qwen2-VL-2B-bbox/checkpoint-7500"
    specialized_model_name: str = "Qwen2-VL-2B-Instruct-LoRA"

    image_base_dir: str = "./data/images"
    output_dir: str = "./output/visible_result"
    detection_results_dir: str = "./output/detection_results"

    max_new_tokens: int = 512
    temperature: float = 0.1
    use_cache: bool = True

    @classmethod
    def from_env(cls):
        return cls(
            general_model_path=os.getenv("GENERAL_MODEL_PATH", cls.general_model_path),
            specialized_model_path=os.getenv("SPECIALIZED_MODEL_PATH", cls.specialized_model_path),
            lora_path=os.getenv("LORA_PATH", cls.lora_path),
            image_base_dir=os.getenv("IMAGE_BASE_DIR", cls.image_base_dir),
        )