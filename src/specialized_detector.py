import os
import re
import json
import torch
import time
from typing import List, Optional, Dict
from PIL import Image
from transformers import AutoProcessor
from peft import PeftModel, LoraConfig, TaskType


class SpecializedVehicleDetector:
    def _init_patterns(self):
        """初始化正则表达式模式"""
        self.bbox_pattern = re.compile(
            r'\[bbox\s*:\s*\[?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]?\s*\]',
            re.IGNORECASE
        )

        self.vehicle_index_pattern = re.compile(
            r'(?:第|#|number\s*)?(\d+)(?:辆|个|辆汽车|辆车|vehicle)',
            re.IGNORECASE
        )

    def is_supported_question(self, question: str) -> bool:

        supported_keywords = [
            '在哪里', '哪个位置', '定位', '坐标', '边界框', 'bbox',
            'where is', 'locate', 'bounding box', 'coordinates',
            '位置', '框出', '标出', '标记'
        ]

        question_lower = question.lower()
        return any(keyword in question_lower for keyword in supported_keywords)

    def extract_vehicle_index(self, question: str) -> int:

        match = self.vehicle_index_pattern.search(question)
        if match:
            try:
                return int(match.group(1))
            except:
                pass
        return 1

    def detect(self, image_path: str, question: str) -> Dict:

        try:
            if not os.path.exists(image_path):
                return {
                    "success": False,
                    "error": f"Image not found: {image_path}",
                    "bbox": None,
                    "raw_response": ""
                }


            vehicle_index = self.extract_vehicle_index(question)


            prompt = f"""请定位图中的车辆。
            如果图中有多辆车，请定位第{vehicle_index}辆车。
            只输出边界框坐标，格式为：[bbox: [x1, y1, x2, y2]]
            坐标范围是0到1的归一化坐标。"""

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )


            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                image = Image.open(image_path).convert('RGB')
                image_inputs = [image]
                video_inputs = None

            inputs = self.processor(
                text=[text],
                images=[image_inputs[0]] if image_inputs else None,
                videos=[video_inputs[0]] if video_inputs else None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            )

            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    temperature=0.1
                )

            gen_ids = output_ids[0, inputs.input_ids.shape[1]:]
            response = self.processor.decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            # 提取边界框
            match = self.bbox_pattern.search(response)
            if match:
                bbox = [
                    float(match.group(1)),
                    float(match.group(2)),
                    float(match.group(3)),
                    float(match.group(4))
                ]


                bbox = [max(0.0, min(1.0, coord)) for coord in bbox]

                return {
                    "success": True,
                    "bbox": bbox,
                    "vehicle_index": vehicle_index,
                    "raw_response": response
                }
            else:
                return {
                    "success": False,
                    "error": "No bounding box found in response",
                    "bbox": None,
                    "raw_response": response
                }

        except Exception as e:
            import traceback
            print(f"Detection error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "bbox": None,
                "raw_response": ""
            }
    def __init__(self,
                 model_path: str = "./models/Qwen2-VL-2B-Instruct",
                 lora_path: str = "./output/Qwen2-VL-2B-bbox/checkpoint-7500"):
        self.model = None
        self.processor = None
        self.model_path = model_path
        self.lora_path = lora_path
        self._load_model()
        self._init_patterns()

    def _clear_memory(self):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _load_model(self):
        if self.model is not None:
            return

        print(f"Qwen2-VL-2B")
        print(f" Model path: {self.model_path}")

        if not os.path.exists(self.model_path):
            print(f" The model directory does not exist: {self.model_path}")
            print("   huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir ./models/Qwen2-VL-2B-Instruct")
            raise FileNotFoundError(f"The model directory does not exist: {self.model_path}")

        if self.lora_path and os.path.exists(self.lora_path):
            print(f"LoRA weight: {self.lora_path}")
        else:
            print(f" don‘t found LoRA : {self.lora_path}")
            print("will using general model")

        self._clear_memory()
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            inference_mode=True,
            r=128,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none"
        )

        from transformers import Qwen2VLForConditionalGeneration

        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                trust_remote_code=True
            )

            if os.path.exists(self.lora_path):
                print(f" loading LoRA ")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.lora_path,
                    config=lora_cfg,
                    adapter_name="vehicle_detection"
                )
                print(" LoRA complete")

            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            self.model.eval()
            self.model.config.use_cache = False

            print(" loading complete")

        except Exception as e:
            print(f" error: {str(e)}")
            raise