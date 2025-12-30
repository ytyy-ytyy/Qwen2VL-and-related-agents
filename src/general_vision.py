import torch
import gc
from transformers import AutoProcessor
from typing import Optional
import re
import os

class GeneralVisionModel:
    """ Qwen2.5-VL-7B """

    def __init__(self, model_path: str = "./models/Qwen2.5-VL-7B-Instruct"):
        self.model = None
        self.processor = None
        self.model_path = model_path
        self._load_model()

    def _clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _load_model(self):
        if self.model is not None:
            return

        print(f"loading Qwen2.5-VL-7B...")
        print(f"model_path: {self.model_path}")

        if not os.path.exists(self.model_path):
            print(f"Model directory does not exist: {self.model_path}")
            print("   huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./models/Qwen2.5-VL-7B-Instruct")
            raise FileNotFoundError(f"Model directory does not exist: {self.model_path}")

        if torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                quantization_config = bnb_config
                print(" 4-bit quantification")
            except ImportError:
                quantization_config = None
                print("Cannot use 4-bit quantization")
        else:
            quantization_config = None

        # 清理显存
        self._clear_memory()

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_class = Qwen2_5_VLForConditionalGeneration
            print(" Qwen2.5-VL model")
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration
            model_class = Qwen2VLForConditionalGeneration
            print(" Qwen2VL model")

        try:
            # 加载模型
            self.model = model_class.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                offload_folder="./offload",
                trust_remote_code=True
            )

            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            self.model.eval()
            self.model.config.use_cache = False

            print("Qwen2.5-VL model susscess load")

        except Exception as e:
            print(f"error: {str(e)}")
            raise

    def process(self, image_path: str, question: str) -> str:
        """
        Args:
            image_path: Image path
            question: Question Text
        Returns:
            str: Model response
        """
        try:
            print(f" picture : {os.path.basename(image_path)}")
            print(f" question : {question}")

            if not os.path.exists(image_path):
                return f" The image does not exist: {image_path}"

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question}
                ]
            }]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                from PIL import Image
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
                    max_new_tokens=512,
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

            del inputs, output_ids
            self._clear_memory()

            return response.strip()

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"error: {str(e)}")
            return f"error: {str(e)}"

    def model_for_text_reasoning(self, text_prompt: str) -> str:

        try:
            react_optimized_prompt = f"""你是一个ReAct智能体，需要按照以下格式输出：

    1. 先思考（用<thought>标签包裹）
    2. 然后决定是执行工具还是给出最终答案

    如果是执行工具：
    <thought>你的思考</thought>
    <action>工具名(参数)</action>

    如果是给出最终答案：
    <thought>你的思考</thought>
    <final_answer>最终答案</final_answer>

    请严格遵循这个格式！

    任务：{text_prompt}
    """

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": react_optimized_prompt}
                ]
            }]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )

            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
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

            del inputs, output_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not re.search(r'<(\w+)>.*?</\1>', response):
                response = f"<thought>{response}</thought>"

            return response.strip()

        except Exception as e:
            return f"<thought>: {str(e)}</thought><action>list_car_images()</action>"