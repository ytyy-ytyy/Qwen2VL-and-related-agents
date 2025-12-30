#!/usr/bin/env python3
import os
import sys

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加当前目录到sys.path
sys.path.insert(0, current_dir)

try:
    from general_vision import GeneralVisionModel
    from specialized_detector import SpecializedVehicleDetector
    from configs.settings import ModelConfig
    from tools.system_tools import list_car_images
    from tools.visualization_tools import draw_bounding_box
    IMPORT_SUCCESS = True
    print("All modules imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    print("Some modules may not be available")
    IMPORT_SUCCESS = False
    # 定义空函数作为fallback
    def list_car_images(directory=None):
        return "system_tools not available"
    def draw_bounding_box(*args, **kwargs):
        return "visualization_tools not available"


class CarVisionToolkit:

    def __init__(self):
        self.detector = None
        self.general_model = None
        self.config = ModelConfig()

    def get_detector(self) -> SpecializedVehicleDetector:
        if self.detector is None:
            self.detector = SpecializedVehicleDetector(
                model_path=self.config.specialized_model_path,
                lora_path=self.config.lora_path
            )
        return self.detector

    def get_general_model(self) -> GeneralVisionModel:
        if self.general_model is None:
            self.general_model = GeneralVisionModel(
                model_path=self.config.general_model_path
            )
        return self.general_model

    def process_vehicle_question(self, image_path: str, question: str) -> str:
        try:
            if not os.path.exists(image_path):
                possible_paths = [
                    image_path,
                    f"./data/images/{os.path.basename(image_path)}",
                    f"../data/images/{os.path.basename(image_path)}"
                ]

                for path in possible_paths:
                    if os.path.exists(path):
                        image_path = path
                        break

                if not os.path.exists(image_path):
                    return f"Image not found: {image_path}"

            print(f"Processing: {os.path.basename(image_path)}")
            print(f"Question: {question}")

            # 获取检测器
            detector = self.get_detector()

            # 判断问题类型
            if detector.is_supported_question(question):
                print("Using specialized detector")

                # 使用专用检测模型
                result = detector.detect(image_path, question)

                if not result["success"]:
                    # 如果专用模型失败，尝试使用通用模型
                    print("Specialized model failed, trying general model...")
                    general_model = self.get_general_model()
                    answer = general_model.process(image_path, question)

                    info = f"Specialized detection failed, using general model:\n"
                    info += f"Image: {os.path.basename(image_path)}\n"
                    info += f"Question: {question}\n"
                    info += f"Result: {answer}\n"

                    return info

                # 提取坐标并绘图
                bbox = result["bbox"]
                vehicle_index = result.get("vehicle_index", 1)

                draw_result = draw_bounding_box(
                    image_path,
                    bbox,
                    question=question,
                    vehicle_index=vehicle_index
                )

                # 构造返回信息
                info = f"Specialized detection completed\n"
                info += f"Image: {os.path.basename(image_path)}\n"
                info += f"Question: {question}\n"
                if vehicle_index > 1:
                    info += f"Target: vehicle #{vehicle_index}\n"
                info += f"Bounding box: {bbox}\n"
                info += f"Raw output: {result['raw_response']}\n"
                info += f"Visualization saved to: {draw_result}\n"

                # 添加坐标解释
                info += f"\nCoordinates (normalized 0-1):\n"
                info += f"  Top-left x: {bbox[0]:.4f}, y: {bbox[1]:.4f}\n"
                info += f"  Bottom-right x: {bbox[2]:.4f}, y: {bbox[3]:.4f}\n"
                info += f"  Width: {bbox[2] - bbox[0]:.4f}, Height: {bbox[3] - bbox[1]:.4f}"

                return info

            else:
                print("Using general vision model")

                # 使用通用视觉模型
                general_model = self.get_general_model()
                answer = general_model.process(image_path, question)

                # 构造返回信息
                info = f"General vision analysis completed\n"
                info += f"Image: {os.path.basename(image_path)}\n"
                info += f"Question: {question}\n"
                info += f"Result: {answer}\n"

                return info

        except Exception as e:
            return f"Error during processing: {str(e)}"

    def interactive_mode(self):
        print("Car Vision Toolkit - Interactive Mode")
        print("Type 'quit' or 'exit' to exit")
        print("Type 'list' to view available images")

        while True:
            try:
                user_input = input("\n>>> ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                elif user_input.lower() == 'list':
                    if IMPORT_SUCCESS:
                        print(list_car_images())
                    else:
                        print("System tools not available")

                elif user_input.startswith('process'):
                    # 解析命令: process 图片路径 "问题"
                    parts = user_input.split(' ', 2)
                    if len(parts) >= 3:
                        image_path = parts[1]
                        question = parts[2].strip('"\'')
                        result = self.process_vehicle_question(image_path, question)
                        print(result)
                    else:
                        print("Format error. Correct format: process image_path 'question'")

                else:
                    print("Unknown command. Available commands:")
                    print("  list - show image list")
                    print("  process <image_path> '<question>' - process image")
                    print("  quit - exit")

            except KeyboardInterrupt:
                print("\nUser interrupted, exiting")
                break
            except Exception as e:
                print(f"Error: {str(e)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Car Vision Toolkit")
    parser.add_argument("--image", "-i", help="Image path")
    parser.add_argument("--question", "-q", help="Question")
    parser.add_argument("--interactive", "-t", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    toolkit = CarVisionToolkit()

    if args.interactive:
        toolkit.interactive_mode()
    elif args.image and args.question:
        result = toolkit.process_vehicle_question(args.image, args.question)
        print(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()