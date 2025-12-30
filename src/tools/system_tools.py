import os
import subprocess
from typing import Optional


def list_car_images(directory: str = None) -> str:
    if directory is None:
        from configs.settings import ModelConfig
        directory = ModelConfig.image_base_dir

    try:
        if not os.path.exists(directory):
            return f" 目录不存在: {directory}"

        images = [f for f in os.listdir(directory)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if not images:
            return " 目录中没有找到图片"

        result = f" 目录: {directory}\n"
        result += f" 图片数量: {len(images)}\n"
        result += " 图片列表（前20张）:\n"

        for i, img in enumerate(images[:20]):
            result += f"  {i + 1}. {img}\n"

        if len(images) > 20:
            result += f"  ... 还有 {len(images) - 20} 张图片\n"

        return result

    except Exception as e:
        return f" 列出图片时出错: {str(e)}"


def read_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f" 读取文件时出错: {str(e)}"


def run_terminal_command(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return f" 命令执行成功:\n{result.stdout}"
        else:
            return f" 命令执行失败:\n{result.stderr}"
    except Exception as e:
        return f" 执行命令时出错: {str(e)}"