import os
import re


def extract_bbox(text: str) -> Optional[List[float]]:
    """从文本中提取边界框"""
    text = text.replace("（", "(").replace("）", ")")
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return list(map(float, nums[:4])) if len(nums) >= 4 else None


def find_image_path(image_name: str, search_paths: List[str] = None) -> Optional[str]:
    """在多个路径中查找图片"""
    if search_paths is None:
        search_paths = [
            "",
            "./car/images/",
            "/root/autodl-tmp/qwen2.5vl/qwen2_vl/car/images/",
            "./images/"
        ]

    for path in search_paths:
        full_path = os.path.join(path, image_name)
        if os.path.exists(full_path):
            return full_path

    return None