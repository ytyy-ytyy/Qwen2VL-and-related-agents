import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def draw_bounding_box(image_path: str, bbox: List[float],
                     question: str = "", vehicle_index: int = 1,
                     output_dir: str = None) -> str:
    """绘制边界框并保存图片"""
    # 原有的draw_bounding_box方法内容...
    pass