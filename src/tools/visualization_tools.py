# tools/visualization_tools.py
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def draw_bounding_box(image_path: str, bbox: list,
                      question: str = "",
                      vehicle_index: int = 1,
                      output_dir: str = "./output/visible_result") -> str:
    """
    在图像上绘制边界框

    Args:
        image_path: 图像路径
        bbox: 边界框坐标 [x1, y1, x2, y2] (归一化坐标)
        question: 问题文本
        vehicle_index: 车辆索引
        output_dir: 输出目录

    Returns:
        保存的图像路径
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 打开图像
        img = Image.open(image_path).convert('RGB')
        img_width, img_height = img.size

        # 转换归一化坐标为像素坐标
        x1 = int(bbox[0] * img_width)
        y1 = int(bbox[1] * img_height)
        x2 = int(bbox[2] * img_width)
        y2 = int(bbox[3] * img_height)

        # 确保坐标在图像范围内
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))

        # 创建绘图对象
        draw = ImageDraw.Draw(img)

        # 绘制边界框
        box_color = (255, 0, 0)  # 红色
        box_width = 3

        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)

        # 绘制标签背景
        label_text = f"Vehicle {vehicle_index}"
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        # 计算文本大小
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # 绘制文本背景
        label_padding = 5
        label_x = x1
        label_y = max(0, y1 - text_height - 2 * label_padding)

        draw.rectangle(
            [label_x, label_y,
             label_x + text_width + 2 * label_padding,
             label_y + text_height + 2 * label_padding],
            fill=box_color
        )

        # 绘制文本
        draw.text(
            (label_x + label_padding, label_y + label_padding),
            label_text,
            fill=(255, 255, 255),  # 白色
            font=font
        )

        # 保存图像
        output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_bbox.png"
        output_path = os.path.join(output_dir, output_filename)
        img.save(output_path)

        # 添加坐标信息到图像
        info_img = img.copy()
        info_draw = ImageDraw.Draw(info_img)

        info_text = f"BBox: [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]"
        info_y = img_height - 50

        # 绘制信息背景
        info_draw.rectangle(
            [0, info_y - 20, img_width, info_y + 30],
            fill=(0, 0, 0, 128)  # 半透明黑色
        )

        # 绘制信息文本
        info_draw.text(
            (10, info_y),
            info_text,
            fill=(255, 255, 255),
            font=font
        )

        # 保存带信息的图像
        info_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_bbox_info.png"
        info_path = os.path.join(output_dir, info_filename)
        info_img.save(info_path)

        return info_path

    except Exception as e:
        print(f"Visualization error: {e}")
        return f"Error saving visualization: {str(e)}"


def create_summary_image(image_paths: list, output_path: str = "./output/summary.png") -> str:
    """
    创建多图像摘要

    Args:
        image_paths: 图像路径列表
        output_path: 输出路径

    Returns:
        保存的摘要图像路径
    """
    try:
        images = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img.thumbnail((300, 300))  # 缩略图
                images.append(img)

        if not images:
            return "No images to create summary"

        # 创建网格布局
        cols = 3
        rows = (len(images) + cols - 1) // cols

        cell_width = max(img.width for img in images)
        cell_height = max(img.height for img in images)

        summary_img = Image.new('RGB',
                                (cols * cell_width, rows * cell_height),
                                (240, 240, 240))

        draw = ImageDraw.Draw(summary_img)

        for i, img in enumerate(images):
            row = i // cols
            col = i % cols

            x = col * cell_width + (cell_width - img.width) // 2
            y = row * cell_height + (cell_height - img.height) // 2

            summary_img.paste(img, (x, y))

            # 绘制图像名称
            img_name = os.path.basename(image_paths[i])
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()

            draw.text((x, y - 15), img_name, fill=(0, 0, 0), font=font)

        summary_img.save(output_path)
        return output_path

    except Exception as e:
        return f"Error creating summary: {str(e)}"