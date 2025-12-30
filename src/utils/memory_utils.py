import torch
import gc


def clear_memory():
    """清理显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_device_info() -> Dict:
    """获取设备信息"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info.update({
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_allocated": torch.cuda.memory_allocated() / 1024 ** 3,  # GB
            "memory_reserved": torch.cuda.memory_reserved() / 1024 ** 3,  # GB
            "max_memory_allocated": torch.cuda.max_memory_allocated() / 1024 ** 3,
        })

    return info