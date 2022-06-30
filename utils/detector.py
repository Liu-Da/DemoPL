import platform

import psutil
import pytorch_lightning as pl
import torch


def hardware_detector():
    gpu_num = torch.cuda.device_count()
    cpu_num = psutil.cpu_count()
    ram_size = psutil.virtual_memory().total >> 30
    print_hardware_result(gpu_num, cpu_num, ram_size)
    return gpu_num, cpu_num, ram_size


def platform_detector():
    info = platform.platform()
    os_info = "Unknown OS"
    if "mac".casefold() in info.casefold():
        os_info = "MacOS"
    elif "linux".casefold() in info.casefold():
        os_info = "Linux"
    elif "win".casefold() in info.casefold():
        os_info = "Windows"

    arch_info = "x86"
    if "arm".casefold() in info.casefold():
        arch_info = "ARM"
    print_platform_info(os_info, arch_info)
    return os_info, arch_info


@pl.utilities.rank_zero_only
def print_hardware_result(gpu_num, cpu_num, ram_size):
    if gpu_num > 0:
        print(f"{gpu_num} GPUs detected")
    else:
        print("No GPUs detected")

    print(f"{cpu_num} CPUs detected")
    print(f"{ram_size} GB RAM detected")


@pl.utilities.rank_zero_only
def print_platform_info(os_info, arch_info):
    print(f"{os_info} {arch_info} detected")


if __name__ == "__main__":
    platform_detector()
    hardware_detector()
