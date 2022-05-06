import torch
import pytorch_lightning as pl


def gpu_detecter():
    gpu_num = torch.cuda.device_count()
    print_detecter_result(gpu_num)
    return gpu_num


# https://github.com/PyTorchLightning/pytorch-lightning/discussions/6501
@pl.utilities.rank_zero_only
def print_detecter_result(gpu_num):
    if gpu_num > 1:
        print(f"{gpu_num} GPUs detected")
    else:
        print("No GPUs detected")
