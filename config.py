import functools

import omegaconf
import pytorch_lightning as pl

import utils


def AutoConf(config_path="config", config_name="mnist", detect_info=True, verbose=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper():
            config = omegaconf.OmegaConf.from_cli()
            pth = f"{config_path}/{config.get('config_name', config_name)}.yaml"
            config.merge_with(omegaconf.OmegaConf.load(pth))
            if detect_info:
                load_detector_info(config)
            if verbose:
                log_config(config)
            return func(config) 
        return wrapper
    return decorator

def load_detector_info(config):
    os_info, arch_info = utils.platform_detector()
    gpu_num, cpu_num, ram_size = utils.hardware_detector()
    config.os_info = os_info
    config.arch_info = arch_info
    config.gpu_num = gpu_num
    config.cpu_num = cpu_num
    config.ram_size = ram_size

    # auto detect if we are using GPUs
    config.trainer.train.accelerator = "gpu" if gpu_num > 0 else "cpu"
    config.trainer.train.strategy = "ddp" if gpu_num > 0 else None
    config.trainer.train.devices = gpu_num if gpu_num > 0 else None
    
    config.trainer.test.accelerator = "gpu" if gpu_num > 0 else "cpu"
    config.trainer.test.devices = 1

    # auto detect if we run on M1 Mac
    num_workers = 0 if arch_info == "ARM" else cpu_num
    config.datamodule.params.num_workers = num_workers

        
@pl.utilities.rank_zero_only
def log_config(config):
    print(f"{'Applying Parameters':=^100}\n")
    # for key, value in config.items():
    #     print('{0:<50}:{1:<100}\n'.format(str(key), str(value)))
    print(omegaconf.OmegaConf.to_yaml(config))
    print(f"{'':=^100}\n")