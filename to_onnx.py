import warnings

import omegaconf
import torch

import config
import utils
import os

warnings.filterwarnings('ignore')

@config.AutoConf(config_path="config", config_name="clip_match")
def main(config: omegaconf.OmegaConf):

    # Init lightning model
    model = utils.instantiate(config.model)

    ckpt_path = config.callbacks.train.model_checkpoint.dirpath
    last_ckpt = [fn for fn in os.listdir(ckpt_path) if fn.endswith(".ckpt")]
    last_ckpt.sort()
    last_ckpt = last_ckpt[-1]
    best_ckpt_path = f'{ckpt_path}/{last_ckpt}'
    print(f"Best checkpoint: {best_ckpt_path}")
    best_model = model.load_from_checkpoint(best_ckpt_path)
    
    # Save onnx
    ckpt_path = config.callbacks.train.model_checkpoint.dirpath
    filepath = f"{ckpt_path}"
    best_model.to_onnx(filepath,opset_version=11)
    print(f"Saved onnx model to {filepath}")
    # python ./onnx-typecast/fix-clip-text-vit-32-float32---scratch.py clip_text.onnx clip_text_32.onnx

    # # Save torchscript
    # filepath = f"{ckpt_path}/model.pt"
    # script = best_model.to_torchscript(method="trace")
    # torch.jit.save(script, filepath)
    # print(f"Saved torchscript model to {filepath}")

if __name__ == '__main__':
    main()
