import warnings

import omegaconf
import torch
import torchvision

import config
import utils
import os

warnings.filterwarnings('ignore')

@config.AutoConf(config_path="config", config_name="clip")
def main(config: omegaconf.OmegaConf):

    # Init lightning model
    model = utils.instantiate(config.model)

    ckpt_path = config.callbacks.train.model_checkpoint.dirpath
    last_ckpt = [fn for fn in os.listdir(ckpt_path) if fn.endswith(".ckpt")][-1]
    best_ckpt_path = f'{ckpt_path}/{last_ckpt}'
    print(f"Best checkpoint: {best_ckpt_path}")
    best_model = model.load_from_checkpoint(best_ckpt_path)
    
    # Save onnx
    ckpt_path = config.callbacks.train.model_checkpoint.dirpath
    filepath = f"{ckpt_path}"


    torch.manual_seed(0)
    input_ids = torch.randint(high=1000, size=(1,128))
    attention_mask = torch.randint(high=2, size=(1,128))
    pixel_values = torch.rand(size=(1, 3, 224,224),dtype=torch.float32)

    best_model.eval()
    input_names = [ "image_input"]
    output_names = ['image_embeds']
    # overwrite forwad with get_image_features, no norm function for this feature
    original_forward = best_model.forward
    best_model.forward = best_model.get_image_feature
    torch.onnx.export(best_model,\
                    pixel_values,\
                    f"{filepath}clip_image.onnx", \
                    verbose=True, \
                    input_names=input_names,\
                    output_names=output_names,\
                    opset_version=11)
    # overwrite forwad with get_text_features, no norm function for this feature
    input_names = [ "text_input",'text_attention_mask']
    output_names = [ "text_embeds"]
    best_model.forward = best_model.get_text_feature
    torch.onnx.export(best_model,\
                    (input_ids,attention_mask),\
                    f"{filepath}/clip_text.onnx", \
                    verbose=True, \
                    input_names=input_names,\
                    output_names=output_names,\
                    opset_version=11)

    best_model.forward = original_forward

if __name__ == '__main__':
    main()
