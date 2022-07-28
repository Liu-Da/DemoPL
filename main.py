import warnings

import omegaconf
import torch

import config
import utils

warnings.filterwarnings('ignore')

@config.AutoConf(config_path="config", config_name="clip_match")
def main(config: omegaconf.OmegaConf):

    # Init train lightning callbacks
    train_callbacks = utils.instantiate(config.callbacks.train)

    # Init test lightning callbacks
    test_callbacks = utils.instantiate(config.callbacks.test)

    # Init lightning loggers
    logger = utils.instantiate(config.logger)

    # Init lightning datamodule
    datamodule = utils.instantiate(config.datamodule)

    # Init lightning model
    model = utils.instantiate(config.model)
    # Training
    trainer = utils.instantiate(config.trainer.train, callbacks=train_callbacks, logger=logger)

    if config.resume.ckpt_path:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=config.resume.ckpt_path)
    else:
        trainer.fit(model=model, datamodule=datamodule)

    best_ckpt_path = trainer.checkpoint_callback.best_model_path

    # Testing
    torch.distributed.destroy_process_group()
    if trainer.is_global_zero:
        best_model = model.load_from_checkpoint(best_ckpt_path)
        trainer = utils.instantiate(config.trainer.test, callbacks=test_callbacks)
        trainer.test(model=best_model, datamodule=datamodule,  verbose=False)

        # Save onnx
        ckpt_path = config.callbacks.train.model_checkpoint.dirpath
        filepath = f"{ckpt_path}"
        best_model.to_onnx(filepath, export_params=True, opset_version=11)
        print(f"Saved onnx model to {filepath}")

        # # Save torchscript
        # filepath = f"{ckpt_path}/model.pt"
        # script = best_model.to_torchscript(method="trace")
        # torch.jit.save(script, filepath)
        # print(f"Saved torchscript model to {filepath}")

if __name__ == '__main__':
    main()
