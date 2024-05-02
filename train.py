import argparse
import os
# do this before importing numpy! (doing it right up here in case numpy is dependency of e.g. json)
os.environ["MKL_NUM_THREADS"] = "1"  # noqa: E402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa: E402

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import torch
from config.default import cfg


# from lib.models.regression.model import RegressionModel
from lib.models.regression.model_copy import RegressionModel

def main(args):
    cfg.merge_from_file(args.dataset_config)
    cfg.merge_from_file(args.config)

    model = RegressionModel(cfg)
    wandb_logger = WandbLogger(project='MFR', name=args.experiment)

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(save_last=True, save_top_k=5, verbose=True, monitor='val_loss/loss', mode='min')
    epochend_callback = pl.callbacks.ModelCheckpoint(filename='e{epoch}-last', save_top_k=1, every_n_epochs=1, save_on_train_epoch_end=True)
    lr_monitoring_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(devices=1, check_val_every_n_epoch=3, max_epochs=cfg.TRAINING.EPOCHS, logger=[wandb_logger], 
                         callbacks=[lr_monitoring_callback, epochend_callback], num_sanity_val_steps=0, 
                         gradient_clip_val=cfg.TRAINING.GRAD_CLIP,
                        #  overfit_batches=2
                         )
    trainer.fit(model, ckpt_path=args.resume)



if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')  
    pl.seed_everything(42)

    stringer = 'ggl'
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default=f'config/regression/{stringer}/3d3d.yaml', help='path to config file')
    parser.add_argument('dataset_config', default=f'config/{stringer}.yaml', help='path to dataset config file')
    parser.add_argument('--experiment', help='experiment name', default='default')
    parser.add_argument('--resume', help='resume from checkpoint path', default=None)
    args = parser.parse_args([f'config/regression/{stringer}/3d3d.yaml', f'config/{stringer}.yaml', '--experiment', 'default'])
    main(args)
