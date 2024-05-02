import torch
import lightning.pytorch as pl
import numpy as np
import matplotlib.pyplot as plt
from lib.models.regression.aggregator import *
from lib.models.regression.head import *
from lib.models.regression.encoder.resnet import ResNet
from lib.models.regression.encoder.resunet import ResUNet

from lib.utils.loss import *
from lib.utils.metrics import pose_error_torch, error_auc, A_metrics

from torch.utils.data import DataLoader as DL
from torchvision.transforms import ColorJitter, Grayscale
from lib.datasets.sampler import RandomConcatSampler
from lib.datasets.scannet import ScanNetDataset
from lib.datasets.sevenscenes import SevenScenesDataset
from lib.datasets.mapfree import MapFreeDataset
from lib.datasets.ggl import GGLDataset, GraphDataset, GraphPoseDataset


class RegressionModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        encoder = eval(cfg.ENCODER.TYPE)
        aggregator = eval(cfg.AGGREGATOR.TYPE)
        head = eval(cfg.HEAD.TYPE)
        self.encoder = encoder(cfg.ENCODER)
        self.aggregator = aggregator(cfg.AGGREGATOR, self.encoder.num_out_layers)
        self.head = head(cfg, self.aggregator.num_out_layers)
        self.rot_loss = eval(cfg.TRAINING.ROT_LOSS)
        self.trans_loss = eval(cfg.TRAINING.TRANS_LOSS)
        self.LAMBDA = cfg.TRAINING.LAMBDA
        if cfg.TRAINING.LAMBDA == 0.:
            self.s_r = torch.nn.Parameter(torch.zeros(1))
            self.s_t = torch.nn.Parameter(torch.zeros(1))

        self.val_outputs = []
        self.val_scenes = []

    def get_sampler(self, dataset, reset_epoch=False):
        if self.cfg.TRAINING.SAMPLER == 'scene_balance':
            sampler = RandomConcatSampler(dataset, self.cfg.TRAINING.N_SAMPLES_SCENE, self.cfg.TRAINING.SAMPLE_WITH_REPLACEMENT, 
                                          shuffle=True, reset_on_iter=reset_epoch)
        else: sampler = None
        return sampler

    def train_dataloader(self):
        transforms = ColorJitter() if self.cfg.DATASET.AUGMENTATION_TYPE == 'colorjitter' else None
        transforms = Grayscale(num_output_channels=3) if self.cfg.DATASET.BLACK_WHITE else transforms
        dataset = GGLDataset(self.cfg, 'train', transforms=transforms)
        sampler = self.get_sampler(dataset)
        return DL(dataset, batch_size=self.cfg.TRAINING.BATCH_SIZE, num_workers=self.cfg.TRAINING.NUM_WORKERS, sampler=sampler)

    def val_dataloader(self):
        data = GraphDataset(self.cfg)
        dataset = GraphPoseDataset('val', data)
        return DL(dataset, batch_size=8, num_workers=1, drop_last=True)

    def test_dataloader(self):
        dataset = self.dataset_type(self.cfg, 'test')
        return DL(dataset, batch_size=8, num_workers=1, shuffle=False)

    def forward(self, data):
        vol0 = self.encoder(data['image0'])
        vol1 = self.encoder(data['image1'])
        global_volume = self.aggregator(vol0, vol1)
        R, t = self.head(global_volume, data)
        data['R'] = R
        data['t'] = t
        data['inliers'] = 0
        return R, t

    def loss_fn(self, data):
        R_loss = self.rot_loss(data)
        t_loss = self.trans_loss(data)
        if self.LAMBDA == 0: loss = R_loss * torch.exp(-self.s_r) + t_loss * torch.exp(-self.s_t) + self.s_r + self.s_t
        else: loss = R_loss + self.LAMBDA * t_loss
        return R_loss, t_loss, loss

    def training_step(self, batch, batch_idx):
        self(batch)
        R_loss, t_loss, loss = self.loss_fn(batch)
        self.log('train/R_loss', R_loss), self.log('train/t_loss', t_loss), self.log('train/loss', loss)
        if self.LAMBDA == 0.: self.log('train/s_R', self.s_r), self.log('train/s_t', self.s_t)
        return loss

    def validation_step(self, batch, batch_idx):
        # 'image0': image1, 'image1': image2, 'T_0to1': T, 'abs_q_0': q1, 'abs_c_0': c1, 'abs_q_1': q2, 'abs_c_1': c2, 
        # 'K_color0': K, 'K_color1': K, 'scene': pair['scene'], 'pair': pair['imgs']
        Tgt = batch['T_0to1']
        R, t = self(batch)

        R_loss, t_loss, loss = self.loss_fn(batch)
        outputs = pose_error_torch(R, t, Tgt, reduce=None)
        inds = batch['inds'][0]

        # reshape t
        t = t.squeeze(1)

        # Add relative pose estimation to the anchor pose
        for b_idx in range(len(batch['scene'])): 
            scene = (batch['scene'][0][b_idx].item(), batch['scene'][1][b_idx].item())
            self.val_scenes.append({'scene': scene, 'inds': inds[b_idx], 't_err': outputs['t_err_euc'][b_idx].item(), 
                                    't_err_scale': outputs['t_err_scale'][b_idx].item(), 't_err_scale_sym': outputs['t_err_scale_sym'][b_idx].item(), 
                                    'pose0': batch['abs_c_0'][b_idx].cpu().detach().numpy(), 'pose1': batch['abs_c_1'][b_idx].cpu().detach().numpy(),
                                    't': t[b_idx].cpu().detach().numpy(), 'R': R[b_idx].cpu().detach().numpy()})

    def on_validation_epoch_end(self):
        aggregated = {}
        for v_out in self.val_scenes:
            scene = v_out['scene']
            if scene not in aggregated.keys():
                aggregated[scene] = {'inds': [], 't_err_euc': [], 't_err_scale': [], 't_err_scale_sym': [], 't': []}

            aggregated[scene]['t_err_euc'].append(v_out['t_err'])
            aggregated[scene]['t_err_scale'].append(v_out['t_err_scale'])
            aggregated[scene]['t_err_scale_sym'].append(v_out['t_err_scale_sym'])
            aggregated[scene]['inds'].append(v_out['inds'].item())
            aggregated[scene]['pose0'] = v_out['pose0'] # Query?
            aggregated[scene]['pose1'] = v_out['pose1'] # Reference?
            # aggregated[scene]['t'].append(v_out['t'])

        for scene in aggregated.keys():
            plt.subplot(111)
            t_err_euc = np.array(aggregated[scene]['t_err_euc'])
            t_err_scale = np.array(aggregated[scene]['t_err_scale'])
            t_err_scale_sym = np.array(aggregated[scene]['t_err_scale_sym'])
            plt.plot(t_err_euc, label='t_err_euc')
            plt.plot(t_err_scale, label='t_err_scale')
            plt.plot(t_err_scale_sym, label='t_err_scale_sym')
            pose_1 = aggregated[scene]['pose0']
            # t = np.array(aggregated[scene]['t'])
            p1 = np.hypot(pose_1[0], pose_1[1])
            plt.axvline(x=p1, color='r', linestyle='--')
            plt.title(f'{scene}')
            plt.xlabel('Sample')
            plt.ylabel('Error')
            plt.legend()
            plt.savefig(f'plots/{scene}_{self.current_epoch}.png')
            plt.clf()

        self.val_outputs.clear(), self.val_scenes.clear()

    def configure_optimizers(self):
        tcfg = self.cfg.TRAINING
        opt = torch.optim.Adam(self.parameters(), lr=tcfg.LR, eps=1e-6)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, verbose=True, patience=2)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sch, 'interval': 'epoch', 'monitor': 'train/loss'}}





