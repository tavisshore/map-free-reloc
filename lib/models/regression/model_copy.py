import torch
import lightning.pytorch as pl

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
from lib.datasets.ggl import GGLDataset


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

        # DATA
        datasets = {'ScanNet': ScanNetDataset, '7Scenes': SevenScenesDataset, 'MapFree': MapFreeDataset, 'ggl': GGLDataset}
        self.dataset_type = datasets[cfg.DATASET.DATA_SOURCE]
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
        dataset = self.dataset_type(self.cfg, 'train', transforms=transforms)
        sampler = self.get_sampler(dataset)
        return DL(dataset, batch_size=self.cfg.TRAINING.BATCH_SIZE, num_workers=self.cfg.TRAINING.NUM_WORKERS, sampler=sampler)

    def val_dataloader(self):
        dataset = self.dataset_type(self.cfg, 'val')
        if isinstance(dataset, ScanNetDataset): sampler = self.get_sampler(dataset, reset_epoch=True)
        else: sampler = None
        return DL(dataset, batch_size=8, num_workers=self.cfg.TRAINING.NUM_WORKERS, sampler=sampler, drop_last=True)

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
        nan_bool = torch.isnan(batch['abs_c_0'])
        if nan_bool.any():
            nan_inds = (nan_bool == True).nonzero(as_tuple=True)[0]
            nan_inds = torch.unique(nan_inds)
            for k in batch.keys():
                for n in nan_inds:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = torch.cat([batch[k][:n], batch[k][n+1:]])
                    elif isinstance(batch[k], list):
                        batch[k] = batch[k][:n] + batch[k][n+1:]

        nan_bool = torch.isnan(batch['abs_c_1'])
        if nan_bool.any():
            nan_inds = (nan_bool == True).nonzero(as_tuple=True)[0]
            nan_inds = torch.unique(nan_inds)
            for k in batch.keys():
                for n in nan_inds:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = torch.cat([batch[k][:n], batch[k][n+1:]])
                    elif isinstance(batch[k], list):
                        batch[k] = batch[k][:n] + batch[k][n+1:]

        self(batch)
        R_loss, t_loss, loss = self.loss_fn(batch)
        self.log('train/R_loss', R_loss), self.log('train/t_loss', t_loss), self.log('train/loss', loss)
        if self.LAMBDA == 0.: self.log('train/s_R', self.s_r), self.log('train/s_t', self.s_t)
        return loss

    def validation_step(self, batch, batch_idx):

        nan_bool = torch.isnan(batch['abs_c_0'])
        if nan_bool.any():
            nan_inds = (nan_bool == True).nonzero(as_tuple=True)[0]
            nan_inds = torch.unique(nan_inds)
            for k in batch.keys():
                for n in nan_inds:
                    if isinstance(batch[k], torch.Tensor): batch[k] = torch.cat([batch[k][:n], batch[k][n+1:]])
                    elif isinstance(batch[k], list): batch[k] = batch[k][:n] + batch[k][n+1:]

        nan_bool = torch.isnan(batch['abs_c_1'])
        if nan_bool.any():
            nan_inds = (nan_bool == True).nonzero(as_tuple=True)[0]
            nan_inds = torch.unique(nan_inds)
            for k in batch.keys():
                for n in nan_inds:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = torch.cat([batch[k][:n], batch[k][n+1:]])
                    elif isinstance(batch[k], list):
                        batch[k] = batch[k][:n] + batch[k][n+1:]

        Tgt = batch['T_0to1']
        R, t = self(batch)
        R_loss, t_loss, loss = self.loss_fn(batch)
        outputs = pose_error_torch(R, t, Tgt, reduce=None)

        for b_idx in range(len(batch['scene'])): 
            self.val_scenes.append({'scene': batch['scene'][b_idx], 'inds': (batch['inds'][b_idx].detach().cpu().numpy()[1], batch['inds'][b_idx].detach().cpu().numpy()[3]), 
                                    't_err': outputs['t_err_euc'][b_idx].item(), 'R_err': outputs['R_err'][b_idx].item(), 't_err_ang': outputs['t_err_ang'][b_idx].item(), 
                                    't_err_scale': outputs['t_err_scale'][b_idx].item(), 't_err_scale_sym': outputs['t_err_scale_sym'][b_idx].item(),
                                    'T_0to1': Tgt[b_idx]})


    def on_validation_epoch_end(self):

        # final = {scene: {'pairs': [{1: 0.24123, 2: 0.13512}], 'gt': (0.41261, 0.1262)}}
        aggregated = {}
        for v_out in self.val_scenes:
            scene = v_out['scene']

            if scene not in aggregated.keys():
                aggregated[scene] = {'inds': [], 't_err_euc': [], 'R_err': [], 't_err_ang': [], 't_err_scale': [], 't_err_scale_sym': []}

            aggregated[scene]['t_err_euc'].append(v_out['t_err'])
            aggregated[scene]['R_err'].append(v_out['R_err'])
            aggregated[scene]['t_err_ang'].append(v_out['t_err_ang'])
            aggregated[scene]['t_err_scale'].append(v_out['t_err_scale'])
            aggregated[scene]['t_err_scale_sym'].append(v_out['t_err_scale_sym'])
            aggregated[scene]['inds'].append(v_out['inds'])

        print(aggregated[list(aggregated.keys())[0]])
                    



        breakpoint()

        median_t_ang_err = aggregated['t_err_ang'].median()
        median_t_scale_err = aggregated['t_err_scale'].median()
        median_t_euclidean_err = aggregated['t_err_euc'].median()
        median_R_err = aggregated['R_err'].median()
        mean_R_loss = aggregated['R_loss'].mean()
        mean_t_loss = aggregated['t_loss'].mean()
        mean_loss = aggregated['loss'].mean()

        a1, a2, a3 = A_metrics(aggregated['t_err_scale_sym'])
        AUC_euc_10, AUC_euc_50, AUC_euc_100 = error_auc(aggregated['t_err_euc'].view(-1).detach().cpu().numpy(), [0.1, 0.5, 1.0]).values()
        pose_error = torch.maximum(aggregated['t_err_ang'].view(-1), aggregated['R_err'].view(-1)).detach().cpu()
        AUC_pos_5, AUC_pos_10, AUC_pos_20 = error_auc(pose_error.numpy(), [5, 10, 20]).values()
        rot_error = aggregated['R_err'].view(-1).detach().cpu()
        AUC_rot_5, AUC_rot_10, AUC_rot_20 = error_auc(rot_error.numpy(), [5, 10, 20]).values()
        t_ang_error = aggregated['t_err_ang'].view(-1).detach().cpu()
        AUC_tang_5, AUC_tang_10, AUC_tang_20 = error_auc(t_ang_error.numpy(), [5, 10, 20]).values()

        self.val_outputs.clear(), self.val_scenes.clear()
        return mean_loss

    def configure_optimizers(self):
        tcfg = self.cfg.TRAINING
        opt = torch.optim.Adam(self.parameters(), lr=tcfg.LR, eps=1e-6)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, verbose=True, patience=2)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sch, 'interval': 'epoch', 'monitor': 'train/loss'}}




