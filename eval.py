import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from zipfile import ZipFile

import torch
import numpy as np
from tqdm import tqdm

from config.default import cfg
from lib.datasets.datamodules import DataModule
from lib.models.builder import build_model
from lib.utils.data import data_to_model_device
from transforms3d.quaternions import mat2quat


@dataclass
class Pose:
    image_name: str
    q: np.ndarray
    t: np.ndarray
    inliers: float

    def __str__(self) -> str:
        formatter = {'float': lambda v: f'{v:.6f}'}
        max_line_width = 1000
        q_str = np.array2string(self.q, formatter=formatter, max_line_width=max_line_width)[1:-1]
        t_str = np.array2string(self.t, formatter=formatter, max_line_width=max_line_width)[1:-1]
        return f'{self.image_name} {q_str} {t_str} {self.inliers}'


def predict(loader, model):
    results_dict = defaultdict(list)

    for data in tqdm(loader):
        # run inference
        translation_gt = data['abs_c_1']
        data = data_to_model_device(data, model)


        with torch.no_grad(): R, t = model(data)
        R = R.detach().cpu().numpy()
        t = t.reshape(-1).detach().cpu().numpy()
        inliers = data['inliers']
        scene = data['scene_id'][0]
        query_img = data['pair_names'][1][0]

        # ignore frames without poses (e.g. not enough feature matches)
        if np.isnan(R).any() or np.isnan(t).any() or np.isinf(t).any():
            continue

        # populate results_dict
        estimated_pose = Pose(image_name=query_img, q=mat2quat(R).reshape(-1), t=t.reshape(-1), inliers=inliers)
        # results_dict[scene].append(estimated_pose)
        trans_pred = estimated_pose.t

        results_dict[scene] = {'gt: ': translation_gt.detach().numpy().tolist(), 'pred': trans_pred.tolist()}

    return results_dict


def eval(config, checkpoint):
    cfg.merge_from_file('config/ggl.yaml')
    cfg.merge_from_file(config)

    cfg.TRAINING.BATCH_SIZE = 1
    cfg.TRAINING.NUM_WORKERS = 1
    dataloader = DataModule(cfg).val_dataloader()
    model = build_model(cfg, checkpoint)
    results_dict = predict(dataloader, model)

    for k in results_dict.keys():
        print(results_dict[k])



if __name__ == '__main__':
    config = 'config/regression/ggl/3d3d.yaml'
    checkpoint = 'weights/mapfree/3d3d.ckpt'
    eval(config, checkpoint)
