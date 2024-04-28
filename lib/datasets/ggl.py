from pathlib import Path
import torch
import torch.utils.data as data
import numpy as np
from transforms3d.quaternions import qinverse, qmult, rotate_vector, quat2mat


from lib.datasets.utils import read_color_image, read_depth_image, correct_intrinsic_scale

# from utils import read_color_image, read_depth_image, correct_intrinsic_scale



class GGLScene(data.Dataset):
    def __init__(self, scene_root, resize, sample_factor=1, overlap_limits=None, transforms=None,
                 estimated_depth=None):
        super().__init__()
        self.scene_root = Path(scene_root)
        self.resize = resize
        self.sample_factor = sample_factor
        self.transforms = transforms
        self.estimated_depth = estimated_depth

        self.poses = self.read_poses(self.scene_root)
        self.K = self.read_intrinsics(self.scene_root, resize)
        self.pairs = self.load_pairs(self.scene_root, overlap_limits, self.sample_factor)

    @staticmethod
    def read_intrinsics(scene_root: Path, resize=None):
        Ks = {}
        with (scene_root / 'intrinsics.txt').open('r') as f:
            for line in f.readlines():
                if '#' in line:
                    continue

                line = line.strip().split(' ')
                img_name = line[0]
                img_name = img_name.split('/')
                img_name = "/".join(img_name[-2:])
                fx, fy, cx, cy, W, H = map(float, line[1:])

                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                if resize is not None:
                    K = correct_intrinsic_scale(K, resize[0] / W, resize[1] / H)
                Ks[img_name] = K
        return Ks

    @staticmethod
    def read_poses(scene_root: Path):
        """
        Returns a dictionary that maps: img_path -> (q, t) where
        np.array q = (qw, qx qy qz) quaternion encoding rotation matrix;
        np.array t = (tx ty tz) translation vector;
        (q, t) encodes absolute pose (world-to-camera), i.e. X_c = R(q) X_W + t
        """
        poses = {}
        with (scene_root / 'poses.txt').open('r') as f:
            for line in f.readlines():
                if '#' in line:
                    continue

                line = line.strip().split(' ')
                img_name = line[0]

                # CUSTOM
                img_name = img_name.split('/')
                img_name = "/".join(img_name[-2:])

                qt = np.array(list(map(float, line[1:])))
                poses[img_name] = (qt[:4], qt[4:])
        return poses

    def load_pairs(self, scene_root: Path, overlap_limits: tuple = None, sample_factor: int = 1):
        """
        For training scenes, filter pairs of frames based on overlap (pre-computed in overlaps.npz)
        For test/val scenes, pairs are formed between keyframe and every other sample_factor query frames.
        If sample_factor == 1, all query frames are used. Note: sample_factor applicable only to test/val
        Returns:
        pairs: nd.array [Npairs, 4], where each column represents seqA, imA, seqB, imB, respectively
        """         

        seq_a = [int(fn[-9:-4]) for fn in self.poses.keys() if 'seq0' in fn]
        seq_b = [int(fn[-9:-4]) for fn in self.poses.keys() if 'seq0' not in fn]
        new_arr = []
        # seq_a pairs with values within 2 indices of each anchor
        for i_a, anchor in enumerate(seq_a):
            a_rems = seq_a[max(0, i_a-2):min(len(seq_a), i_a+3)]
            a_rems.remove(anchor)
            for i_an in a_rems:
                new_arr.append([0, i_a, 0, i_an])

        for i_a, anchor in enumerate(seq_a):
            for i_bn in seq_b:
                new_arr.append([0, i_a, 1, i_bn])

        new_arr = new_arr[::sample_factor]

        return np.array(new_arr, dtype=np.uint16)
        
    def get_pair_path(self, pair):
        seqA, imgA, seqB, imgB = pair
        return (f'seq{seqA}/frame_{imgA:05}.jpg', f'seq{seqB}/frame_{imgB:05}.jpg')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        im1_path, im2_path = self.get_pair_path(self.pairs[index])
        image1 = read_color_image(self.scene_root / im1_path, self.resize, augment_fn=self.transforms)
        image2 = read_color_image(self.scene_root / im2_path, self.resize, augment_fn=self.transforms)

        if self.estimated_depth is not None:
            dim1_path = str(self.scene_root / im1_path).replace('.jpg', f'.{self.estimated_depth}.png')
            dim2_path = str(self.scene_root / im2_path).replace('.jpg', f'.{self.estimated_depth}.png')
            depth1 = read_depth_image(dim1_path)
            depth2 = read_depth_image(dim2_path)
        else: depth1 = depth2 = torch.tensor([])

        # get absolute pose of im0 and im1
        q1, t1 = self.poses[im1_path]
        q2, t2 = self.poses[im2_path]
        c1 = rotate_vector(-t1, qinverse(q1))  # center of camera 1 in world coordinates)
        c2 = rotate_vector(-t2, qinverse(q2))  # center of camera 2 in world coordinates)

        # get 4 x 4 relative pose transformation matrix (from im1 to im2)
        # for test/val set, q1,t1 is the identity pose, so the relative pose matches the absolute pose
        # print(q1, q2)
        q12 = qmult(q2, qinverse(q1))


        t12 = t2 - rotate_vector(t1, q12)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = quat2mat(q12)
        T[:3, -1] = t12
        T = torch.from_numpy(T)
        return {'image0': image1, 'depth0': depth1, 'image1': image2, 'depth1': depth2, 'T_0to1': T, 'abs_q_0': q1, 
                'abs_c_0': c1, 'abs_q_1': q2, 'abs_c_1': c2, 'K_color0': self.K[im1_path].copy(), 'K_color1': self.K[im2_path].copy(), 
                'dataset_name': 'GGL', 'scene_id': self.scene_root.stem, 'scene_root': str(self.scene_root), 
                'pair_id': index*self.sample_factor, 'pair_names': (im1_path, im2_path), 'sim': 0.}


class GGLDataset(data.ConcatDataset):
    def __init__(self, cfg, mode, transforms=None):
        assert mode in ['train', 'val', 'test'], 'Invalid dataset mode'
        scenes = cfg.DATASET.SCENES
        data_root = Path(cfg.DATASET.DATA_ROOT) / mode
        resize = (cfg.DATASET.WIDTH, cfg.DATASET.HEIGHT)
        estimated_depth = cfg.DATASET.ESTIMATED_DEPTH
        overlap_limits = (cfg.DATASET.MIN_OVERLAP_SCORE, cfg.DATASET.MAX_OVERLAP_SCORE)
        sample_factor = {'train': 1, 'val': 5, 'test': 5}[mode]
        if scenes is None: scenes = [s.name for s in data_root.iterdir() if s.is_dir()]

        data_srcs = [GGLScene(data_root / scene, resize, sample_factor, overlap_limits, transforms, estimated_depth) for scene in scenes]

        super().__init__(data_srcs)


if __name__ == '__main__':
    import argparse
    from time import sleep
    from default import cfg
    torch.set_printoptions(linewidth=200)

    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='config/regression/ggl/3d3d.yaml', help='path to config file')
    parser.add_argument('dataset_config', default='config/ggl.yaml', help='path to dataset config file')
    parser.add_argument('--experiment', help='experiment name', default='default')
    parser.add_argument('--resume', help='resume from checkpoint path', default=None)
    args = parser.parse_args(['config/regression/ggl/3d3d.yaml', 'config/ggl.yaml', '--experiment', 'default'])
    cfg.merge_from_file(args.dataset_config)
    cfg.merge_from_file(args.config)

    dataset = GGLDataset(cfg, 'train')

    for i in range(0,100):    
        item = dataset[i]   
        for k in item.keys():
            if 'image' in k:
                print(f'{k}: {item[k].shape}')
            elif 'depth' not in k:
                print(f'{k}: {item[k]}')
        print()
        sleep(2)