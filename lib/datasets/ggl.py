from pathlib import Path
import torch
import torch.utils.data as data
import numpy as np
from transforms3d.quaternions import qinverse, qmult, rotate_vector, quat2mat
import utm 
import math
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from tqdm import tqdm

if __name__ == '__main__':
    from utils import read_color_image, read_depth_image, correct_intrinsic_scale
    from database import ImageDatabase
else:
    from lib.datasets.utils import read_color_image, read_depth_image, correct_intrinsic_scale
    from lib.datasets.database import ImageDatabase

def weird_division(n, d):
    return n / d if d else 0

def calculate_initial_compass_bearing(start, end):
    lat1 = math.radians(start[0])
    lat2 = math.radians(end[0])
    diffLong = math.radians(end[1] - start[1])
    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))
    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return round(compass_bearing, 3)


class GGLScene(data.Dataset):
    def __init__(self, scene_root, scene, resize, sample_factor=1, overlap_limits=None, transforms=None,
                 estimated_depth=None, stage='train'):
        super().__init__()
        self.stage = stage
        self.scene = scene
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
        new_arr = []
        if self.stage == 'train':
            for i_a, anchor in enumerate(seq_a):
                a_rems = seq_a[max(0, i_a-2):min(len(seq_a), i_a+3)]
                a_rems.remove(anchor)
                for i_an in a_rems:
                    new_arr.append([0, i_a, 0, i_an])
        else:
            seq_b = [int(fn[-9:-4]) for fn in self.poses.keys() if 'seq0' not in fn]
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
        pairs = self.pairs[index]
        im1_path, im2_path = self.get_pair_path(pairs)
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

        # pyth_diff = np.hypot(t2[0], t2[1])

        # get 4 x 4 relative pose transformation matrix (from im1 to im2)
        # for test/val set, q1,t1 is the identity pose, so the relative pose matches the absolute pose
        # print(q1, q2)
        q12 = qmult(q2, qinverse(q1))
        t12 = t2 - rotate_vector(t1, q12)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = quat2mat(q12)
        T[:3, -1] = t12
        T = torch.from_numpy(T)

        # return {'image0': image1, 'depth0': depth1, 'image1': image2, 'depth1': depth2, 'T_0to1': T, 'abs_q_0': q1, 
        #         'abs_c_0': c1, 'abs_q_1': q2, 'abs_c_1': c2, 'K_color0': self.K[im1_path].copy(), 'K_color1': self.K[im2_path].copy(), 
        #         'dataset_name': 'GGL', 'scene_id': self.scene_root.stem, 'scene_root': str(self.scene_root), 
        #         'pair_id': index*self.sample_factor, 'pair_names': (im1_path, im2_path), 'sim': 0., 'scene': self.scene, 'inds': pairs,
        #         'gt': pyth_diff, 't1': t1, 't2': t2}

        data_dict = {'image0': image1, 'image1': image2, 'T_0to1': T, 'abs_q_0': q1, 'abs_c_0': c1, 'abs_q_1': q2, 'abs_c_1': c2,
                'K_color0': self.K[im1_path].copy(), 'K_color1': self.K[im2_path].copy(), 'scene_id': self.scene_root.stem, 
                'scene_root': str(self.scene_root), 'pair_id': index*self.sample_factor, 'pair_names': (im1_path, im2_path), 
                'scene': self.scene, 'inds': pairs}
        return data_dict

class GGLDataset(data.ConcatDataset):
    def __init__(self, cfg, mode, transforms=None):
        assert mode in ['train', 'val', 'test'], 'Invalid dataset mode'
        scenes = cfg.DATASET.SCENES
        data_root = Path(cfg.DATASET.DATA_ROOT) / 'train'
        resize = (cfg.DATASET.WIDTH, cfg.DATASET.HEIGHT)
        estimated_depth = cfg.DATASET.ESTIMATED_DEPTH
        overlap_limits = (cfg.DATASET.MIN_OVERLAP_SCORE, cfg.DATASET.MAX_OVERLAP_SCORE)
        sample_factor = {'train': 1, 'val': 1, 'test': 5}[mode]
        if scenes is None: scenes = [s.name for s in data_root.iterdir() if s.is_dir()]

        data_srcs = [GGLScene(data_root / scene, scene, resize, sample_factor, overlap_limits, transforms, estimated_depth, mode) for scene in scenes]
        super().__init__(data_srcs)


class GraphDataset():
    def __init__(self, cfg=None, path: Path = Path('data')):
        self.cities = [torch.load(str(g)) for g in Path(path / 'graphs').glob('*.pt')]
        self.graph_to_scenes()
        self.load_pairs()
        self.lmdb = ImageDatabase(path / 'lmdb')
        self.fov = 90
        self.cfg = cfg

    def graph_to_scenes(self):
        self.scenes = {}
        for g_idx, graph in enumerate(self.cities):
            for idx, e in enumerate(graph.edges): # TEMPORARILY ONLY USING FIRST EDGE
                if idx == 0:

                    if len(graph.edges[e]['images']) > 2: # edges have images within
                        node_a_x, node_a_y = utm.from_latlon(*graph.edges[e]['images'][0]['point'])[:2]
                        node_b_x, node_b_y = utm.from_latlon(*graph.edges[e]['images'][-1]['point'])[:2]
                        node_b_x = abs(node_b_x - node_a_x)
                        node_b_y = abs(node_b_y - node_a_y)

                        if node_b_x and node_b_y:
                            sub_nodes = graph.edges[e]['images']
                            scene = {'images': [], 'norths': [], 'pos_x': [], 'pos_y': []}

                            for node in sub_nodes:
                                n_pos = np.subtract(utm.from_latlon(*node['point'])[:2], (node_a_x, node_a_y))
                                rel_pos_x = abs(round(weird_division(n_pos[0], node_b_x), 8))
                                rel_pos_y = abs(round(weird_division(n_pos[1], node_b_y), 8))
                                node_image = f'{node["point"]}_{node["north"]}'
                                scene['images'].append(node_image), scene['pos_x'].append(rel_pos_x), scene['pos_y'].append(rel_pos_y)
                                scene['norths'].append(node['north'])
                            query = graph.edges[e]['query']
                            query_pos = np.subtract(utm.from_latlon(*query['point'])[:2], (node_a_x, node_a_y))
                            rel_pos_x = abs(round(weird_division(query_pos[0], node_b_x), 8))
                            rel_pos_y = abs(round(weird_division(query_pos[1], node_b_y), 8))
                            query_image = f'{query["point"]}_{query["north"]}'
                            scene['query'] = {'image': query_image, 'pos_x': rel_pos_x, 'pos_y': rel_pos_y}  
                            scene['graph'] = g_idx                    
                            self.scenes[e] = scene

    def load_pairs(self):
        self.train_pairs, self.val_pairs = [], []
        for s in self.scenes:
            imgs = self.scenes[s]['images']
            node_1, node_2 = self.cities[self.scenes[s]['graph']].edges[s]['nodes'][0], self.cities[self.scenes[s]['graph']].edges[s]['nodes'][-1]
            edge_angle = calculate_initial_compass_bearing(node_1, node_2)

            for i_a, img in enumerate(imgs):
                a_rems = imgs[max(0, i_a-1):min(len(imgs), i_a+2)]
                a_rems.remove(img)
                for i_an in a_rems:
                    self.val_pairs.append({'scene': s, 'imgs': [imgs.index(i_an)], 'heading': edge_angle})


class GraphPoseDataset(data.Dataset):
    def __init__(self, stage='train', dataset: GraphDataset = None) -> None:
        super().__init__()
        self.stage = stage
        self.dataset = dataset
        resize = (dataset.cfg.DATASET.WIDTH, dataset.cfg.DATASET.HEIGHT)
        self.normalise = Compose([ToTensor(), Resize(resize)])#, Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        if stage == 'train': self.pairs = self.dataset.train_pairs
        else: self.pairs = self.dataset.val_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        scene = self.dataset.scenes[pair['scene']]
        indices = pair['imgs']
        heading = pair['heading']
        query = scene['query']
        query_north = float(query['image'].split('_')[-1])
        image_query = np.array(self.dataset.lmdb[query['image']].convert('RGB'))
        image_ref = np.array(self.dataset.lmdb[scene['images'][indices[0]]].convert('RGB'))

        # Heading Roll
        _, width = image_query.shape[:2]
        query_north = int((query_north / 360) * width)
        anchor_angle = int((heading / 360) * width)
        image_query = np.roll(image_query, query_north, axis=1)
        image_query = np.roll(image_query, -anchor_angle, axis=1)
        ref_north = scene['norths'][indices[0]]
        _, width = image_query.shape[:2]
        ref_north = int((ref_north / 360) * width)
        anchor_angle = int((heading / 360) * width)
        image_ref = np.roll(image_ref, ref_north, axis=1)
        image_ref = np.roll(image_ref, -anchor_angle, axis=1)

        # FOV Crop
        new_half_width = int((width * (self.dataset.fov/360))/2)
        image_query = image_query[:, (width//2)-new_half_width:(width//2)+new_half_width]
        image_ref = image_ref[:, (width//2)-new_half_width:(width//2)+new_half_width]
        image1 = self.normalise(image_query)
        image2 = self.normalise(image_ref)

        pos_x_1 = query['pos_x']
        pos_y_1 = query['pos_y']
        pos_x_2 = scene['pos_x'][indices[0]]
        pos_y_2 = scene['pos_y'][indices[0]]
        t1 = torch.tensor([pos_x_1, pos_y_1, 0], dtype=torch.float32)
        q1 = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        t2 = torch.tensor([pos_x_2, pos_y_2, 0], dtype=torch.float32)
        q2 = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        c1 = rotate_vector(-t1, qinverse(q1))
        c2 = rotate_vector(-t2, qinverse(q2))  
        q12 = qmult(q2, qinverse(q1))
        t12 = t2 - rotate_vector(t1, q12)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = quat2mat(q12)
        T[:3, -1] = t12
        T = torch.from_numpy(T)

        K = np.array([[295.191, 0., 511.75], [0., 295.191, 127.75], [0., 0., 1.]], dtype=np.float32) # ESTIMATED
        data_dict = {'image0': image1, 'image1': image2, 'T_0to1': T, 'abs_q_0': q1, 'abs_c_0': c1, 'abs_q_1': q2, 'abs_c_1': c2, 
                'K_color0': K, 'K_color1': K, 'scene': pair['scene'], 'inds': pair['imgs']} 
        return data_dict     


if __name__ == '__main__':
    import argparse
    from default import cfg

    stringer = 'ggl'
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default=f'config/regression/{stringer}/3d3d.yaml', help='path to config file')
    parser.add_argument('dataset_config', default=f'config/{stringer}.yaml', help='path to dataset config file')
    parser.add_argument('--experiment', help='experiment name', default='default')
    parser.add_argument('--resume', help='resume from checkpoint path', default=None)
    args = parser.parse_args([f'config/regression/{stringer}/3d3d.yaml', f'config/{stringer}.yaml', '--experiment', 'default'])

    cfg.merge_from_file(args.dataset_config)
    cfg.merge_from_file(args.config)

    da = GraphDataset(cfg)
    dataset = GraphPoseDataset('val', da)

    # 'image0': image1, 'image1': image2, 'T_0to1': T, 'abs_q_0': q1, 'abs_c_0': c1, 'abs_q_1': q2, 'abs_c_1': c2, 
    # 'K_color0': K, 'K_color1': K, 'scene': pair['scene'], 'pair': pair['imgs']
    item = dataset.__getitem__(1)

    t_01 = item['T_0to1'] # relative pose 0 to 1?
    q_0 = item['abs_q_0'] # no rotational difference
    c_0 = item['abs_c_0'] # image 0 x,y,z
    q_1 = item['abs_q_1'] # no rotational difference
    c_1 = item['abs_c_1'] # image 1 x,y,z
    K_0 = item['K_color0'] # intrinsic matrix
    K_1 = item['K_color1'] # intrinsic matrix
    scene = item['scene'] # edge id
    pair = item['pair'] # reference image index in edge

    print(t_01)
    print(q_0)
    print(c_0)
    print(q_1)
    print(c_1)
    print(K_0)
    print(K_1)
    print(scene)
    print(pair)

