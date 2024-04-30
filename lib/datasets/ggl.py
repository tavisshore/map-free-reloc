from pathlib import Path
import torch
import torch.utils.data as data
import numpy as np
from transforms3d.quaternions import qinverse, qmult, rotate_vector, quat2mat
import utm 
import math
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

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
    def __init__(self, scene_root, resize, sample_factor=1, overlap_limits=None, transforms=None,
                 estimated_depth=None, stage='train'):
        super().__init__()
        self.stage = stage
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
        data_root = Path(cfg.DATASET.DATA_ROOT) / 'train'
        resize = (cfg.DATASET.WIDTH, cfg.DATASET.HEIGHT)
        estimated_depth = cfg.DATASET.ESTIMATED_DEPTH
        overlap_limits = (cfg.DATASET.MIN_OVERLAP_SCORE, cfg.DATASET.MAX_OVERLAP_SCORE)
        sample_factor = {'train': 1, 'val': 1, 'test': 5}[mode]
        if scenes is None: scenes = [s.name for s in data_root.iterdir() if s.is_dir()]

        data_srcs = [GGLScene(data_root / scene, resize, sample_factor, overlap_limits, transforms, estimated_depth, mode) for scene in scenes]

        super().__init__(data_srcs)


class GraphDataset():
    def __init__(self, path: Path = Path('data')):
        self.cities = [torch.load(str(g)) for g in Path(path / 'graphs').glob('*.pt')]
        self.graph_to_scenes()
        self.load_pairs()
        self.lmdb = ImageDatabase(path / 'lmdb')
        self.fov = 90
        self.img_dim = (256, 256)

    def graph_to_scenes(self):
        self.scenes = {}
        for g_idx, graph in enumerate(self.cities):
            for e in graph.edges:
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
                a_rems = imgs[max(0, i_a-2):min(len(imgs), i_a+3)]
                a_rems.remove(img)
                for i_an in a_rems:
                    self.train_pairs.append({'scene': s, 'imgs': [imgs.index(img), imgs.index(i_an)], 'heading': edge_angle})


class GraphPoseDataset(data.Dataset):
    def __init__(self, stage='train', dataset: GraphDataset = None) -> None:
        super().__init__()
        self.stage = stage
        self.dataset = dataset
        self.transforms = None
        self.normalise = Compose([ToTensor(), Resize(self.dataset.img_dim), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        if stage == 'train': self.pairs = self.dataset.train_pairs
        else: self.pairs = self.dataset.val_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        scene = self.dataset.scenes[pair['scene']]
        indices = pair['imgs']
        heading = pair['heading']
        image_query = np.array(self.dataset.lmdb[scene['images'][indices[0]]].convert('RGB'))
        image_ref = np.array(self.dataset.lmdb[scene['images'][indices[1]]].convert('RGB'))
        # Heading Roll
        query_north = scene['norths'][indices[0]]
        _, width = image_query.shape[:2]
        query_north = int((query_north / 360) * width)
        anchor_angle = int((heading / 360) * width)
        image_query = np.roll(image_query, query_north, axis=1)
        image_query = np.roll(image_query, -anchor_angle, axis=1)
        ref_north = scene['norths'][indices[1]]
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

        pos_x_1 = scene['pos_x'][indices[0]]
        pos_y_1 = scene['pos_y'][indices[0]]
        pos_x_2 = scene['pos_x'][indices[1]]
        pos_y_2 = scene['pos_y'][indices[1]]
        t1 = torch.tensor([pos_x_1, pos_y_1, 0], dtype=torch.float32)
        q1 = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        t2 = torch.tensor([pos_x_2, pos_y_2, 0], dtype=torch.float32)
        q2 = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        # From works code
        c1 = rotate_vector(-t1, qinverse(q1))  # center of camera 1 in world coordinates)
        c2 = rotate_vector(-t2, qinverse(q2))  # center of camera 2 in world coordinates)
        q12 = qmult(q2, qinverse(q1))
        t12 = t2 - rotate_vector(t1, q12)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = quat2mat(q12)
        T[:3, -1] = t12
        T = torch.from_numpy(T)

        return {'image0': image1, 'depth0': torch.tensor([]), 'image1': image2, 'depth1': torch.tensor([]), 'T_0to1': T, 'abs_q_0': q1, 
                'abs_c_0': c1, 'abs_q_1': q2, 'abs_c_1': c2} #, 'K_color0': self.K[im1_path].copy(), 'K_color1': self.K[im2_path].copy(), 
                # 'dataset_name': 'GGL', 'scene_id': self.scene_root.stem, 'scene_root': str(self.scene_root), 
                # 'pair_id': index*self.sample_factor, 'pair_names': (im1_path, im2_path), 'sim': 0.}



if __name__ == '__main__':
    d = GraphPoseDataset()
    item = d.__getitem__(0)
