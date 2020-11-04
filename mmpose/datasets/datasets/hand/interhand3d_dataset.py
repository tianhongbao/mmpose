import os
from collections import OrderedDict

import json_tricks as json
import numpy as np

from mmpose.datasets.builder import DATASETS
from .hand_base_dataset import HandBaseDataset


@DATASETS.register_module()
class InterHand3DDataset(HandBaseDataset):
    """InterHand2.6M 3D dataset for top-down hand pose estimation.

    `InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose
    Estimation from a Single RGB Image' Moon, Gyeongsik etal. ECCV'2020
    More details can be found in the `paper
    <https://arxiv.org/pdf/2008.09309.pdf>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    InterHand2.6M keypoint indexes::

        0: 'r_thumb4',
        1: 'r_thumb3',
        2: 'r_thumb2',
        3: 'r_thumb1',
        4: 'r_forefinger4',
        5: 'r_forefinger3',
        6: 'r_forefinger2',
        7: 'r_forefinger1',
        8: 'r_middle_finger4',
        9: 'r_middle_finger3',
        10: 'r_middle_finger2',
        11: 'r_middle_finger1',
        12: 'r_ring_finger4',
        13: 'r_ring_finger3',
        14: 'r_ring_finger2',
        15: 'r_ring_finger1',
        16: 'r_pinky_finger4',
        17: 'r_pinky_finger3',
        18: 'r_pinky_finger2',
        19: 'r_pinky_finger1',
        20: 'r_wrist',
        21: 'l_thumb4',
        22: 'l_thumb3',
        23: 'l_thumb2',
        24: 'l_thumb1',
        25: 'l_forefinger4',
        26: 'l_forefinger3',
        27: 'l_forefinger2',
        28: 'l_forefinger1',
        29: 'l_middle_finger4',
        30: 'l_middle_finger3',
        31: 'l_middle_finger2',
        32: 'l_middle_finger1',
        33: 'l_ring_finger4',
        34: 'l_ring_finger3',
        35: 'l_ring_finger2',
        36: 'l_ring_finger1',
        37: 'l_pinky_finger4',
        38: 'l_pinky_finger3',
        39: 'l_pinky_finger2',
        40: 'l_pinky_finger1',
        41: 'l_wrist',

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (str): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 camera_file,
                 joint_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):
        super().__init__(
            ann_file, img_prefix, data_cfg, pipeline, test_mode=test_mode)

        self.ann_info['use_different_joint_weights'] = False
        assert self.ann_info['num_joints'] == 42
        self.ann_info['joint_weights'] = \
            np.ones((self.ann_info['num_joints'], 1), dtype=np.float32)
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {
            'right':
            np.arange(0, self.ann_info['num_joints'] // 2),
            'left':
            np.arange(self.ann_info['num_joints'] // 2,
                      self.ann_info['num_joints'])
        }

        self.dataset_name = 'interhand2d'
        self.camera_file = camera_file
        self.joint_file = joint_file
        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _cam2pixel(self, cam_coord, f, c):
        """Transform the joints from their camera coordinates to their pixel
        coordinates.

        Note:
            N: number of joints

        Args:
            cam_coord (ndarray[N, 3]): 3D joints coordinates
                in the camera coordinate system
            f (ndarray[2]): focal length of x and y axis
            c (ndarray[2]): principal point of x and y axis

        Returns:
            img_coord (ndarray[N, 3]): the coordinates (x, y, 0)
                in the image plane.
        """
        x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
        y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
        z = cam_coord[:, 2]
        img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
        return img_coord

    def _world2cam(self, world_coord, R, T):
        """Transform the joints from their world coordinates to their camera
        coordinates.

        Note:
            N: number of joints

        Args:
            world_coord (ndarray[3, N]): 3D joints coordinates
                in the world coordinate system
            R (ndarray[3, 3]): camera rotation matrix
            T (ndarray[3]): camera position (x, y, z)

        Returns:
            cam_coord (ndarray[3, N]): 3D joints coordinates
                in the camera coordinate system
        """
        cam_coord = np.dot(R, world_coord - T)
        return cam_coord

    def handtype_encoding(self, hand_type):
        if hand_type == 'right':
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0, 1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1, 1], dtype=np.float32)
        else:
            raise AssertionError('Not supported hand type: ' + hand_type)

    def _get_db(self):
        """Load dataset.

        Adapted from 'https://github.com/facebookresearch/InterHand2.6M/'
                        'blob/master/data/InterHand2.6M/dataset.py'
        Copyright (c) FaceBook Research, under CC-BY-NC 4.0 license.
        """
        with open(self.camera_file, 'r') as f:
            cameras = json.load(f)
        with open(self.joint_file, 'r') as f:
            joints = json.load(f)
        gt_db = []
        for img_id in self.img_ids:
            ann_id = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            ann = self.coco.loadAnns(ann_id)[0]
            img = self.coco.loadImgs(img_id)[0]

            capture_id = str(img['capture'])
            camera_name = img['camera']
            frame_idx = str(img['frame_idx'])
            image_file = os.path.join(self.img_prefix, self.id2name[img_id])

            camera_pos, camera_rot = np.array(
                cameras[capture_id]['campos'][camera_name],
                dtype=np.float32), np.array(
                    cameras[capture_id]['camrot'][camera_name],
                    dtype=np.float32)
            focal, princpt = np.array(
                cameras[capture_id]['focal'][camera_name],
                dtype=np.float32), np.array(
                    cameras[capture_id]['princpt'][camera_name],
                    dtype=np.float32)
            joint_world = np.array(
                joints[capture_id][frame_idx]['world_coord'], dtype=np.float32)
            joint_cam = self._world2cam(
                joint_world.transpose(1, 0), camera_rot,
                camera_pos.reshape(3, 1)).transpose(1, 0)
            # [u, v, d]
            joints_3d = self._cam2pixel(joint_cam, focal, princpt)

            joint_valid = np.array(
                ann['joint_valid'], dtype=np.float32).reshape(-1)
            # if root is not valid -> root-relative 3D pose is also not valid.
            # Therefore, mark all joints as invalid
            joint_valid[self.joint_type['right']] *= joint_valid[
                self.root_joint_idx['right']]
            joint_valid[self.joint_type['left']] *= joint_valid[
                self.root_joint_idx['left']]

            rec = []
            joints_3d[joint_valid == 0] = 0
            joints_3d_visible = np.minimum(1.0, joint_valid.reshape(-1, 1))

            rel_root_depth = np.array([
                joints_3d[self.root_joint_idx['left'], 2] -
                joints_3d[self.root_joint_idx['right'], 2]
            ],
                                      dtype=np.float32).reshape(1)

            joints_3d[self.joint_type['right'],
                      2] -= joints_3d[self.root_joint_idx['right'], 2]
            joints_3d[self.joint_type['left'],
                      2] -= joints_3d[self.root_joint_idx['left'], 2]

            hand_type = self.handtype_encoding(ann['hand_type'])

            # use 1.25bbox as input
            bbox = ann['bbox']
            center, scale = self._xywh2cs(*bbox, 1.25)

            rec.append({
                'image_file': image_file,
                'center': center,
                'scale': scale,
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'focal': focal,
                'princpt': princpt,
                'rel_root_depth': rel_root_depth,
                'hand_type': hand_type,
                'hand_type_valid': ann['hand_type_valid'],
                'dataset': self.dataset_name,
                'bbox': bbox,
                'bbox_score': 1
            })
            gt_db.extend(rec)

        return gt_db

    def evaluate(self, outputs, res_folder, metric='MPJPE', **kwargs):
        """Evaluate interhand2d keypoint results. The pose prediction results
        will be saved in `${res_folder}/result_keypoints.json`.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            outputs (list(preds, boxes, image_path, output_heatmap))
                :preds (np.ndarray[1,K,3]): The first two dimensions are
                    coordinates, score is the third dimension of the array.
                :boxes (np.ndarray[1,6]): [center[0], center[1], scale[0]
                    , scale[1],area, score]
                :image_path (list[str]): For example, ['C', 'a', 'p', 't',
                    'u', 'r', 'e', '1', '2', '/', '0', '3', '9', '0', '_',
                    'd', 'h', '_', 't', 'o', 'u', 'c', 'h', 'R', 'O', 'M',
                    '/', 'c', 'a', 'm', '4', '1', '0', '2', '0', '9', '/',
                    'i', 'm', 'a', 'g', 'e', '6', '2', '4', '3', '4', '.',
                    'j', 'p', 'g']
                :output_heatmap (np.ndarray[N, K, H, W]): model outpus.

            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed.
                Options: 'MPJPE', 'MRRPE', 'HAND_ACC'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['MPJPE', 'MRRPE', 'HAND_ACC']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')

        kpts = []

        for preds, boxes, image_path, _ in outputs:
            str_image_path = ''.join(image_path)
            image_id = self.name2id[str_image_path[len(self.img_prefix):]]

            kpts.append({
                'keypoints': preds[0].tolist(),
                'center': boxes[0][0:2].tolist(),
                'scale': boxes[0][2:4].tolist(),
                'area': float(boxes[0][4]),
                'score': float(boxes[0][5]),
                'image_id': image_id,
            })

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        return name_value
