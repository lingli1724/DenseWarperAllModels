from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import pickle
import collections

from dataset.joints_dataset_3dhp import JointsDataset3dhp

class MultiView3dhp(JointsDataset3dhp):

    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.actual_joints = {
            0: 'pelvis',
            1: 'head_top',
            2: 'neck',
            3: 'right_shoulder',
            4: 'right_elbow',
            5: 'right_wrist',
            6: 'left_shoulder',
            7: 'left_elbow',
            8: 'left_wrist',
            9: 'right_hip',
            10: 'right_knee',
            11: 'right_ankle',
            12: 'left_hip',
            13: 'left_knee',
            14: 'left_ankle',
            15: 'spine',
            16: 'head',
        }

        if cfg.DATASET.CROP:
            anno_file = osp.join(self.root, 'mpi-inf-3dhp', 'annot',
                                 '3dhp_{}.pkl'.format(image_set))
        else:
            anno_file = osp.join(self.root, 'mpi-inf-3dhp', 'annot',
                                 '3dhp_{}_uncrop.pkl'.format(image_set))
            
        self.db = self.load_db(anno_file)

        if not cfg.DATASET.WITH_DAMAGE:
            print('before filter', len(self.db))
            self.db = [db_rec for db_rec in self.db if not self.isdamaged(db_rec)]
            print('after filter', len(self.db))

        self.u2a_mapping = super().get_mapping()
        super().do_mapping()

        self.grouping = self.get_group(self.db)
        self.group_size = len(self.grouping)

    def index_to_action_names(self):
        return {
            1: 'Walking/Standing',
            2: 'Exercise',
            3: 'Sitting(1)',
            4: 'Crouch/Reach',
            5: 'On the Floor',
            6: 'Sports',
            7: 'Sitting(2)',
            8: 'Miscellaneous',
        }
    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            return dataset
        
    def get_group(self, db):
        grouping = {}
        nitems = len(db)
        mpp={}
        mpp[0] = 0
        mpp[2] = 1
        mpp[7] = 2
        mpp[8] = 3
        for i in range(nitems):
            keystr ='s_{:01}_seq_{:01}_imgid_{:06}'.format(db[i]['action'], db[i]['seq'],db[i]['image_id'])
            #print(keystr)
            if not osp.exists(self.root+'mpi-inf-3dhp/images/'+ 's_{:01}_seq_{:01}_ca_{:01}'.format(db[i]['action'], db[i]['seq'],db[i]['camera_id'])):
                continue

            camera_id = mpp[db[i]['camera_id']]
            if keystr not in grouping:
                grouping[keystr] = [-1, -1, -1, -1]
            grouping[keystr][camera_id] = i

        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)

        if self.is_train:
            filtered_grouping = filtered_grouping[::]
        else:
            filtered_grouping = filtered_grouping[::]

        return filtered_grouping
    
    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        items = self.grouping[idx]
        for item in items:
            i, t, w, m = super().__getitem__(item)
            input.append(i)
            target.append(t)
            weight.append(w)
            meta.append(m)
        return input, target, weight, meta
    
    def __len__(self):
        return self.group_size
    
    def get_key_str(self, datum):
        return 's_{:01}_seq_{:01}_imgid_{:06}'.format(
            datum['action'], datum['seq'],
            datum['image_id'])
    
    def evaluate(self, pred, *args, **kwargs):
        pred = pred.copy()

        headsize = self.image_size[0] / 10.0
        threshold = 0.5

        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))    # [ 0  1  2  3  4  5  6  7  9 11 12 14 15 16 17 18 19]

        gt = []
        for items in self.grouping:
            for item in items:
                gt.append(self.db[item]['joints_2d'][su, :2])       # (17, 2) in original scale
        gt = np.array(gt)           # (num_sample, 17, 2) in original scale
        pred = pred[:, su, :2]      # (num_sample, 17, 2) in original scale

        distance = np.sqrt(np.sum((gt - pred)**2, axis=2))
        detected = (distance <= headsize * threshold)

        joint_detection_rate = np.sum(detected, axis=0) / np.float(gt.shape[0])

        name_values = collections.OrderedDict()
        joint_names = self.actual_joints
        for i in range(len(a2u)):
            name_values[joint_names[sa[i]]] = joint_detection_rate[i]
        return name_values, np.mean(joint_detection_rate)

    def isdamaged(self, db_rec):
        return False