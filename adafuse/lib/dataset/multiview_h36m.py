# ------------------------------------------------------------------------------
# multiview.pose3d.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import pickle
import collections

from dataset.joints_dataset import JointsDataset
import multiviews.cameras as cam_utils


class MultiViewH36M(JointsDataset):

    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.actual_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'neck',
            9: 'nose',
            10: 'head',
            11: 'lsho',
            12: 'lelb',
            13: 'lwri',
            14: 'rsho',
            15: 'relb',
            16: 'rwri'
        }

        self.u2a_mapping = super().get_mapping()

        grouping_db_pickle_file = osp.join(self.root, 'human36', 'quickload',
                                           'h36m_quickload_{}.pkl'
                                           .format(image_set))
        if osp.isfile(grouping_db_pickle_file) and False:
            with open(grouping_db_pickle_file, 'rb') as f:
                grouping_db = pickle.load(f)
                self.grouping = grouping_db['grouping']
                self.db = grouping_db['db']
        else:
            anno_file = osp.join(self.root, 'human36', 'annot',
                                 'h36m_{}.pkl'.format(image_set))
            self.db = self.load_db(anno_file)

            self.u2a_mapping = super().get_mapping()
            super().do_mapping()

            self.grouping = self.get_group(self.db)
            grouping_db_to_dump = {'grouping': self.grouping, 'db': self.db}
            #with open(grouping_db_pickle_file, 'wb') as f:
            #    pickle.dump(grouping_db_to_dump, f)
                
        keypoints = np.load("data_2d_h36m_cpn_ft_h36m_dbb.npz",
                            allow_pickle=True)
        keypoints = keypoints['positions_2d'].item()
        for item in range(len(self.db)):
            aa, bb ,cc=self.db[item]['subject'],self.db[item]['action'],self.db[item]['subaction']
            m2 = {}
            if aa==11 and bb==16 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'WalkTogether'
            if aa==11 and bb==16 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'WalkTogether 1'
            if aa==11 and bb==15 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'WalkDog'
            if aa==11 and bb==15 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'WalkDog 1'
            if aa==11 and bb==14 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Walking'
            if aa==11 and bb==14 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Walking 1'
            if aa==11 and bb==13 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Waiting'
            if aa==11 and bb==13 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Waiting 1'
            if aa==11 and bb==12 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Photo'
            if aa==11 and bb==12 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Photo 1'
            if aa==11 and bb==11 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Smoking'
            if aa==11 and bb==11 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Smoking 2'
            if aa==11 and bb==10 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'SittingDown 1'
            if aa==11 and bb==10 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'SittingDown'
            if aa==11 and bb==9 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Sitting'
            if aa==11 and bb==9 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Sitting 1'
            if aa==11 and bb==8 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Purchases'
            if aa==11 and bb==8 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Purchases 1'
            if aa==11 and bb==7 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Posing'
            if aa==11 and bb==7 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Posing 1'
            if aa==11 and bb==6 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Phoning 2'
            if aa==11 and bb==6 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Phoning 3'
            if aa==11 and bb==5 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Greeting'
            if aa==11 and bb==5 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Greeting 2'
            if aa==11 and bb==4 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Eating'
            if aa==11 and bb==4 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Eating 1'
            if aa==11 and bb==3 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Discussion 2'
            if aa==11 and bb==3 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Discussion 1'
            if aa==11 and bb==2 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Directions 1'
            
            if aa==9 and bb==16 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'WalkTogether'
            if aa==9 and bb==16 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'WalkTogether 1'
            if aa==9 and bb==15 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'WalkDog'
            if aa==9 and bb==15 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'WalkDog 1'
            if aa==9 and bb==14 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Walking'
            if aa==9 and bb==14 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Walking 1'
            if aa==9 and bb==13 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Waiting'
            if aa==9 and bb==13 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Waiting 1'
            if aa==9 and bb==12 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Photo'
            if aa==9 and bb==12 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Photo 1'
            if aa==9 and bb==11 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Smoking'
            if aa==9 and bb==11 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Smoking 1'
            if aa==9 and bb==10 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'SittingDown 1'
            if aa==9 and bb==10 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'SittingDown'
            if aa==9 and bb==9 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Sitting'
            if aa==9 and bb==9 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Sitting 1'
            if aa==9 and bb==8 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Purchases'
            if aa==9 and bb==8 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Purchases 1'
            if aa==9 and bb==7 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Posing'
            if aa==9 and bb==7 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Posing 1'
            if aa==9 and bb==6 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Phoning'
            if aa==9 and bb==6 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Phoning 1'
            if aa==9 and bb==5 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Greeting'
            if aa==9 and bb==5 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Greeting 1'
            if aa==9 and bb==4 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Eating'
            if aa==9 and bb==4 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Eating 1'
            if aa==9 and bb==3 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Discussion 2'
            if aa==9 and bb==3 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Discussion 1'
            if aa==9 and bb==2 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Directions'
            if aa==9 and bb==2 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Directions 1'

            if aa==8 and bb==16 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether 2'
            if aa==8 and bb==16 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether 1'
            if aa==8 and bb==15 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog'
            if aa==8 and bb==15 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog 1'
            if aa==8 and bb==14 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking'
            if aa==8 and bb==14 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking 1'
            if aa==8 and bb==13 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting'
            if aa==8 and bb==13 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting 1'
            if aa==8 and bb==12 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo'
            if aa==8 and bb==12 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo 1'
            if aa==8 and bb==11 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking'
            if aa==8 and bb==11 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking 1'
            if aa==8 and bb==10 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown 1'
            if aa==8 and bb==10 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown'
            if aa==8 and bb==9 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting'
            if aa==8 and bb==9 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting 1'
            if aa==8 and bb==8 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases'
            if aa==8 and bb==8 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases 1'
            if aa==8 and bb==7 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing'
            if aa==8 and bb==7 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing 1'
            if aa==8 and bb==6 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning'
            if aa==8 and bb==6 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning 1'
            if aa==8 and bb==5 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting'
            if aa==8 and bb==5 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting 1'
            if aa==8 and bb==4 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating'
            if aa==8 and bb==4 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating 1'
            if aa==8 and bb==3 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion'
            if aa==8 and bb==3 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion 1'
            if aa==8 and bb==2 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions'
            if aa==8 and bb==2 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions 1'
            
            if aa==7 and bb==16 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether'
            if aa==7 and bb==16 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether 1'
            if aa==7 and bb==15 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog'
            if aa==7 and bb==15 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog 1'
            if aa==7 and bb==14 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking 2'
            if aa==7 and bb==14 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking 1'
            if aa==7 and bb==13 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting 2'
            if aa==7 and bb==13 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting 1'
            if aa==7 and bb==12 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo 1'
            if aa==7 and bb==12 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo'
            if aa==7 and bb==11 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking'
            if aa==7 and bb==11 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking 1'
            if aa==7 and bb==10 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown 1'
            if aa==7 and bb==10 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown'
            if aa==7 and bb==9 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting'
            if aa==7 and bb==9 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting 1'
            if aa==7 and bb==8 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases'
            if aa==7 and bb==8 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases 1'
            if aa==7 and bb==7 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing'
            if aa==7 and bb==7 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing 1'
            if aa==7 and bb==6 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning'
            if aa==7 and bb==6 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning 2'
            if aa==7 and bb==5 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting'
            if aa==7 and bb==5 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting 1'
            if aa==7 and bb==4 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating'
            if aa==7 and bb==4 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating 1'
            if aa==7 and bb==3 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion'
            if aa==7 and bb==3 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion 1'
            if aa==7 and bb==2 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions'
            if aa==7 and bb==2 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions 1'
            
            if aa==6 and bb==16 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether'
            if aa==6 and bb==16 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether 1'
            if aa==6 and bb==15 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog'
            if aa==6 and bb==15 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog 1'
            if aa==6 and bb==14 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking'
            if aa==6 and bb==14 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking 1'
            if aa==6 and bb==13 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting'
            if aa==6 and bb==13 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting 3'
            if aa==6 and bb==12 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo 1'
            if aa==6 and bb==12 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo'
            if aa==6 and bb==11 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking'
            if aa==6 and bb==11 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking 1'
            if aa==6 and bb==10 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown'
            if aa==6 and bb==10 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown 1'
            if aa==6 and bb==9 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting 2'
            if aa==6 and bb==9 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting 1'
            if aa==6 and bb==8 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases'
            if aa==6 and bb==8 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases 1'
            if aa==6 and bb==7 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing'
            if aa==6 and bb==7 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing 2'
            if aa==6 and bb==6 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning'
            if aa==6 and bb==6 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning 1'
            if aa==6 and bb==5 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting'
            if aa==6 and bb==5 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting 1'
            if aa==6 and bb==4 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating 2'
            if aa==6 and bb==4 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating 1'
            if aa==6 and bb==3 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion'
            if aa==6 and bb==3 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion 1'
            if aa==6 and bb==2 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions'
            if aa==6 and bb==2 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions 1'
            
            if aa==5 and bb==16 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether'
            if aa==5 and bb==16 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether 1'
            if aa==5 and bb==15 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog'
            if aa==5 and bb==15 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog 1'
            if aa==5 and bb==14 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking'
            if aa==5 and bb==14 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking 1'
            if aa==5 and bb==13 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting 2'
            if aa==5 and bb==13 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting 1'
            if aa==5 and bb==12 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo 2'
            if aa==5 and bb==12 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo'
            if aa==5 and bb==11 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking'
            if aa==5 and bb==11 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking 1'
            if aa==5 and bb==10 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown 1'
            if aa==5 and bb==10 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown'
            if aa==5 and bb==9 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting'
            if aa==5 and bb==9 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting 1'
            if aa==5 and bb==8 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases'
            if aa==5 and bb==8 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases 1'
            if aa==5 and bb==7 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing'
            if aa==5 and bb==7 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing 1'
            if aa==5 and bb==6 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning'
            if aa==5 and bb==6 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning 1'
            if aa==5 and bb==5 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting 2'
            if aa==5 and bb==5 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting 1'
            if aa==5 and bb==4 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating'
            if aa==5 and bb==4 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating 1'
            if aa==5 and bb==3 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion 3'
            if aa==5 and bb==3 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion 2'
            if aa==5 and bb==2 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions 2'
            if aa==5 and bb==2 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions 1'
            
            if aa==1 and bb==16 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether'
            if aa==1 and bb==16 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether 1'
            if aa==1 and bb==15 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog'
            if aa==1 and bb==15 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog 1'
            if aa==1 and bb==14 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking'
            if aa==1 and bb==14 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking 1'
            if aa==1 and bb==13 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting'
            if aa==1 and bb==13 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting 1'
            if aa==1 and bb==12 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo'
            if aa==1 and bb==12 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo 1'
            if aa==1 and bb==11 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking'
            if aa==1 and bb==11 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking 1'
            if aa==1 and bb==10 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown'
            if aa==1 and bb==10 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown 2'
            if aa==1 and bb==9 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting 2'
            if aa==1 and bb==9 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting 1'
            if aa==1 and bb==8 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases'
            if aa==1 and bb==8 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases 1'
            if aa==1 and bb==7 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing'
            if aa==1 and bb==7 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing 1'
            if aa==1 and bb==6 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning'
            if aa==1 and bb==6 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning 1'
            if aa==1 and bb==5 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting'
            if aa==1 and bb==5 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting 1'
            if aa==1 and bb==4 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating'
            if aa==1 and bb==4 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating 2'
            if aa==1 and bb==3 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion'
            if aa==1 and bb==3 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion 1'
            if aa==1 and bb==2 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions'
            if aa==1 and bb==2 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions 1'
            
            #print(aa)
            #print(bb)
            #print(cc)
            joint_idx = [0,1,2,3,4,5,6,7,9,11,12,14,15,16,17,18,19]
            if 'subject' in m2:
                self.db[item]['joints_2d'][joint_idx] = keypoints[m2['subject']][m2['action']][self.db[item]['camera_id']][self.db[item]['image_id']-1][:,0:2]



        if self.is_train:
            self.grouping = self.grouping[::20]
        else:
            self.grouping2 = []
            cnt = 0
            le = len(self.grouping)
            for i in range(le):
                if cnt == 0 and i+4<le and self.db[self.grouping[i][0]]['image_id'] + 4 == self.db[self.grouping[i+4][0]]['image_id'] and self.db[self.grouping[i][0]]['image_id'] % 4 == 1:
                    cnt = 62
                    self.grouping2.append(self.grouping[i])
                    self.grouping2.append(self.grouping[i+1])
                    self.grouping2.append(self.grouping[i+2])
                    self.grouping2.append(self.grouping[i+3])
                    self.grouping2.append(self.grouping[i+4])
                    #print(i)
                elif cnt>0:
                    cnt = cnt - 1
                    
            self.grouping = self.grouping2

        self.group_size = len(self.grouping)
        self.selected_cam = [0,1,2,3]

    def index_to_action_names(self):
        return {
            2: 'Direction',
            3: 'Discuss',
            4: 'Eating',
            5: 'Greet',
            6: 'Phone',
            7: 'Photo',
            8: 'Pose',
            9: 'Purchase',
            10: 'Sitting',
            11: 'SittingDown',
            12: 'Smoke',
            13: 'Wait',
            14: 'WalkDog',
            15: 'Walk',
            16: 'WalkTwo'
        }

    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            return dataset

    def get_group(self, db):
        grouping = {}
        nitems = len(db)
        for i in range(nitems):
            keystr = self.get_key_str(db[i])
            if not osp.exists(self.root+'human36/images/'+ 's_{:02}_act_{:02}_subact_{:02}_ca_{:02}'.format(db[i]['subject'], db[i]['action'], db[i]['subaction'],db[i]['camera_id']+1)):
                continue
            #print(keystr)
            camera_id = db[i]['camera_id']
            if keystr not in grouping:
                grouping[keystr] = [-1, -1, -1, -1]
            grouping[keystr][camera_id] = i
            #print(grouping[keystr])

        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)

        return filtered_grouping

    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        items = self.grouping[idx]
        meta2 = []
        for item in items:
            i, t, w, m = super().__getitem__(item)
            # data type convert to float32
            m['scale'] = m['scale'].astype(np.float32)
            m['center'] = m['center'].astype(np.float32)
            m['rotation'] = int(m['rotation'])
            if 'name' in m['camera']:
                del m['camera']['name']
            for k in m['camera']:
                m['camera'][k] = m['camera'][k].astype(np.float32)

            aa, bb ,cc=self.db[item]['subject'],self.db[item]['action'],self.db[item]['subaction']
            m2 = {}
            if aa==11 and bb==16 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'WalkTogether'
            if aa==11 and bb==16 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'WalkTogether 1'
            if aa==11 and bb==15 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'WalkDog'
            if aa==11 and bb==15 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'WalkDog 1'
            if aa==11 and bb==14 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Walking'
            if aa==11 and bb==14 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Walking 1'
            if aa==11 and bb==13 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Waiting'
            if aa==11 and bb==13 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Waiting 1'
            if aa==11 and bb==12 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Photo'
            if aa==11 and bb==12 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Photo 1'
            if aa==11 and bb==11 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Smoking'
            if aa==11 and bb==11 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Smoking 2'
            if aa==11 and bb==10 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'SittingDown 1'
            if aa==11 and bb==10 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'SittingDown'
            if aa==11 and bb==9 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Sitting'
            if aa==11 and bb==9 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Sitting 1'
            if aa==11 and bb==8 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Purchases'
            if aa==11 and bb==8 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Purchases 1'
            if aa==11 and bb==7 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Posing'
            if aa==11 and bb==7 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Posing 1'
            if aa==11 and bb==6 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Phoning 2'
            if aa==11 and bb==6 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Phoning 3'
            if aa==11 and bb==5 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Greeting'
            if aa==11 and bb==5 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Greeting 2'
            if aa==11 and bb==4 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Eating'
            if aa==11 and bb==4 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Eating 1'
            if aa==11 and bb==3 and cc==2:
                m2['subject'] = 'S11'
                m2['action'] = 'Discussion 2'
            if aa==11 and bb==3 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Discussion 1'
            if aa==11 and bb==2 and cc==1:
                m2['subject'] = 'S11'
                m2['action'] = 'Directions 1'
            
            if aa==9 and bb==16 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'WalkTogether'
            if aa==9 and bb==16 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'WalkTogether 1'
            if aa==9 and bb==15 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'WalkDog'
            if aa==9 and bb==15 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'WalkDog 1'
            if aa==9 and bb==14 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Walking'
            if aa==9 and bb==14 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Walking 1'
            if aa==9 and bb==13 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Waiting'
            if aa==9 and bb==13 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Waiting 1'
            if aa==9 and bb==12 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Photo'
            if aa==9 and bb==12 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Photo 1'
            if aa==9 and bb==11 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Smoking'
            if aa==9 and bb==11 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Smoking 1'
            if aa==9 and bb==10 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'SittingDown 1'
            if aa==9 and bb==10 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'SittingDown'
            if aa==9 and bb==9 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Sitting'
            if aa==9 and bb==9 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Sitting 1'
            if aa==9 and bb==8 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Purchases'
            if aa==9 and bb==8 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Purchases 1'
            if aa==9 and bb==7 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Posing'
            if aa==9 and bb==7 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Posing 1'
            if aa==9 and bb==6 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Phoning'
            if aa==9 and bb==6 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Phoning 1'
            if aa==9 and bb==5 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Greeting'
            if aa==9 and bb==5 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Greeting 1'
            if aa==9 and bb==4 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Eating'
            if aa==9 and bb==4 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Eating 1'
            if aa==9 and bb==3 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Discussion 2'
            if aa==9 and bb==3 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Discussion 1'
            if aa==9 and bb==2 and cc==2:
                m2['subject'] = 'S9'
                m2['action'] = 'Directions'
            if aa==9 and bb==2 and cc==1:
                m2['subject'] = 'S9'
                m2['action'] = 'Directions 1'

            if aa==8 and bb==16 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether 2'
            if aa==8 and bb==16 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether 1'
            if aa==8 and bb==15 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog'
            if aa==8 and bb==15 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog 1'
            if aa==8 and bb==14 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking'
            if aa==8 and bb==14 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking 1'
            if aa==8 and bb==13 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting'
            if aa==8 and bb==13 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting 1'
            if aa==8 and bb==12 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo'
            if aa==8 and bb==12 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo 1'
            if aa==8 and bb==11 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking'
            if aa==8 and bb==11 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking 1'
            if aa==8 and bb==10 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown 1'
            if aa==8 and bb==10 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown'
            if aa==8 and bb==9 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting'
            if aa==8 and bb==9 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting 1'
            if aa==8 and bb==8 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases'
            if aa==8 and bb==8 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases 1'
            if aa==8 and bb==7 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing'
            if aa==8 and bb==7 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing 1'
            if aa==8 and bb==6 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning'
            if aa==8 and bb==6 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning 1'
            if aa==8 and bb==5 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting'
            if aa==8 and bb==5 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting 1'
            if aa==8 and bb==4 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating'
            if aa==8 and bb==4 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating 1'
            if aa==8 and bb==3 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion'
            if aa==8 and bb==3 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion 1'
            if aa==8 and bb==2 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions'
            if aa==8 and bb==2 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions 1'
            
            if aa==7 and bb==16 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether'
            if aa==7 and bb==16 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether 1'
            if aa==7 and bb==15 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog'
            if aa==7 and bb==15 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog 1'
            if aa==7 and bb==14 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking 2'
            if aa==7 and bb==14 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking 1'
            if aa==7 and bb==13 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting 2'
            if aa==7 and bb==13 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting 1'
            if aa==7 and bb==12 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo 1'
            if aa==7 and bb==12 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo'
            if aa==7 and bb==11 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking'
            if aa==7 and bb==11 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking 1'
            if aa==7 and bb==10 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown 1'
            if aa==7 and bb==10 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown'
            if aa==7 and bb==9 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting'
            if aa==7 and bb==9 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting 1'
            if aa==7 and bb==8 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases'
            if aa==7 and bb==8 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases 1'
            if aa==7 and bb==7 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing'
            if aa==7 and bb==7 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing 1'
            if aa==7 and bb==6 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning'
            if aa==7 and bb==6 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning 2'
            if aa==7 and bb==5 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting'
            if aa==7 and bb==5 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting 1'
            if aa==7 and bb==4 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating'
            if aa==7 and bb==4 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating 1'
            if aa==7 and bb==3 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion'
            if aa==7 and bb==3 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion 1'
            if aa==7 and bb==2 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions'
            if aa==7 and bb==2 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions 1'
            
            if aa==6 and bb==16 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether'
            if aa==6 and bb==16 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether 1'
            if aa==6 and bb==15 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog'
            if aa==6 and bb==15 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog 1'
            if aa==6 and bb==14 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking'
            if aa==6 and bb==14 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking 1'
            if aa==6 and bb==13 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting'
            if aa==6 and bb==13 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting 3'
            if aa==6 and bb==12 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo 1'
            if aa==6 and bb==12 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo'
            if aa==6 and bb==11 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking'
            if aa==6 and bb==11 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking 1'
            if aa==6 and bb==10 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown'
            if aa==6 and bb==10 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown 1'
            if aa==6 and bb==9 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting 2'
            if aa==6 and bb==9 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting 1'
            if aa==6 and bb==8 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases'
            if aa==6 and bb==8 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases 1'
            if aa==6 and bb==7 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing'
            if aa==6 and bb==7 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing 2'
            if aa==6 and bb==6 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning'
            if aa==6 and bb==6 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning 1'
            if aa==6 and bb==5 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting'
            if aa==6 and bb==5 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting 1'
            if aa==6 and bb==4 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating 2'
            if aa==6 and bb==4 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating 1'
            if aa==6 and bb==3 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion'
            if aa==6 and bb==3 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion 1'
            if aa==6 and bb==2 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions'
            if aa==6 and bb==2 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions 1'
            
            if aa==5 and bb==16 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether'
            if aa==5 and bb==16 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether 1'
            if aa==5 and bb==15 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog'
            if aa==5 and bb==15 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog 1'
            if aa==5 and bb==14 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking'
            if aa==5 and bb==14 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking 1'
            if aa==5 and bb==13 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting 2'
            if aa==5 and bb==13 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting 1'
            if aa==5 and bb==12 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo 2'
            if aa==5 and bb==12 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo'
            if aa==5 and bb==11 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking'
            if aa==5 and bb==11 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking 1'
            if aa==5 and bb==10 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown 1'
            if aa==5 and bb==10 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown'
            if aa==5 and bb==9 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting'
            if aa==5 and bb==9 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting 1'
            if aa==5 and bb==8 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases'
            if aa==5 and bb==8 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases 1'
            if aa==5 and bb==7 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing'
            if aa==5 and bb==7 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing 1'
            if aa==5 and bb==6 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning'
            if aa==5 and bb==6 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning 1'
            if aa==5 and bb==5 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting 2'
            if aa==5 and bb==5 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting 1'
            if aa==5 and bb==4 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating'
            if aa==5 and bb==4 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating 1'
            if aa==5 and bb==3 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion 3'
            if aa==5 and bb==3 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion 2'
            if aa==5 and bb==2 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions 2'
            if aa==5 and bb==2 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions 1'
            
            if aa==1 and bb==16 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether'
            if aa==1 and bb==16 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkTogether 1'
            if aa==1 and bb==15 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog'
            if aa==1 and bb==15 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'WalkDog 1'
            if aa==1 and bb==14 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking'
            if aa==1 and bb==14 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Walking 1'
            if aa==1 and bb==13 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting'
            if aa==1 and bb==13 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Waiting 1'
            if aa==1 and bb==12 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo'
            if aa==1 and bb==12 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Photo 1'
            if aa==1 and bb==11 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking'
            if aa==1 and bb==11 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Smoking 1'
            if aa==1 and bb==10 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown'
            if aa==1 and bb==10 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'SittingDown 2'
            if aa==1 and bb==9 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting 2'
            if aa==1 and bb==9 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Sitting 1'
            if aa==1 and bb==8 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases'
            if aa==1 and bb==8 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Purchases 1'
            if aa==1 and bb==7 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing'
            if aa==1 and bb==7 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Posing 1'
            if aa==1 and bb==6 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning'
            if aa==1 and bb==6 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Phoning 1'
            if aa==1 and bb==5 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting'
            if aa==1 and bb==5 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Greeting 1'
            if aa==1 and bb==4 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating'
            if aa==1 and bb==4 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Eating 2'
            if aa==1 and bb==3 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion'
            if aa==1 and bb==3 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Discussion 1'
            if aa==1 and bb==2 and cc==2:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions'
            if aa==1 and bb==2 and cc==1:
                m2['subject'] = 'S'+str(aa)
                m2['action'] = 'Directions 1'

            m2['frames'] = self.db[item]['image_id']
            m2['cameraID'] = self.db[item]['camera_id']

            #print(m2['subject'])
            #print(m2['action'])
            input.append(i)
            target.append(t)
            weight.append(w)
            meta.append(m)
            meta2.append(m2)
        return input, target, weight, meta, meta2

    def __len__(self):
        return self.group_size

    def get_key_str(self, datum):
        return 's_{:02}_act_{:02}_subact_{:02}_imgid_{:06}'.format(
            datum['subject'], datum['action'], datum['subaction'],
            datum['image_id'])

    def evaluate(self, pred, *args, **kwargs):
        pred = pred.copy()
        nview = 4
        if 'threshold' in kwargs:
            threshold = kwargs['threshold']
        else:
            threshold = 0.0125
        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))

        gt = []
        flat_items = []
        box_lengthes = []
        for items in self.grouping:
            for item in items:
                gt.append(self.db[item]['joints_2d'][su, :2])
                flat_items.append(self.db[item])
                boxsize = np.array(self.db[item]['scale']).sum() * 100.0  # crop img pixels
                box_lengthes.append(boxsize)
        gt = np.array(gt)
        if pred.shape[1] == 20:
            pred = pred[:, su, :2]
        elif pred.shape[1] == 17:
            pred = pred[:, :, :2]
        detection_threshold = np.array(box_lengthes).reshape((-1, 1)) * threshold

        distance = np.sqrt(np.sum((gt - pred)**2, axis=2))
        detected = (distance <= detection_threshold)

        joint_detection_rate = np.sum(detected, axis=0) / np.float(gt.shape[0])

        name_values = collections.OrderedDict()
        joint_names = self.actual_joints
        for i in range(len(a2u)):
            name_values[joint_names[sa[i]]] = joint_detection_rate[i]
        detected_int = detected.astype(np.int)
        nsamples, njoints = detected.shape
        per_grouping_detected = detected_int.reshape(nsamples // nview, nview * njoints)
        return name_values, np.mean(joint_detection_rate), per_grouping_detected

    def evaluate_3d(self, preds3d, thresholds=None):
        if thresholds is None:
            thresholds = [5., 10., 15., 20., 25., 50., 75., 100., 125., 150.,]

        gt3d = []
        for idx, items in enumerate(self.grouping):
            # note that h36m joints_3d is in camera frame
            db_rec = self.db[items[0]]
            j3d_global = cam_utils.camera_to_world_frame(db_rec['joints_3d'], db_rec['camera']['R'], db_rec['camera']['T'])
            gt3d.append(j3d_global)
        gt3d = np.array(gt3d)

        assert preds3d.shape == gt3d.shape, 'shape mismatch of preds and gt'
        distance = np.sum((preds3d - gt3d)**2, axis=2)

        num_groupings = len(gt3d)
        pcks = []
        for thr in thresholds:
            detections = distance <= thr**2
            detections_perjoint = np.sum(detections, axis=0)
            pck_perjoint = detections_perjoint / num_groupings
            # pck_avg = np.average(pck_perjoint, axis=0)
            pcks.append(pck_perjoint)

        return thresholds, pcks
