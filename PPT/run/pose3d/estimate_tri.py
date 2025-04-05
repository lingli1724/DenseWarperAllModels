
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
import h5py
import pickle
import json
import argparse
import numpy as np

import _init_paths
from core.inference import get_max_preds
from core.config import config
from core.config import update_config
from utils.utils import create_logger
from multiviews.pictorial import rpsm
from multiviews.body import HumanBody
from multiviews.cameras import camera_to_world_frame
from multiviews.triangulate import triangulate_poses
from core.interpolation import *
import dataset
import models

CUDA_ID = [0, 1, 2, 3]
device = torch.device("cuda")

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate 3D Pose Estimation')
    parser.add_argument(
        '--cfg', help='configuration file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def compute_limb_length(body, pose):
    limb_length = {}
    skeleton = body.skeleton
    for node in skeleton:
        idx = node['idx']
        children = node['children']

        for child in children:
            length = np.linalg.norm(pose[idx] - pose[child])
            limb_length[(idx, child)] = length
    return limb_length


def main():
    answer={}
    nviews = 4
    njoints = 17
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'test3d-tri')

    prediction_path = os.path.join(final_output_dir,
                                   config.TEST.HEATMAP_LOCATION_FILE)
    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False)

    all_locations = h5py.File(prediction_path)['locations']     # (#sample, 17, 2)
    all_heatmaps = h5py.File(prediction_path)['heatmaps']       # (#sample, 17, 64, 64)

    cnt = 0
    grouping = test_dataset.grouping
    mpjpes = []
    mpjpe_score_output = {}
    pose3d_output = {}
    for items in grouping:      # all 4 views for one subject
        heatmaps = []
        locations = []
        poses = []          # save same 3D world coordinate 4 times
        cameras = []
        #print(test_dataset.db[idx]['subject'] and )
        # get information of all 4 views
        for idx in items:   #
            datum = test_dataset.db[idx]
            camera = datum['camera']        # dict: {R, T, fx, fy, cx, cy}
            cameras.append(camera)

            poses.append(
                camera_to_world_frame(datum['joints_3d_camera'], camera['R'],
                                      camera['T']))       # pose in 3D world coordinate (17, 3)

            locations.append(all_locations[cnt])
            heatmaps.append(all_heatmaps[cnt])
            cnt += 1
        # s_11_act_16_subact_01_ca_04/s_11_act_16_subact_01_ca_04_000090.jpg
        keypoint_vis = datum['joints_vis']  # (20, 3)
        u2a = test_dataset.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        u = np.array(list(a2u.values()))
        keypoint_vis = keypoint_vis[u]          # (17, 3)

        locations = np.array(locations)[:, :, :2]               # (#view, 17, 2) in original scale
        heatmaps = np.array(heatmaps)                           # (#view, 17, 64, 64)
        _, confs = get_max_preds(heatmaps)                      # (#view, 17, 1)

        prediction = triangulate_poses(cameras, locations, confs.squeeze())      # list, element: (17, 3)
        prediction[0] = prediction[0] - prediction[0][0]
        poses[0] = poses[0] - poses[0][0]
        mpjpe = np.mean(np.sqrt(np.sum((prediction[0] * keypoint_vis - poses[0] * keypoint_vis)**2, axis=1)))
        #print(mpjpe)
        seq_, frame_ = datum['image'].split('/')[1].split('_ca_')
        subject = int(seq_[2:4])
        action = seq_[5:]
        frames = int(frame_[3:-4])
        if subject not in answer:
            answer[subject]={}
        if action not in answer[subject]:
            answer[subject][action]={} 
        if frames not in answer[subject][action]:
            answer[subject][action][frames]={}
        answer[subject][action][frames]['predicted']=(torch.from_numpy(prediction[0]).float().to(device))/1000
        answer[subject][action][frames]['groundtruth']=(torch.from_numpy(poses[0]).float().to(device))/1000

        mpjpes.append(mpjpe)
        # print(mpjpe)
        if mpjpe > 150:
            print('Wrong MPJPE !!! ', datum['image'])

        # ================== save MPJPE score ==================
        if config.DATASET.TRAIN_DATASET == 'multiview_h36m':
            seq, frame = datum['image'].split('/')[1].split('_ca_')      # s_11_act_16_subact_01, 04_000090.jpg
            frame_name = seq + frame[2:-4]
            mpjpe_score_output[frame_name] = mpjpe

            # ================== save 3D pose ================
            pose3d_output[frame_name] = {}
            pose3d_output[frame_name]['pred'] = prediction[0].tolist()      # from numpy to list
            pose3d_output[frame_name]['GT'] = poses[0].tolist()

    logger.info('Triangulation MPJPE {}'.format(np.mean(mpjpes)))
    
    if config.DATASET.TRAIN_DATASET == 'multiview_h36m':
        json.dump(mpjpe_score_output, open(os.path.join(final_output_dir, 'mpjpe_score.json'), 'w'), indent=4, sort_keys=True)
        json.dump(pose3d_output, open(os.path.join(final_output_dir, 'output_3d_joint.json'), 'w'), indent=4, sort_keys=True)

        # MPJPE on Each Action Sequence
        action_map = test_dataset.index_to_action_names()
        avg = []
        avg_pose_seq = {}
        for k in action_map:
            avg_pose_seq[k] = []

        for frame in mpjpe_score_output:
            # frame name: s_09_act_02_subact_01_000001
            act = frame[9:11]
            act = int(act)
            avg_pose_seq[act].append(mpjpe_score_output[frame])

            avg.append(mpjpe_score_output[frame])

        pose_seq_out_str = "\n"
        for k in action_map:
            res = avg_pose_seq[k]
            mpjpe = sum(res) / len(res)
            pose_seq_out_str = pose_seq_out_str + action_map[k] + '\t{}\n'.format(mpjpe)

        logger.info('MPJPE on each Action: ' + pose_seq_out_str)

        print(sum(avg) / len(avg))


    from tqdm import tqdm
    model_train_interpolator = AutoEncoder(3, njoints, 2, nviews+1)
    model_train_interpolator=torch.nn.DataParallel(model_train_interpolator, device_ids=CUDA_ID).to(device)
    model_train_interpolator=model_train_interpolator.to(device)

###########################################################################################################################
    pre_dict = torch.load('MCC.pth')
    model_dict = model_train_interpolator.state_dict()
    state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model_train_interpolator.load_state_dict(model_dict)

    cnt = 0
    import copy
    last = copy.deepcopy(answer)
    interpolator=model_train_interpolator
    interpolator.eval()
    mpjpe_interpolation={}
    cnt_action={}
    for subject in tqdm(answer.keys(), desc="Subjects"):
        for action in tqdm(answer[subject].keys(), desc="Actions", leave=False):
        
            maxx = 0
            for frame in sorted(answer[subject][action].keys()):
                if (frame-1)%nviews==0 and (frame-1)!=0:
                    source=answer[subject][action][frame-nviews]['predicted'].clone().unsqueeze(0).to(device)#(1,17,3)
                    target=answer[subject][action][frame]['predicted'].clone().unsqueeze(0).to(device)
                    input__ = torch.stack((source, target), dim=1).permute(0,3,2,1)#(1,2,17,3)
                    interpolation=interpolator(input__).clone()
                    interpolation=interpolation.permute(0,3,2,1)
                    for num in range(nviews+1):
                        answer[subject][action][frame-nviews+num]['predicted']=interpolation[0][num]
                    maxx=frame

            for frame in sorted(answer[subject][action].keys()):
                if frame > maxx:
                    break
                end_index = action.find(' ')
                if end_index != -1:
                    action_name = action[:end_index]
                else:
                    action_name = action
                
                if action_name not in mpjpe_interpolation:
                    mpjpe_interpolation[action_name] = 0
                    cnt_action[action_name] = 0
                
                cnt_action[action_name] = cnt_action[action_name] + 1
                mpjpe_interpolation[action_name] = mpjpe_interpolation[action_name] + 1000*torch.mean(torch.norm(answer[subject][action][frame]['predicted'] - answer[subject][action][frame]['groundtruth'], dim=len(answer[subject][action][frame]['groundtruth'].shape) - 1), dim=len(answer[subject][action][frame]['groundtruth'].shape) - 2)
                #print(mpjpe_interpolation[action_name]/cnt_action[action_name])

    mpjpe_answer = 0
    cnt = 0
    for action in mpjpe_interpolation.keys():
        mpjpe_answer = mpjpe_answer + mpjpe_interpolation[action]
        cnt = cnt + cnt_action[action]
        print(action+":  ",end='')
        print(mpjpe_interpolation[action] / cnt_action[action])
    print(mpjpe_answer/cnt)
    answer = copy.deepcopy(last)
###########################################################################################################################
    cnt = 0
    interpolator=PoseInterpolator()
    mpjpe_interpolation={}
    cnt_action={}
    for subject in tqdm(answer.keys(), desc="Subjects"):
        for action in tqdm(answer[subject].keys(), desc="Actions", leave=False):
            
            maxx = 0
            for frame in sorted(answer[subject][action].keys()):
                if (frame-1)%nviews==0 and (frame-1)!=0:
                    interpolation=interpolator.interpolate(answer[subject][action][frame-nviews]['predicted'],answer[subject][action][frame]['predicted'],nviews+1).clone()
                    for num in range(nviews+1):
                        answer[subject][action][frame-nviews+num]['predicted']=interpolation[num]
                    maxx=frame
            
            for frame in sorted(answer[subject][action].keys()):
                if frame > maxx:
                    break
                end_index = action.find(' ')
                if end_index != -1:
                    action_name = action[:end_index]
                else:
                    action_name = action
                    
                if action_name not in mpjpe_interpolation:
                    mpjpe_interpolation[action_name] = 0
                    cnt_action[action_name] = 0
                    
                cnt_action[action_name] = cnt_action[action_name] + 1
                mpjpe_interpolation[action_name] = mpjpe_interpolation[action_name] + 1000*torch.mean(torch.norm(answer[subject][action][frame]['predicted'] - answer[subject][action][frame]['groundtruth'], dim=len(answer[subject][action][frame]['groundtruth'].shape) - 1), dim=len(answer[subject][action][frame]['groundtruth'].shape) - 2)
    mpjpe_answer = 0
    cnt = 0
    for action in mpjpe_interpolation.keys():
        mpjpe_answer = mpjpe_answer + mpjpe_interpolation[action]
        cnt = cnt + cnt_action[action]
        print(action+":  ",end='')
        print(mpjpe_interpolation[action] / cnt_action[action])
    print(mpjpe_answer/cnt)

if __name__ == '__main__':
    main()

