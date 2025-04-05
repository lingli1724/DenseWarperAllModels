from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import contextlib
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from core.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.vis import save_debug_images, save_debug_fused_images, save_debug_images_2
from utils.pose_utils import align_to_pelvis
from core.interpolation import *
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0  # latest iter's avg
        self.avg = 0  # avg of iter 0 - now
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def collate_first_two_dims(tensor):
    dim0 = tensor.shape[0]
    dim1 = tensor.shape[1]
    left = tensor.shape[2:]
    return tensor.view(dim0 * dim1, *left)


def frozen_backbone_bn(model, backbone_name='resnet'):
    for name, m in model.named_modules():
        if backbone_name in name:
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # logger.info(name)
                m.eval()
            else:
                pass


@contextlib.contextmanager
def dummy_context_mgr():
    yield None


def run_model(
        config,
        dataset,
        loader,
        model,
        criterion_mse,
        criterion_mpjpe,
        final_output_dir,
        tb_writer=None,
        optimizer=None,
        epoch=None,
        is_train=True,
        **kwargs):
    # preparing meters
    mdd = AutoEncoder(3, 17, 2, 4+1)
    print(sum(p.numel() for n,p in mdd.named_parameters()))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    mpjpe_meters = None
    detail_mpjpes = None
    detail_preds = None
    detail_preds2d = None
    detail_weights = None

    nviews = len(dataset.selected_cam)
    nsamples = len(dataset) * nviews
    njoints = config.NETWORK.NUM_JOINTS
    n_used_joints = config.DATASET.NUM_USED_JOINTS
    height = int(config.NETWORK.HEATMAP_SIZE[0])
    width = int(config.NETWORK.HEATMAP_SIZE[1])
    all_view_weights = []
    all_maxvs = []
    all_nview_vis_gt = np.zeros((len(dataset), n_used_joints), dtype=np.int)
    answer={}

    if not is_train:
        do_save_heatmaps = kwargs['save_heatmaps']
        all_preds = np.zeros((nsamples, njoints, 3), dtype=np.float32)
        all_preds_3d = np.zeros((len(dataset), n_used_joints, 3), dtype=np.float32)
        if do_save_heatmaps:
            all_heatmaps = np.zeros((nsamples, njoints, height, width), dtype=np.float32)
        idx_sample = 0

    if is_train:
        phase = 'train'
        model.train()
        frozen_backbone_bn(model, backbone_name='resnet')  # do not change backbone bn params
    else:
        phase = 'test'
        model.eval()
    with dummy_context_mgr() if is_train else torch.no_grad():
        cnttot = {}
        mpj = {}
        # if eval then use no_grad context manager
        end = time.time()
        for i, (input_, target_, weight_, meta_, meta2) in enumerate(loader):
            #print(i)
            data_time.update(time.time() - end)
            debug_bit = False
            batch = input_.shape[0]

            train_2d_backbone = False
            run_view_weight = True

            input = collate_first_two_dims(input_)
            target = collate_first_two_dims(target_)
            weight = collate_first_two_dims(weight_)
            meta = dict()
            for kk in meta_:
                meta[kk] = collate_first_two_dims(meta_[kk])

            extra_params = dict()
            extra_params['run_view_weight'] = run_view_weight
            extra_params['joint_vis'] = weight
            extra_params['run_phase'] = phase

            hms, extra = model(input_, target, **meta_, **extra_params)  # todo
            output = hms
            origin_hms = extra['origin_hms']
            fused_hms_smax = extra['fused_hms_smax']

            target_cuda = target.cuda(2,non_blocking=True)
            weight_cuda = weight.cuda(2,non_blocking=True)
            pose3d_gt = meta_['joints_gt'][:,0,:,:].contiguous().cuda(2,non_blocking=True)  # (batch, njoint, 3)
            num_total_joints = batch * n_used_joints
            # --- --- forward end here

            joint_2d_loss = extra['joint_2d_loss'].mean()

            # obtain all j3d predictions
            final_preds_name = 'j3d_AdaFuse'
            pred3d = extra[final_preds_name]
            j3d_keys = []
            j2d_keys = []
            for k in extra.keys():
                if 'j3d' in k:
                    j3d_keys.append(k)
                if 'j2d' in k:
                    j2d_keys.append(k)
            # initialize only once
            if mpjpe_meters is None:
                logger.info(j3d_keys)
                mpjpe_meters = dict()
                for k in j3d_keys:
                    mpjpe_meters[k] = AverageMeter()
            if detail_mpjpes is None:
                detail_mpjpes = dict()
                for k in j3d_keys:
                    detail_mpjpes[k] = list()
            if detail_preds is None:
                detail_preds = dict()
                for k in j3d_keys:
                    detail_preds[k] = list()
                detail_preds['joints_gt'] = list()
            if detail_preds2d is None:
                detail_preds2d = dict()
                for k in j2d_keys:
                    detail_preds2d[k] = list()
            if detail_weights is None:
                detail_weights = dict()
                detail_weights['maxv'] = list()
                detail_weights['learn'] = list()

            # save all weights
            maxvs = extra['maxv']  # batch njoint, nview
            for b in range(batch):
                maxvs_tmp = []
                for j in range(n_used_joints):
                    maxv_str = ''.join(['{:.2f}, '.format(v) for v in maxvs[b, j]])
                    maxvs_tmp.append(maxv_str)
                all_maxvs.append(maxvs_tmp)
            view_weight = extra['pred_view_weight']
            for b in range(batch):
                maxvs_tmp = []
                for j in range(n_used_joints):
                    maxv_str = ''.join(['{:.2f}, '.format(v) for v in view_weight[b, j]])
                    maxvs_tmp.append(maxv_str)
                all_view_weights.append(maxvs_tmp)

            nviews_vis = extra['nviews_vis']
            all_nview_vis_gt[i*batch:(i+1)*batch] = nviews_vis.view(batch, n_used_joints).detach().cpu().numpy().astype(np.int)

            joints_vis_3d = torch.as_tensor(nviews_vis >= 2, dtype=torch.float32).cuda(2)
            for k in j3d_keys:
                preds = extra[k]
                if config.DATASET.TRAIN_DATASET in ['multiview_h36m','multiview_3dhp']:
                    preds = align_to_pelvis(preds, pose3d_gt, 0)
                
                avg_mpjpe, detail_mpjpe, n_valid_joints, jointsvis= criterion_mpjpe(preds, pose3d_gt, joints_vis_3d=joints_vis_3d, output_batch_mpjpe=True)
                if 'AdaFuse' in k:
                    for o in range(batch):
                        ppreds = preds[o].clone()
                        ppreds = ppreds - ppreds[0]
                        ppreds = ppreds / 1000
                        ppose3d_gt = pose3d_gt[o].clone()
                        ppose3d_gt = ppose3d_gt - ppose3d_gt[0]
                        ppose3d_gt = ppose3d_gt / 1000
                        if meta2[o][0]['subject'] not in answer:
                            answer[meta2[o][0]['subject']]={}
                        if meta2[o][0]['action'] not in answer[meta2[o][0]['subject']]:
                            answer[meta2[o][0]['subject']][meta2[o][0]['action']]={}
                        if meta2[o][0]['frames'] not in answer[meta2[o][0]['subject']][meta2[o][0]['action']]:
                            answer[meta2[o][0]['subject']][meta2[o][0]['action']][meta2[o][0]['frames']]={}
                        answer[meta2[o][0]['subject']][meta2[o][0]['action']][meta2[o][0]['frames']]['predicted']=ppreds
                        answer[meta2[o][0]['subject']][meta2[o][0]['action']][meta2[o][0]['frames']]['groundtruth']=ppose3d_gt
                        """
                        print(preds[o].shape)
                        print(preds[o])
                        print(ppreds.dtype)
                        print(ppose3d_gt.shape)
                        print(pose3d_gt[o])
                        print(ppose3d_gt.dtype)
                        """
                        for x in range(detail_mpjpe.shape[1]):
                            #print(len(meta2[o]))
                            #print(meta2[o][0])
                            action = meta2[o][0]['action']
                            space_index = action.find(' ')
                            if space_index != -1:
                                action = action[:space_index]
                        
                            if action not in mpj:
                                mpj[action] = 0
                                cnttot[action] = 0
                            if jointsvis[o][x]:
                                mpj[action] = mpj[action] + detail_mpjpe[o][x]
                                cnttot[action] = cnttot[action] + 1
                
                mpjpe_meters[k].update(avg_mpjpe, n=n_valid_joints)
                detail_mpjpes[k].extend(detail_mpjpe.detach().cpu().numpy().tolist())
                detail_preds[k].extend(preds.detach().cpu().numpy())
            detail_preds['joints_gt'].extend(pose3d_gt.detach().cpu().numpy())

            for k in j2d_keys:
                p2d = extra[k]
                p2d = p2d.permute(0, 1, 3, 2).contiguous()
                p2d = p2d.detach().cpu().numpy()
                detail_preds2d[k].extend(p2d)

            maxv_weight = extra['maxv'].detach().cpu().numpy()
            detail_weights['maxv'].extend(maxv_weight)
            learn_weight = extra['pred_view_weight'].detach().cpu().numpy()
            detail_weights['learn'].extend(learn_weight)

            if is_train:
                loss = 0
                if train_2d_backbone:
                    loss_mse = criterion_mse(hms, target_cuda, weight_cuda)
                    loss += loss_mse
                loss += joint_2d_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.update(loss.item(), len(input))
            else:
                # validation
                loss = 0
                loss_mse = criterion_mse(hms, target_cuda, weight_cuda)
                loss += loss_mse
                losses.update(loss.item(), len(input))
                nimgs = input.shape[0]

            _, acc, cnt, pre = accuracy(output.detach().cpu().numpy(), target.detach().cpu().numpy(), thr=0.083)
            avg_acc.update(acc, cnt)

            batch_time.update(time.time() - end)
            end = time.time()

            # ---- print logs
            if i % config.PRINT_FREQ == 0 or i == len(loader)-1 or debug_bit:
                gpu_memory_usage = torch.cuda.max_memory_allocated(0)  # bytes
                gpu_memory_usage_gb = gpu_memory_usage / 1.074e9
                mpjpe_log_string = ''
                for k in mpjpe_meters:
                    mpjpe_log_string += '{:.1f}|'.format(mpjpe_meters[k].avg)
                msg = 'Ep:{0}[{1}/{2}]\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'Acc {acc.val:.3f} ({acc.avg:.3f})\t' \
                      'Memory {memory:.2f}G\t' \
                      'MPJPEs {mpjpe_str}'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    speed=input.shape[0] / batch_time.val,
                    data_time=data_time, loss=losses, acc=avg_acc, memory=gpu_memory_usage_gb, mpjpe_str=mpjpe_log_string)
                logger.info(msg)

                # ---- save debug images
                view_name = 'view_{}'.format(0)
                prefix = '{}_{}_{:08}'.format(
                    os.path.join(final_output_dir, phase), view_name, i)
                meta_for_debug_imgs = dict()
                meta_for_debug_imgs['joints_vis'] = meta['joints_vis']
                meta_for_debug_imgs['joints_2d_transformed'] = meta['joints_2d_transformed']
                save_debug_images(config, input, meta_for_debug_imgs, target,
                                  pre * 4, origin_hms, prefix)
                # save_debug_images_2(config, input, meta_for_debug_imgs, target,
                #                   pre * 4, output, prefix, suffix='fuse')
                save_debug_images_2(config, input, meta_for_debug_imgs, target,
                                    pre * 0, fused_hms_smax, prefix, suffix='smax', normalize=True, IMG=False)

            if is_train:
                pass
            else:
                pred, maxval = get_final_preds(config,
                                               output.clone().cpu().numpy(),
                                               meta['center'],
                                               meta['scale'])
                pred = pred[:, :, 0:2]
                pred = np.concatenate((pred, maxval), axis=2)
                all_preds[idx_sample:idx_sample + nimgs] = pred
                all_preds_3d[i * batch:(i + 1) * batch] = pred3d.detach().cpu().numpy()
                if do_save_heatmaps:
                    all_heatmaps[idx_sample:idx_sample + nimgs] = output.cpu().numpy()
                idx_sample += nimgs
        # -- End epoch

        if is_train:
            pass
        else:
            cur_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
            # save mpjpes
            for k in detail_mpjpes:
                detail_mpjpe = detail_mpjpes[k]
                out_path = os.path.join(final_output_dir, '{}_ep_{}_mpjpes_{}.csv'.format(cur_time, epoch, k,))
                np.savetxt(out_path, detail_mpjpe, delimiter=',')
                logger.info('MPJPE summary: {} {:.2f}'.format(k, np.array(detail_mpjpe).mean()))

            # save preds pose detail into h5
            pred_path = os.path.join(final_output_dir, '{}_ep_{}_3dpreds.h5'.format(cur_time, epoch))
            pred_file = h5py.File(pred_path, 'w')
            for k in detail_preds:
                pred_file[k] = np.array(detail_preds[k])
            for k in detail_preds2d:
                pred_file[k] = np.array(detail_preds2d[k])
            for k in detail_weights:
                pred_file[k] = np.array(detail_weights[k])
            pred_file.close()

            if do_save_heatmaps:
                # save heatmaps and joint locations
                u2a = dataset.u2a_mapping
                a2u = {v: k for k, v in u2a.items() if v != '*'}
                a = list(a2u.keys())
                u = np.array(list(a2u.values()))

                save_file = config.TEST.HEATMAP_LOCATION_FILE
                file_name = os.path.join(final_output_dir, save_file)
                file = h5py.File(file_name, 'w')
                file['heatmaps'] = all_heatmaps[:, u, :, :]
                file['locations'] = all_preds[:, u, :]
                file['joint_names_order'] = a
                file.close()
            for act in mpj.keys():
                print(act,end="")
                print(":")
                print(mpj[act]/cnttot[act])

            CUDA_ID = [2, 3]
            device = torch.device("cuda:2")
            model_train_interpolator = AutoEncoder(3, 17, 2, 4+1)
            model_train_interpolator=torch.nn.DataParallel(model_train_interpolator, device_ids=CUDA_ID).to(device)
            model_train_interpolator=model_train_interpolator.to(device)
            #test interpolator2
            pre_dict = torch.load("MCC.pth")
            
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
                    #torch.cuda.empty_cache()
                    maxx = 0
                    for frame in sorted(answer[subject][action].keys()):
                        if frame-4 in answer[subject][action]:
                            source=answer[subject][action][frame-4]['predicted'].clone().unsqueeze(0).to(device)#(1,17,3)
                            target=answer[subject][action][frame]['predicted'].clone().unsqueeze(0).to(device)
                            input__ = torch.stack((source, target), dim=1).permute(0,3,2,1)#(1,2,17,3)
                            interpolation=interpolator(input__).clone()
                            interpolation=interpolation.permute(0,3,2,1)
                            for num in range(4+1):
                                answer[subject][action][frame-4+num]['predicted']=interpolation[0][num]
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
                mpjpe_answer = mpjpe_answer + mpjpe_interpolation[action] / cnt_action[action]
                cnt = cnt + 1
                print(action+":  ",end='')
                print(mpjpe_interpolation[action] / cnt_action[action])
            print(mpjpe_answer/cnt)
            answer = copy.deepcopy(last)

            #test interpolator1
            cnt = 0
            interpolator=PoseInterpolator()
            mpjpe_interpolation={}
            cnt_action={}
            for subject in tqdm(answer.keys(), desc="Subjects"):
                for action in tqdm(answer[subject].keys(), desc="Actions", leave=False):
                    
                    maxx = 0
                    for frame in sorted(answer[subject][action].keys()):
                        if frame-4 in answer[subject][action]:
                            interpolation=interpolator.interpolate(answer[subject][action][frame-4]['predicted'],answer[subject][action][frame]['predicted'],4+1).clone()
                            for num in range(4+1):
                                answer[subject][action][frame-4+num]['predicted']=interpolation[num]
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
                mpjpe_answer = mpjpe_answer + mpjpe_interpolation[action] / cnt_action[action]
                cnt = cnt + 1
                print(action+":  ",end='')
                print(mpjpe_interpolation[action] / cnt_action[action])
            print(mpjpe_answer/cnt)

            return 0
