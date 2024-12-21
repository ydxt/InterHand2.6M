# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, *(['..'] * 3))))

from tqdm import tqdm
import numpy as np
import cv2
from glob import glob
import os.path as osp
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json
from pycocotools.coco import COCO

from main.config import cfg
from common.utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, transform_input_to_output_space, trans_point2d
from common.utils.transforms import world2cam, cam2pixel, pixel2cam

def coco_tmp(annotation_file, mode="train"):
    import time
    print("Load annotation from  " + annotation_file)

    tic = time.time()
    with open(annotation_file, 'r') as f:
        dataset = json.load(f)
    assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
    print('Loaded Done (t={:0.2f}s)'.format(time.time()- tic))
    
    if 'images' in dataset.keys():
        for image_info in tqdm(dataset['images']):
            image_info['file_name'] = os.path.join("InterHand2.6M_5fps",mode, image_info['file_name'])
            image_info['file_name'] = os.path.normpath(image_info['file_name'])
    
    with open(annotation_file.replace(".json", "_new.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False)
    print('Done!!')
    

class InterHand26MDataset:
    def __init__(self, img_path, ann_path, mode):
        self.mode = mode # train, test, val
        self.img_path = img_path
        self.ann_path = ann_path

        self.joint_num = 21 # single hand
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)}
        self.skeleton = load_skeleton(osp.join(self.ann_path, 'skeleton.txt'), self.joint_num*2)
        
        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = []
        
        # load annotation
        print("Load annotation from  " + osp.join(self.ann_path, self.mode))
        db = COCO(osp.join(self.ann_path, self.mode, 'InterHand2.6M_' + self.mode + '_data.json'))
        with open(osp.join(self.ann_path, self.mode, 'InterHand2.6M_' + self.mode + '_camera.json')) as f:
            cameras = json.load(f)
        with open(osp.join(self.ann_path, self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
            joints = json.load(f)
        
        # ignore_seq_names = ['ROM03_LT_No_Occlusion','ROM03_RT_No_Occlusion',
        #                     'ROM04_LT_Occlusion','ROM04_RT_Occlusion',
        #                     'ROM05_LT_Wrist_ROM','ROM05_RT_Wrist_ROM',
        #                     'ROM07_Rt_Finger_Occlusions','ROM08_Lt_Finger_Occlusions']
        ignore_seq_names = []
        for aid in tqdm(db.anns.keys()):
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
 
            capture_id = img['capture']
            seq_name = img['seq_name']
            cam = img['camera']
            frame_idx = img['frame_idx']
            img_path = osp.join(self.img_path, self.mode, img['file_name'])
            
            if seq_name in ignore_seq_names:
                continue            
            
            campos = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32)
            camrot = np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            focal = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32)
            princpt = np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
            joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
            joint_img = cam2pixel(joint_cam, focal, princpt)[:,:2]

            joint_valid = np.array(ann['joint_valid'],dtype=np.float32).reshape(self.joint_num*2)
            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
            joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]
            hand_type = ann['hand_type']
            hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)
            
            img_width, img_height = img['width'], img['height']
            bbox = np.array(ann['bbox'],dtype=np.float32) # x,y,w,h
            bbox = process_bbox(bbox, (img_height, img_width))
            abs_depth = {'right': joint_cam[self.root_joint_idx['right'],2], 'left': joint_cam[self.root_joint_idx['left'],2]}

            cam_param = {'focal': focal, 'princpt': princpt}
            joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
            data = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 
                    'bbox': bbox, 'joint': joint, 'hand_type': hand_type, 
                    'hand_type_valid': hand_type_valid, 'abs_depth': abs_depth, 
                    'file_name': img['file_name'], 'capture': capture_id, 
                    'cam': cam, 'frame': frame_idx, 
                    'image_id': image_id, 'width':img_width, 'height':img_height}
            
            if hand_type == 'right' or hand_type == 'left':
                self.datalist_sh.append(data)
            else:
                self.datalist_ih.append(data)
            if seq_name not in self.sequence_names:
                self.sequence_names.append(seq_name)
            
            vis = False # True
            if vis:
                img_path = data['img_path']
                cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if not isinstance(cvimg, np.ndarray):
                    raise IOError("Fail to read %s" % img_path)
                # _img = cvimg[:,:,::-1].transpose(2,0,1)
                _img = cvimg
                vis_kps = joint_img.copy()
                vis_valid = joint_valid.copy()
                capture = str(data['capture'])
                cam = str(data['cam'])
                frame = str(data['frame'])
                filename = 'out_' + str(image_id) + '_' + hand_type + '.jpg'
                vis_img = self.vis_keypoints(_img, vis_kps, vis_valid, self.skeleton, filename, save_path='./vis')
                _cut_inputs, _cut_targets, _cut_meta_info = self._cut_image(_img, data)
                _cut_img = _cut_inputs['img']
                vis_cut_kps = _cut_targets['joint_coord'].copy()
                vis_cut_valid = _cut_meta_info['joint_valid'].copy()
                cutfilename = 'out_' + str(image_id) + '_' + hand_type + '_cut.jpg'
                vis_cut_img = self.vis_keypoints(_cut_img, vis_cut_kps, vis_cut_valid, self.skeleton, cutfilename, save_path='./vis')

        self.datalist = self.datalist_sh + self.datalist_ih
        print('Number of annotations in single hand sequences: ' + str(len(self.datalist_sh)))
        print('Number of annotations in interacting hand sequences: ' + str(len(self.datalist_ih)))

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)
            
    def handtype_array2str(self, hand_type):
        if (hand_type == np.array([1,0], dtype=np.float32)).all():
            return 'right'
        elif (hand_type == np.array([0,1], dtype=np.float32)).all():
            return 'left'
        elif (hand_type == np.array([1,1], dtype=np.float32)).all():
            return 'interacting'
        else:
            assert 0, print('Not supported hand type: ' + hand_type)
    
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, bbox, joint, hand_type, hand_type_valid = data['img_path'], data['bbox'], data['joint'], data['hand_type'], data['hand_type_valid']
        joint_cam = joint['cam_coord'].copy()
        joint_img = joint['img_coord'].copy()
        joint_valid = joint['valid'].copy()
        hand_type = self.handtype_str2array(hand_type)
        joint_coord = np.concatenate((joint_img, joint_cam[:,2,None]),1)
        
        # image load
        img = load_img(img_path, order='BGR')
        
        # augmentation
        inv_trans = None
        # img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(img, 
        #                                                                    bbox, 
        #                                                                    joint_coord, 
        #                                                                    joint_valid, 
        #                                                                    hand_type, 
        #                                                                    self.mode, 
        #                                                                    self.joint_type)
        rel_root_depth = np.array([joint_coord[self.root_joint_idx['left'],2] - joint_coord[self.root_joint_idx['right'],2]],dtype=np.float32).reshape(1)
        root_valid = np.array([joint_valid[self.root_joint_idx['right']] * joint_valid[self.root_joint_idx['left']]],
                              dtype=np.float32).reshape(1) if hand_type[0]*hand_type[1] == 1 else np.zeros((1),dtype=np.float32)
        # transform to output heatmap space
        # joint_coord, joint_valid, rel_root_depth, root_valid = transform_input_to_output_space(joint_coord, 
        #                                                                                        joint_valid, 
        #                                                                                        rel_root_depth, 
        #                                                                                        root_valid, 
        #                                                                                        self.root_joint_idx, 
        #                                                                                        self.joint_type)
                
        inputs = {'img': img}
        targets = {'joint_coord': joint_coord, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type}
        meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 
                     'hand_type_valid': hand_type_valid, 'inv_trans': inv_trans, 
                     'capture': int(data['capture']), 'cam': int(data['cam']), 
                     'frame': int(data['frame'])}
        return inputs, targets, meta_info
    
    def _cut_image(self, img, datainfo):
        data = datainfo
        img_path, bbox, joint, hand_type, hand_type_valid = data['img_path'], data['bbox'], data['joint'], data['hand_type'], data['hand_type_valid']
        joint_cam = joint['cam_coord'].copy()
        joint_img = joint['img_coord'].copy()
        joint_valid = joint['valid'].copy()
        hand_type = self.handtype_str2array(hand_type)
        joint_coord = np.concatenate((joint_img, joint_cam[:,2,None]),1)

        # augmentation
        img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(img, bbox, joint_coord, joint_valid, hand_type, self.mode, self.joint_type)
        rel_root_depth = np.array([joint_coord[self.root_joint_idx['left'],2] - joint_coord[self.root_joint_idx['right'],2]],
                                  dtype=np.float32).reshape(1)
        root_valid = np.array([joint_valid[self.root_joint_idx['right']] * joint_valid[self.root_joint_idx['left']]],
                              dtype=np.float32).reshape(1) if hand_type[0]*hand_type[1] == 1 else np.zeros((1),dtype=np.float32)
        # transform to output heatmap space
        # joint_coord, joint_valid, rel_root_depth, root_valid = transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, root_valid, self.root_joint_idx, self.joint_type)
                
        inputs = {'img': img}
        targets = {'joint_coord': joint_coord, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type}
        meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 
                     'hand_type_valid': hand_type_valid, 'inv_trans': inv_trans, 
                     'capture': int(data['capture']), 'cam': int(data['cam']), 
                     'frame': int(data['frame'])}
        return inputs, targets, meta_info
    

    # def evaluate(self, preds):

    #     print() 
    #     print('Evaluation start...')

    #     gts = self.datalist
    #     preds_joint_coord, preds_rel_root_depth, preds_hand_type, inv_trans = preds['joint_coord'], preds['rel_root_depth'], preds['hand_type'], preds['inv_trans']
    #     assert len(gts) == len(preds_joint_coord)
    #     sample_num = len(gts)
        
    #     mpjpe_sh = [[] for _ in range(self.joint_num*2)]
    #     mpjpe_ih = [[] for _ in range(self.joint_num*2)]
    #     mrrpe = []
    #     acc_hand_cls = 0; hand_cls_cnt = 0;
    #     for n in range(sample_num):
    #         data = gts[n]
    #         bbox, cam_param, joint, gt_hand_type, hand_type_valid = data['bbox'], data['cam_param'], data['joint'], data['hand_type'], data['hand_type_valid']
    #         focal = cam_param['focal']
    #         princpt = cam_param['princpt']
    #         gt_joint_coord = joint['cam_coord']
    #         joint_valid = joint['valid']
            
    #         # restore xy coordinates to original image space
    #         pred_joint_coord_img = preds_joint_coord[n].copy()
    #         pred_joint_coord_img[:,0] = pred_joint_coord_img[:,0]/cfg.output_hm_shape[2]*cfg.input_img_shape[1]
    #         pred_joint_coord_img[:,1] = pred_joint_coord_img[:,1]/cfg.output_hm_shape[1]*cfg.input_img_shape[0]
    #         for j in range(self.joint_num*2):
    #             pred_joint_coord_img[j,:2] = trans_point2d(pred_joint_coord_img[j,:2],inv_trans[n])
    #         # restore depth to original camera space
    #         pred_joint_coord_img[:,2] = (pred_joint_coord_img[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)
 
    #         # mrrpe
    #         if gt_hand_type == 'interacting' and joint_valid[self.root_joint_idx['left']] and joint_valid[self.root_joint_idx['right']]:
    #             pred_rel_root_depth = (preds_rel_root_depth[n]/cfg.output_root_hm_shape * 2 - 1) * (cfg.bbox_3d_size_root/2)

    #             pred_left_root_img = pred_joint_coord_img[self.root_joint_idx['left']].copy()
    #             pred_left_root_img[2] += data['abs_depth']['right'] + pred_rel_root_depth
    #             pred_left_root_cam = pixel2cam(pred_left_root_img[None,:], focal, princpt)[0]

    #             pred_right_root_img = pred_joint_coord_img[self.root_joint_idx['right']].copy()
    #             pred_right_root_img[2] += data['abs_depth']['right']
    #             pred_right_root_cam = pixel2cam(pred_right_root_img[None,:], focal, princpt)[0]
                
    #             pred_rel_root = pred_left_root_cam - pred_right_root_cam
    #             gt_rel_root = gt_joint_coord[self.root_joint_idx['left']] - gt_joint_coord[self.root_joint_idx['right']]
    #             mrrpe.append(float(np.sqrt(np.sum((pred_rel_root - gt_rel_root)**2))))

           
    #         # add root joint depth
    #         pred_joint_coord_img[self.joint_type['right'],2] += data['abs_depth']['right']
    #         pred_joint_coord_img[self.joint_type['left'],2] += data['abs_depth']['left']

    #         # back project to camera coordinate system
    #         pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)

    #         # root joint alignment
    #         for h in ('right', 'left'):
    #             pred_joint_coord_cam[self.joint_type[h]] = pred_joint_coord_cam[self.joint_type[h]] - pred_joint_coord_cam[self.root_joint_idx[h],None,:]
    #             gt_joint_coord[self.joint_type[h]] = gt_joint_coord[self.joint_type[h]] - gt_joint_coord[self.root_joint_idx[h],None,:]
            
    #         # mpjpe
    #         for j in range(self.joint_num*2):
    #             if joint_valid[j]:
    #                 if gt_hand_type == 'right' or gt_hand_type == 'left':
    #                     mpjpe_sh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
    #                 else:
    #                     mpjpe_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))

    #         # handedness accuray
    #         if hand_type_valid:
    #             if gt_hand_type == 'right' and preds_hand_type[n][0] > 0.5 and preds_hand_type[n][1] < 0.5:
    #                 acc_hand_cls += 1
    #             elif gt_hand_type == 'left' and preds_hand_type[n][0] < 0.5 and preds_hand_type[n][1] > 0.5:
    #                 acc_hand_cls += 1
    #             elif gt_hand_type == 'interacting' and preds_hand_type[n][0] > 0.5 and preds_hand_type[n][1] > 0.5:
    #                 acc_hand_cls += 1
    #             hand_cls_cnt += 1

    #         vis = False
    #         if vis:
    #             img_path = data['img_path']
    #             cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    #             _img = cvimg[:,:,::-1].transpose(2,0,1)
    #             vis_kps = pred_joint_coord_img.copy()
    #             vis_valid = joint_valid.copy()
    #             capture = str(data['capture'])
    #             cam = str(data['cam'])
    #             frame = str(data['frame'])
    #             filename = 'out_' + str(n) + '_' + gt_hand_type + '.jpg'
    #             vis_keypoints(_img, vis_kps, vis_valid, self.skeleton, filename)

    #         vis = False
    #         if vis:
    #             filename = 'out_' + str(n) + '_3d.jpg'
    #             vis_3d_keypoints(pred_joint_coord_cam, joint_valid, self.skeleton, filename)
        

    #     if hand_cls_cnt > 0: print('Handedness accuracy: ' + str(acc_hand_cls / hand_cls_cnt))
    #     if len(mrrpe) > 0: print('MRRPE: ' + str(sum(mrrpe)/len(mrrpe)))
    #     print()
 
    #     tot_err = []
    #     eval_summary = 'MPJPE for each joint: \n'
    #     for j in range(self.joint_num*2):
    #         tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_sh[j]), np.stack(mpjpe_ih[j]))))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
    #         tot_err.append(tot_err_j)
    #     print(eval_summary)
    #     print('MPJPE for all hand sequences: %.2f' % (np.mean(tot_err)))
    #     print()

    #     eval_summary = 'MPJPE for each joint: \n'
    #     for j in range(self.joint_num*2):
    #         mpjpe_sh[j] = np.mean(np.stack(mpjpe_sh[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_sh[j])
    #     print(eval_summary)
    #     print('MPJPE for single hand sequences: %.2f' % (np.mean(mpjpe_sh)))
    #     print()

    #     eval_summary = 'MPJPE for each joint: \n'
    #     for j in range(self.joint_num*2):
    #         mpjpe_ih[j] = np.mean(np.stack(mpjpe_ih[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih[j])
    #     print(eval_summary)
    #     print('MPJPE for interacting hand sequences: %.2f' % (np.mean(mpjpe_ih)))

    def _get_keypoint_rgb(self, skeleton):
        rgb_dict= {}
        for joint_id in range(len(skeleton)):
            joint_name = skeleton[joint_id]['name']

            if joint_name.endswith('thumb_null'):
                rgb_dict[joint_name] = (255, 0, 0)
            elif joint_name.endswith('thumb3'):
                rgb_dict[joint_name] = (255, 51, 51)
            elif joint_name.endswith('thumb2'):
                rgb_dict[joint_name] = (255, 102, 102)
            elif joint_name.endswith('thumb1'):
                rgb_dict[joint_name] = (255, 153, 153)
            elif joint_name.endswith('thumb0'):
                rgb_dict[joint_name] = (255, 204, 204)
            elif joint_name.endswith('index_null'):
                rgb_dict[joint_name] = (0, 255, 0)
            elif joint_name.endswith('index3'):
                rgb_dict[joint_name] = (51, 255, 51)
            elif joint_name.endswith('index2'):
                rgb_dict[joint_name] = (102, 255, 102)
            elif joint_name.endswith('index1'):
                rgb_dict[joint_name] = (153, 255, 153)
            elif joint_name.endswith('middle_null'):
                rgb_dict[joint_name] = (255, 128, 0)
            elif joint_name.endswith('middle3'):
                rgb_dict[joint_name] = (255, 153, 51)
            elif joint_name.endswith('middle2'):
                rgb_dict[joint_name] = (255, 178, 102)
            elif joint_name.endswith('middle1'):
                rgb_dict[joint_name] = (255, 204, 153)
            elif joint_name.endswith('ring_null'):
                rgb_dict[joint_name] = (0, 128, 255)
            elif joint_name.endswith('ring3'):
                rgb_dict[joint_name] = (51, 153, 255)
            elif joint_name.endswith('ring2'):
                rgb_dict[joint_name] = (102, 178, 255)
            elif joint_name.endswith('ring1'):
                rgb_dict[joint_name] = (153, 204, 255)
            elif joint_name.endswith('pinky_null'):
                rgb_dict[joint_name] = (255, 0, 255)
            elif joint_name.endswith('pinky3'):
                rgb_dict[joint_name] = (255, 51, 255)
            elif joint_name.endswith('pinky2'):
                rgb_dict[joint_name] = (255, 102, 255)
            elif joint_name.endswith('pinky1'):
                rgb_dict[joint_name] = (255, 153, 255)
            else:
                rgb_dict[joint_name] = (230, 230, 0)
            
        return rgb_dict

    def vis_keypoints(self, img, kps, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad = 3, save_path=None):
        
        rgb_dict = self._get_keypoint_rgb(skeleton)
        _img = Image.fromarray(img.copy()[:,:,::-1].astype('uint8')) 
        draw = ImageDraw.Draw(_img)
        for i in range(len(skeleton)):
            joint_name = skeleton[i]['name']
            pid = skeleton[i]['parent_id']
            parent_joint_name = skeleton[pid]['name']
            
            kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
            kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

            if score[i] > score_thr and score[pid] > score_thr and pid != -1:
                draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name], width=line_width)
            if score[i] > score_thr:
                draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=rgb_dict[joint_name])
            if score[pid] > score_thr and pid != -1:
                draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=rgb_dict[parent_joint_name])
        
        if save_path is None:
            if not os.path.exists(save_path):
                os.makedirs(cfg.vis_dir)
            _img.save(osp.join(cfg.vis_dir, filename))
        else:
            if not osp.exists(save_path):
                os.makedirs(save_path)
            _img.save(osp.join(save_path, filename))


    def vis_3d_keypoints(self, kps_3d, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad=3, save_path=None):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        rgb_dict = self._get_keypoint_rgb(skeleton)
        
        for i in range(len(skeleton)):
            joint_name = skeleton[i]['name']
            pid = skeleton[i]['parent_id']
            parent_joint_name = skeleton[pid]['name']

            x = np.array([kps_3d[i,0], kps_3d[pid,0]])
            y = np.array([kps_3d[i,1], kps_3d[pid,1]])
            z = np.array([kps_3d[i,2], kps_3d[pid,2]])

            if score[i] > score_thr and score[pid] > score_thr and pid != -1:
                ax.plot(x, z, -y, c = np.array(rgb_dict[parent_joint_name])/255., linewidth = line_width)
            if score[i] > score_thr:
                ax.scatter(kps_3d[i,0], kps_3d[i,2], -kps_3d[i,1], c = np.array(rgb_dict[joint_name]).reshape(1,3)/255., marker='o')
            if score[pid] > score_thr and pid != -1:
                ax.scatter(kps_3d[pid,0], kps_3d[pid,2], -kps_3d[pid,1], c = np.array(rgb_dict[parent_joint_name]).reshape(1,3)/255., marker='o')

        #plt.show()
        #cv2.waitKey(0)
        
        if save_path is None:
            if not os.path.exists(cfg.vis_dir):
                os.makedirs(cfg.vis_dir)
            fig.savefig(osp.join(cfg.vis_dir, filename), dpi=fig.dpi)
        else:
            if not osp.exists(save_path):
                os.makedirs(save_path)
            fig.savefig(osp.join(save_path, filename), dpi=fig.dpi)
            
    def save_raw_and_labelme_annotate(self, img, kps_2d, boxes_2d, kps_3d, score, skeleton, filename, 
                              score_thr=0.4, line_width=3, circle_rad = 3, 
                              save_raw_img_path=None,
                              save_ann_path = None,
                              save_res_img_path=None):
        assert save_raw_img_path is not None, "save_raw_img_path must be set"
        assert save_ann_path is not None, "save_ann_path must be set"
        
        label_names = ['hand']
        hand_keypoints_name = ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
                            "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
                            "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
                            "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
                            "PINKY_FINGER_MCP", "PINKY_FINGER_PIP", "PINKY_FINGER_DIP", "PINKY_FINGER_TIP"]
        hand_skeleton = [[0, 1], [1, 2], [2, 3], [3, 4],
                        [0, 5], [5, 6], [6, 7], [7, 8],
                        [0, 9], [9, 10], [10, 11], [11, 12],
                        [0, 13], [13, 14], [14, 15], [15, 16],
                        [0, 17], [17, 18], [18, 19], [19, 20],
                        [5, 9], [9, 13], [13, 17]]
        # InterHand26M hand pose convert to MediaPipe hand pose index
        # hand_pose_InterHand26M2MediaPipe_IDX = [4, 3, 2, 1,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17,0,
        #                                         4, 3, 2, 1,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17,0]
        hand_pose_InterHand26M2MediaPipe_IDX = [4, 3, 2, 1,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17,0]
        if kps_2d.shape[0] > self.joint_num:
            hand_pose_InterHand26M2MediaPipe_IDX.extend([4, 3, 2, 1,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17,0])
        assert len(hand_pose_InterHand26M2MediaPipe_IDX) == kps_2d.shape[0], "kps_2d length error"
        
        rgb_dict = self._get_keypoint_rgb(skeleton)

        bDraw = False  # show result image
        if save_res_img_path is not None:
            bDraw = True
        img2 = img.copy().astype('uint8')
        if bDraw:
            _img = Image.fromarray(img2[:, :, ::-1]) 
            draw = ImageDraw.Draw(_img)
        
        # `v=0, x=0, y=0`表示该点不可见且未标注，`v=1`表示该点有标注但不可见，`v=2`表示该点有标注且可见，
        # add new shape
        # {
        #     "label": label name,
        #     "points": [
        #         [
        #         0.0,
        #         0.0
        #         ]
        #     ],
        #     "group_id": group id,
        #     "shape_type": "point",
        #     "flags": {},
        #     "visible_id": 0
        # }
        # json_data = {}
        # json_data["version"] = "5.0.1"
        # json_data["flags"] = {}
        # json_data["shapes"] = shapes_new
        # json_data["imagePath"] = os.path.basename(dst_img_full_name)
        # json_data["imageData"] = None
        # json_data["imageHeight"] = box_new_height
        # json_data["imageWidth"] = box_new_width
        box_shapes = []
        hand_id = 0
        if boxes_2d is not None:
            for box_2d in boxes_2d:
                box_shape = {}
                box_shape["label"] = label_names[0] # "hand"
                box_shape["points"] = [
                    [
                        box_2d[0],
                        box_2d[1]
                    ],
                    [
                        box_2d[2],
                        box_2d[1]
                    ],
                    [
                        box_2d[2],
                        box_2d[3]
                    ],
                    [
                        box_2d[0],
                        box_2d[3]
                    ]
                ]
                box_shape["group_id"] = hand_id
                box_shape["shape_type"] = "rectangle"
                box_shape["flags"] = {}
                box_shape["visible_id"] = 2
                box_shapes.append(box_shape)
                hand_id += 1

        point_shapes_tmp = {}
        for i in range(len(skeleton)):
            id_tmp = int(i/self.joint_num)
            if id_tmp not in point_shapes_tmp.keys():
                point_shapes_tmp[id_tmp] = [None for _ in range(self.joint_num)]
            joint_name = skeleton[i]['name']
            pid = skeleton[i]['parent_id']
            parent_joint_name = skeleton[pid]['name']
                        
            kps_i = (kps_2d[i][0].astype(np.int32), kps_2d[i][1].astype(np.int32))
            kps_pid = (kps_2d[pid][0].astype(np.int32), kps_2d[pid][1].astype(np.int32))
            assert i>=0 and i<self.joint_num*2, "kps index error"

            if score[i] > score_thr:
                # v,x,y
                point_shapes_tmp[id_tmp][hand_pose_InterHand26M2MediaPipe_IDX[i]] = [float(2), float(kps_2d[i][0]), float(kps_2d[i][1])]
            else:
                point_shapes_tmp[id_tmp][hand_pose_InterHand26M2MediaPipe_IDX[i]] = [float(0), float(-1),float(-1)]
            
            if bDraw:
                if score[i] > score_thr and score[pid] > score_thr and pid != -1:
                    draw.line([(kps_2d[i][0], kps_2d[i][1]), (kps_2d[pid][0], kps_2d[pid][1])], 
                            fill=rgb_dict[parent_joint_name], width=line_width)
                if score[i] > score_thr:
                    draw.ellipse((kps_2d[i][0]-circle_rad, kps_2d[i][1]-circle_rad, kps_2d[i][0]+circle_rad, kps_2d[i][1]+circle_rad), 
                                fill=rgb_dict[joint_name])
                if score[pid] > score_thr and pid != -1:
                    draw.ellipse((kps_2d[pid][0]-circle_rad, kps_2d[pid][1]-circle_rad, kps_2d[pid][0]+circle_rad, kps_2d[pid][1]+circle_rad),
                                fill=rgb_dict[parent_joint_name])
        
        point_2_box_id_map = {}
        for box_shape in box_shapes:
            assert "group_id" in box_shape.keys(), "box_shape must have visible_id"
            box_id = box_shape["group_id"]
            if box_id is None:
                continue
            assert box_shape["shape_type"].lower() == "rectangle", "shape_type must be rectangle"
            box = box_shape["points"]
            hand_id_count_map = {}
            for point_shape_idx, vis_positions in point_shapes_tmp.items():
                if point_shape_idx not in hand_id_count_map.keys():
                    hand_id_count_map[point_shape_idx] = 0
                
                for i in range(len(vis_positions)):
                    if vis_positions[i] is None:
                        continue
                    if vis_positions[i][0] > 0 and vis_positions[i][1] >= box[0][0] and \
                        vis_positions[i][1] <= box[2][0] and vis_positions[i][2] >= box[0][1] and \
                        vis_positions[i][2] <= box[2][1]:
                        hand_id_count_map[point_shape_idx] += 1
            
            point_shape_idx = -1
            pos_count_max = -1
            for shape_idx, pos_count in hand_id_count_map.items():
                if pos_count > pos_count_max:
                    pos_count_max = pos_count
                    point_shape_idx = shape_idx
            
            if pos_count_max>0 and point_shape_idx not in point_2_box_id_map.keys():
                point_2_box_id_map[point_shape_idx] = box_id
            
        point_shapes = []
        tmp_group_id = 0
        for point_shape_idx, vis_positions in point_shapes_tmp.items():
            assert len(vis_positions)==self.joint_num, "vis_positions length error"
            for v_idx in range(len(vis_positions)):
                point_shape = {}
                point_shape["label"] = hand_keypoints_name[v_idx]
                point_shape["points"] = [
                    [
                        vis_positions[v_idx][1],vis_positions[v_idx][2],
                    ]
                ]
                if point_shape_idx in point_2_box_id_map.keys():
                    point_shape["group_id"] = point_2_box_id_map[point_shape_idx]
                else:
                    while tmp_group_id in point_2_box_id_map.values():
                        tmp_group_id += 1
                    point_2_box_id_map[point_shape_idx]=tmp_group_id
                    point_shape["group_id"] = tmp_group_id
                point_shape["shape_type"] = "point"
                point_shape["flags"] = {}
                point_shape["visible_id"] = vis_positions[v_idx][0]
                point_shapes.append(point_shape)

        shapes_new = []
        if len(box_shapes)>0:
            shapes_new.extend(box_shapes)
        if len(point_shapes)>0:
            shapes_new.extend(point_shapes)
        
        box_new_height, box_new_width = img.shape[:2]
        json_data = {}
        json_data["version"] = "5.0.1"
        json_data["flags"] = {}
        json_data["shapes"] = shapes_new
        json_data["imagePath"] = filename # os.path.basename(dst_img_full_name)
        json_data["imageData"] = None
        json_data["imageHeight"] = float(box_new_height)
        json_data["imageWidth"] = float(box_new_width)
        
        if bDraw:
            if save_res_img_path is None:
                if not os.path.exists(save_res_img_path):
                    os.makedirs(cfg.vis_dir)
                _img.save(osp.join(cfg.vis_dir, filename))
            else:
                if not osp.exists(save_res_img_path):
                    os.makedirs(save_res_img_path)
                _img.save(osp.join(save_res_img_path, filename))
        
        # save raw img
        if save_raw_img_path is not None:
            if not osp.exists(save_raw_img_path):
                os.makedirs(save_raw_img_path)
            cv2.imwrite(osp.join(save_raw_img_path, filename), img2)
            
        # save json file
        json_file_name = filename.replace('.jpg', '.json')
        if save_ann_path is not None:
            if not osp.exists(save_ann_path):
                os.makedirs(save_ann_path)
            with open(osp.join(save_ann_path, json_file_name), 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=1)


    def to_coco_format(self, img_id_start=0, ann_id_start=0, ann_path=None, img_path=None):
        assert ann_path is not None, "ann_path must be specified"
        
        # `v=0, x=0, y=0`表示该点不可见且未标注，`v=1`表示该点有标注但不可见，`v=2`表示该点有标注且可见，
        label_names = {'hand':1}
        hand_keypoints_name = ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
                            "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
                            "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
                            "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
                            "PINKY_FINGER_MCP", "PINKY_FINGER_PIP", "PINKY_FINGER_DIP", "PINKY_FINGER_TIP"]
        hand_skeleton = [[0, 1], [1, 2], [2, 3], [3, 4],
                        [0, 5], [5, 6], [6, 7], [7, 8],
                        [0, 9], [9, 10], [10, 11], [11, 12],
                        [0, 13], [13, 14], [14, 15], [15, 16],
                        [0, 17], [17, 18], [18, 19], [19, 20],
                        [5, 9], [9, 13], [13, 17]]
        # InterHand26M hand pose convert to MediaPipe hand pose index
        # hand_pose_InterHand26M2MediaPipe_IDX = [4, 3, 2, 1,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17,0,
        #                                         4, 3, 2, 1,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17,0]
        hand_pose_InterHand26M2MediaPipe_IDX = [4, 3, 2, 1,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17,0,
                                                4, 3, 2, 1,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17,0]
        score_thr = 0.4
        img_id = img_id_start
        ann_id = ann_id_start
        
        info = {
            "description": "InterHand2.6M",
            "url": "https://mks0601.github.io/InterHand2.6M/",
            "version": "1.0",
            "year": 2021,
            "contributor": "facebookresearch",
            "date_created": "2021/03/22"}
                
        def _init_categories(supercategory_to_id_map=label_names, category_pose_labels=hand_keypoints_name, coco_skeleton=hand_skeleton):
            """
            初始化 COCO 的 标注类别

            例如：
            "categories": [
                {
                    "supercategory": "hand",
                    "id": 1,
                    "name": "hand",
                    "keypoints": [
                        "wrist",
                        "thumb_cmc",
                        "thumb_mcp",
                        ...,
                    ],
                    "skeleton": [
                    ]
                }
            ]
            """
            categories = []
            assert len(supercategory_to_id_map)==1
            for name, id in supercategory_to_id_map.items():
                category = {}

                category['supercategory'] = name
                category['id'] = id
                category['name'] = name
                category['keypoint'] = []
                # 关键点label数据
                for pose in category_pose_labels:
                    category['keypoint'].append(pose.lower())
                
                category['skeleton'] = coco_skeleton
                categories.append(category)
            
            return categories
        
        def _get_box(points):
            min_x = min_y = np.inf
            max_x = max_y = 0
            valid_num = 0
            for x, y, v in points:
                if v<=0:
                    continue
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                valid_num += 1
            return [min_x, min_y, max_x - min_x, max_y - min_y], valid_num
        
        def _image(obj, img_id, image_name):
            """
            解析 生成 coco 的 image 对象

            生成包括：id，file_name，height，width 4个属性

            示例：
                {
                    "file_name": "training/rgb/00031426.jpg",
                    "height": 224,
                    "width": 224,
                    "id": 31426
                }

            """

            image = {}
            
            image['height'] = obj['height']
            image['width'] = obj['width']
            if 'image_id' in obj.keys():
                img_id = obj['image_id']
                image['id'] = img_id + img_id_start
            else:
                img_id = img_id + 1
                image['id'] = img_id

            image['file_name'] = image_name # os.path.basename(path).replace(".json", ".jpg")

            return image, img_id
        
        categories = _init_categories(supercategory_to_id_map=label_names, 
                                      category_pose_labels=hand_keypoints_name, 
                                      coco_skeleton=hand_skeleton)
        # "info" / "images" / "annotations" / "categories" / ~~"licenses"~~
        # "annotations"=[]
        # {"segmentation":[],
        # "num_keypoints": 8,
        # "area": 28292.08625,
        # "iscrowd": 0,
        # "keypoints":[],
        # "image_id": 537548,
        # "bbox":[],
        # "category_id": 1,
        # "id": 183020}
        max_img_id = 0
        images = [] 
        annotations = []
        for index, data in tqdm(enumerate(self.datalist)):
            img_path, bbox, joint, hand_type, hand_type_valid = data['img_path'], data['bbox'], data['joint'], data['hand_type'], data['hand_type_valid']
            joint_cam = joint['cam_coord'].copy()
            joint_img = joint['img_coord'].copy()
            joint_valid = joint['valid'].copy()
            # hand_type = self.handtype_str2array(hand_type)
            joint_coord = np.concatenate((joint_img, joint_cam[:,2,None]), 1)
            
            image, img_id = _image(data, img_id, data['file_name'])
            max_img_id = max(img_id, max_img_id)
            images.append(image)
            
            point_shapes_tmp = {}
            for i in range(len(self.skeleton)):
                id_tmp = int(i/self.joint_num)
                if id_tmp not in point_shapes_tmp.keys():
                    point_shapes_tmp[id_tmp] = [None for _ in range(self.joint_num)]
                joint_name = self.skeleton[i]['name']
                pid = self.skeleton[i]['parent_id']
                parent_joint_name = self.skeleton[pid]['name']
                            
                kps_i = (joint_coord[i][0].astype(np.int32), joint_coord[i][1].astype(np.int32))
                kps_pid = (joint_coord[pid][0].astype(np.int32), joint_coord[pid][1].astype(np.int32))
                assert i>=0 and i<self.joint_num*2, "kps index error"

                if joint_valid[i] > score_thr:
                    # x,y,v
                    point_shapes_tmp[id_tmp][hand_pose_InterHand26M2MediaPipe_IDX[i]] = [int(joint_coord[i][0]), int(joint_coord[i][1]), int(2)]
                else:
                    point_shapes_tmp[id_tmp][hand_pose_InterHand26M2MediaPipe_IDX[i]] = [int(0),int(0),int(0)]
            
            for key_idx, points in point_shapes_tmp.items():
                # {"segmentation":[],
                # "num_keypoints": 8,
                # "area": 28292.08625,
                # "iscrowd": 0,
                # "keypoints":[],
                # "image_id": 537548,
                # "bbox":[],
                # "category_id": 1,
                # "id": 183020}
                box, num_keypoints = _get_box(points)
                if num_keypoints <= 0:
                    continue
                obj = {}                
                obj['segmentation'] = []
                obj['num_keypoints'] = num_keypoints
                obj['area'] = box[2] * box[3]
                obj['iscrowd'] = 0
                obj['keypoints'] = []
                points_tmp = []
                for i in range(len(points)):
                    x, y, v = points[i]
                    obj['keypoints'].extend([x, y, v])
                    points_tmp.extend([x, y])
                obj['segmentation'] = [np.asarray(points_tmp).flatten().tolist()]
                obj['image_id'] = image['id']
                obj['bbox'] = [box[0], box[1], box[2], box[3]]
                obj['category_id'] = 1
                ann_id = ann_id + 1
                obj['id'] = ann_id
                annotations.append(obj)
                
        coco_json = {'info':info, 'images':images,
                      'annotations':annotations, 'categories':categories, 
                      'licenses':[] }
        
        if ann_path is not None:
            ann_path_tmp = osp.join(ann_path, self.mode)
            if not osp.exists(ann_path_tmp):
                os.makedirs(ann_path_tmp)
            with open(osp.join(ann_path_tmp, "InterHand2.6M_"+self.mode+"_coco.json"), 'w', encoding='utf-8') as f:
                # json.dump(coco_json, f, ensure_ascii=False, indent=1)
                json.dump(coco_json, f, ensure_ascii=False)
        
        print(f"annotation id: {ann_id}, image _id: {max_img_id}")
        return coco_json
    
if __name__ == '__main__':
    img_path = 'G:/Datasets/InterHand2_dot_6M/InterHand2.6M_5fps_batch1/images'
    ann_path = 'G:/Datasets/InterHand2_dot_6M/InterHand2.6M_5fps_batch1/annotations'
    mode = 'val'
    # 'train', 'val', 'test'
    dataset = InterHand26MDataset(img_path, ann_path, mode=mode)
    print(len(dataset))
    dataset.to_coco_format(img_id_start=0, ann_id_start=0, ann_path=ann_path, img_path=None)
    print("convert coco format done!")
    # raise Exception("success stop")
    
    print("正在修改'file_name'...")
    # 修改后的'file_name'为`InterHand2.6M_5fps/{mode}/{file_name}`
    # 修改coco格式的标注文件中image中的路径'file_name'
    coco_tmp(osp.join(ann_path, mode, 'InterHand2.6M_' + mode + '_coco.json'), mode=mode)
    raise Exception("success stop")
    
    # TODO： 下面临时不用
    inputs, targets, meta_info = dataset[0]
    print(inputs.keys())
    print(targets.keys())
    print(meta_info.keys())
    
    # inputs = {'img': img}
    # targets = {'joint_coord': joint_coord, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type}
    # meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 
    #              'hand_type_valid': hand_type_valid, 'inv_trans': inv_trans, 
    #              'capture': int(data['capture']), '
    #              cam': int(data['cam']), 'frame': int(data['frame'])}
    
    # img_path = data['img_path']
    cvimg = inputs['img'] # cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    # if not isinstance(cvimg, np.ndarray):
    #     raise IOError("Fail to read %s" % img_path)
    if cvimg is None:
        raise IOError("image is None")
    # _img = cvimg[:,:,::-1].transpose(2,0,1)
    # _img = cvimg[:,:,::-1].transpose(2,0,1)
    _img = cvimg
    
    vis_kps = targets['joint_coord'] # joint_img.copy()
    vis_valid = meta_info['joint_valid'] # joint_valid.copy()
    capture = str(meta_info['capture'])
    cam = str(meta_info['cam'])
    frame = str(meta_info['frame'])
    hand_type = str(dataset.handtype_array2str(targets['hand_type'])) 
    filename = 'out_' + frame + '_' + hand_type + '.jpg'
    vis_img = dataset.vis_keypoints(cvimg, vis_kps, vis_valid, dataset.skeleton, filename, save_path='./vis')
    dataset.save_raw_and_labelme_annotate(img=cvimg, kps_2d=vis_kps, boxes_2d=None, kps_3d=None, score=vis_valid, skeleton=dataset.skeleton, filename=filename, 
                                  save_raw_img_path='./raw_img', save_ann_path='./raw_img', save_res_img_path='./vis')
    
    # _cut_inputs, _cut_targets, _cut_meta_info = dataset._cut_image(_img.transpose(1,2,0), data)
    # _cut_img = _cut_inputs['img'].transpose(2,0,1)
    # vis_cut_kps = _cut_targets['joint_coord'].copy()
    # vis_cut_valid = _cut_meta_info['joint_valid'].copy()
    # cutfilename = 'out_' + frame + '_' + hand_type + '_cut.jpg'
    # vis_cut_img = dataset.vis_keypoints(_cut_img, vis_cut_kps, vis_cut_valid, dataset.skeleton, cutfilename, save_path='./vis')
    
