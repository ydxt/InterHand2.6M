# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, *(['..'] * 3))))

from tqdm import tqdm
import numpy as np
import cv2
import os.path as osp
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json
from pycocotools.coco import COCO

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
            # 修改'file_name'中文件名
            image_info['file_name'] = os.path.join("InterHand2.6M_5fps",mode, image_info['file_name'])
            image_info['file_name'] = os.path.normpath(image_info['file_name'])
    
    with open(annotation_file.replace(".json", "_new.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False)
    print('Done!!')
    

class Config:
    
    ## dataset
    dataset = 'InterHand2.6M' # InterHand2.6M, RHD, STB

    ## input, output
    input_img_shape = (256, 256)
    output_hm_shape = (64, 64, 64) # (depth, height, width)
    sigma = 2.5
    bbox_3d_size = 400 # depth axis
    bbox_3d_size_root = 400 # depth axis
    output_root_hm_shape = 64 # depth axis

    ## model
    resnet_type = 50 # 18, 34, 50, 101, 152

    ## training config
    lr_dec_epoch = [15, 17] if dataset == 'InterHand2.6M' else [45,47]
    end_epoch = 20 if dataset == 'InterHand2.6M' else 50
    lr = 1e-4
    lr_dec_factor = 10
    train_batch_size = 16

    ## testing config
    test_batch_size = 32
    trans_test = 'rootnet' # gt, rootnet

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')

    ## others
    num_thread = 40
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

cfg = Config()


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord

def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord


def load_img(path, order='RGB'):
    
    # load
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def load_skeleton(path, joint_num):

    # load joint info (name, parent_id)
    skeleton = [{} for _ in range(joint_num)]
    with open(path) as fp:
        for line in fp:
            if line[0] == '#': continue
            splitted = line.split(' ')
            joint_name, joint_id, joint_parent_id = splitted
            joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
            skeleton[joint_id]['name'] = joint_name
            skeleton[joint_id]['parent_id'] = joint_parent_id
    # save child_id
    for i in range(len(skeleton)):
        joint_child_id = []
        for j in range(len(skeleton)):
            if skeleton[j]['parent_id'] == i:
                joint_child_id.append(j)
        skeleton[i]['child_id'] = joint_child_id
    
    return skeleton

def get_aug_config():
    trans_factor = 0.15
    scale_factor = 0.25
    rot_factor = 45
    color_factor = 0.2
    
    trans = [np.random.uniform(-trans_factor, trans_factor), np.random.uniform(-trans_factor, trans_factor)]
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    do_flip = random.random() <= 0.5
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]).astype(np.float32)

    return trans, scale, rot, do_flip, color_scale


def process_bbox(bbox, original_img_shape):

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = cfg.input_img_shape[1]/cfg.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.

    return bbox

def generate_patch_image(cvimg, bbox, do_flip, scale, rot, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1
    
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, trans, inv_trans

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


def augmentation(img, bbox, joint_coord, joint_valid, hand_type, mode, joint_type):
    img = img.copy(); 
    joint_coord = joint_coord.copy(); 
    hand_type = hand_type.copy();

    original_img_shape = img.shape
    joint_num = len(joint_coord)
    
    if mode == 'train':
        trans, scale, rot, do_flip, color_scale = get_aug_config()
    else:
        trans, scale, rot, do_flip, color_scale = [0,0], 1.0, 0.0, False, np.array([1,1,1])
        color_scale=color_scale.astype(np.float32)
    
    bbox[0] = bbox[0] + bbox[2] * trans[0]
    bbox[1] = bbox[1] + bbox[3] * trans[1]
    img, trans, inv_trans = generate_patch_image(img, bbox, do_flip, scale, rot, cfg.input_img_shape)
    img = np.clip(img * color_scale[None,None,:], 0, 255)
    
    if do_flip:
        joint_coord[:,0] = original_img_shape[1] - joint_coord[:,0] - 1
        joint_coord[joint_type['right']], joint_coord[joint_type['left']] = joint_coord[joint_type['left']].copy(), joint_coord[joint_type['right']].copy()
        joint_valid[joint_type['right']], joint_valid[joint_type['left']] = joint_valid[joint_type['left']].copy(), joint_valid[joint_type['right']].copy()
        hand_type[0], hand_type[1] = hand_type[1].copy(), hand_type[0].copy()
        
    for i in range(joint_num):
        joint_coord[i,:2] = trans_point2d(joint_coord[i,:2], trans)
        joint_valid[i] = joint_valid[i] * (joint_coord[i,0] >= 0) * \
            (joint_coord[i,0] < cfg.input_img_shape[1]) * (joint_coord[i,1] >= 0) * \
            (joint_coord[i,1] < cfg.input_img_shape[0])

    return img, joint_coord, joint_valid, hand_type, inv_trans


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
    img_path = './InterHand2_dot_6M/InterHand2.6M_5fps_batch1/images'
    ann_path = './InterHand2_dot_6M/InterHand2.6M_5fps_batch1/annotations'
    # mode = 'train'
    for mode in ['train', 'val', 'test']:
        print(f"processing {mode} to coco format...")
        # 'train', 'val', 'test'
        dataset = InterHand26MDataset(img_path, ann_path, mode=mode)
        print(len(dataset))
        dataset.to_coco_format(img_id_start=0, ann_id_start=0, ann_path=ann_path, img_path=None)
        print("convert coco format done!")
        # raise Exception("success stop")
        
        print("正在修改'file_name'...")
        # 修改coco格式的标注文件中image中的路径'file_name'
        # 修改后的'file_name'为`InterHand2.6M_5fps/{mode}/{file_name}`
        coco_tmp(osp.join(ann_path, mode, 'InterHand2.6M_' + mode + '_coco.json'), mode=mode)
    
    print("done!")
    