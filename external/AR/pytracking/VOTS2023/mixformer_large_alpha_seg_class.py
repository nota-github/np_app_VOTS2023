from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import torch
import vot_ as vot
import sys
import time
import os
import numpy as np
sys.path.append('/workspace/wooksu/MixFormer/')
sys.path.append('/workspace/wooksu/MixFormer/external/AR/')
from lib.test.tracker.mixformer_convmae_online import MixFormerOnline
from pytracking.ARcm_seg import ARcm_seg
from pytracking.vot20_utils import *

import lib.test.parameter.mixformer_convmae_online as vot_params


class MIXFORMER_ALPHA_SEG(object):
    # def __init__(self, tracker,  
    #              refine_model_name='ARcm_coco_seg', threshold=0.6):
    def __init__(self, tracker_name, tracker_param, refine_model_name='ARcm_coco_seg', threshold=0.6):
        self.THRES = threshold
        params = vot_params.parameters(tracker_name, model=tracker_param)
        self.tracker = MixFormerOnline(params, "VOT20")
        # self.tracker = tracker
        '''create tracker'''
        '''Alpha-Refine'''
        # project_path = os.path.join(os.path.dirname(__file__), '..', '..')
        # refine_root = os.path.join(project_path, 'ltr/checkpoints/ltr/ARcm_seg/')
        refine_root = os.path.join('/workspace/wooksu/MixFormer/external/AR/ltr/ARcm_seg/')
        refine_path = os.path.join(refine_root, refine_model_name)
        '''2020.4.25 input size: 384x384'''
        self.alpha = ARcm_seg(refine_path, input_sz=384)

    def initialize(self, image, mask):
        region = rect_from_mask(mask)
        # init_info = {'init_bbox': region}
        # self.tracker.initialize(image, init_info)

        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize STARK for specific video'''
        init_info = {'init_bbox': list(gt_bbox_np)}
        self.tracker.initialize(image, init_info)
        '''initilize refinement module for specific video'''
        self.alpha.initialize(image, np.array(gt_bbox_np))

    def track(self, img_RGB):
        '''TRACK'''
        '''base tracker'''
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        '''Step2: Mask report'''
        pred_mask, search, search_mask = self.alpha.get_mask(img_RGB, np.array(pred_bbox), vis=True)
        final_mask = (pred_mask > self.THRES).astype(np.uint8)
        AR_bbox = pred_bbox if np.sum(final_mask) == 0 else rect_from_mask(final_mask)
        return final_mask, pred_bbox, AR_bbox

def main(vis=False):
    color = [(6, 230, 230), (4, 200, 3), (204, 5, 255), 
             (230, 230, 230), (235, 255, 7), 
             (150, 5, 61), (120, 120, 70), (8, 255, 51), (255, 6, 82), 
             (143, 255, 140), (204, 255, 4), (255, 51, 7), (204, 70, 3)]
    save_dir = '/workspace/vots2023_debug_large_1'
    if vis:
        os.makedirs(save_dir, exist_ok=True)
    refine_model_name = 'ARcm_coco_seg_only_mask_384'
    tracker_name = "baseline_large"
    tracker_param = "mixformer_convmae_large_online.pth.tar"
    # tracker_param = "mixformerL_online_22k.pth.tar"
    # params = vot_params.parameters("baseline", model="mixformer_convmae_base_online.pth.tar")
    # mixformer = MixFormerOnline(params, "VOT20")
    
    handle = vot.VOT("mask", multiobject=True)
    objects = handle.objects()
    imagefile = handle.frame()
    if vis:
        seq_name = imagefile.split('/')[-3]
        save_img_dir = os.path.join(save_dir, seq_name)
        # os.makedirs(os.path.join(save_dir, seq_name, 'org'), exist_ok=True)
    if not imagefile:
        sys.exit(0)
    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
    # mask = make_full_size(objects, (image.shape[1], image.shape[0]))

    # tracker = MIXFORMER_ALPHA_SEG(tracker=mixformer, refine_model_name=refine_model_name)
    # tracker.H = image.shape[0]
    # tracker.W = image.shape[1]

    # tracker.initialize(image, mask)
    
    trackers = []
    for object in objects:
        tracker = MIXFORMER_ALPHA_SEG(tracker_name=tracker_name, tracker_param=tracker_param, refine_model_name=refine_model_name)
        mask = make_full_size(object, (image.shape[1], image.shape[0]))
        tracker.initialize(image,mask)
        trackers.append(tracker)
    
    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
        outputs = [t.track(image) for t in trackers]
        masks = list(map(lambda x: x[0], outputs))
        M_bbox = list(map(lambda x: x[1], outputs))
        AR_bbox = list(map(lambda x: x[2], outputs))
        # m, mb, ab = tracker.track(image)
        handle.report(masks)
        if vis:
            img_org = image[:, :, ::-1].copy()
            img_name = imagefile.split('/')[-1]
            save_path = os.path.join(save_dir, seq_name, img_name)
            # cv2.imwrite(save_path, img_org)
            img_m = img_org.copy().astype(np.float32)
            img_b = img_org.copy()

            for i, (mb, ab, m) in enumerate(zip(M_bbox, AR_bbox, masks)):
                cv2.rectangle(img_b, 
                            (int(mb[0]), int(mb[1])), 
                            (int(mb[0] + mb[2]), int(mb[1] + mb[3])),
                            (0, 0, 255), 2)
                cv2.rectangle(img_b, 
                            (int(ab[0]), int(ab[1])), 
                            (int(ab[0] + ab[2]), int(ab[1] + ab[3])),
                            (0, 255, 0), 2)
                m3 = np.stack([m]*3, axis=2)/2
                m3_inv = np.ones(m3.shape) - m3
                img_m *= m3_inv
                # m3 = np.concatenate([m,m,m], axis=2)
                img_m += m3 * color[i]
                # img_m[:, :, 0] += float(color[i][0]) * m
                # img_m[:, :, 1] += float(color[i][1]) * m
                # img_m[:, :, 2] += float(color[i][2]) * m
                contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_m = cv2.drawContours(img_m, contours, -1, color[i], 2)

            img_b_name = img_name.replace('.jpg', '_bbox.jpg')
            save_b_path = os.path.join(save_dir, seq_name, 'bbox',img_b_name)
            img_m = img_m.clip(0, 255).astype(np.uint8)
            image_mask_name_m = img_name.replace('.jpg', '_mask.jpg')
            save_m_path = os.path.join(save_dir, seq_name, 'mask', image_mask_name_m)

            os.makedirs(os.path.join(save_dir, seq_name, 'bbox'), exist_ok=True)
            os.makedirs(os.path.join(save_dir, seq_name, 'mask'), exist_ok=True)

            cv2.imwrite(save_b_path, img_b)
            cv2.imwrite(save_m_path, img_m)

if __name__=='__main__':
    main(vis=True)