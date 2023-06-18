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
sys.path.append('/root/np_app_VOTS2023/')
sys.path.append('/root/np_app_VOTS2023/external/AR/')
from lib.test.tracker.mixformer_convmae_online_12 import MixFormerOnline
from segment_anything import sam_model_registry, SamPredictor

from pytracking.vot20_utils import *
import matplotlib.pyplot as plt
import lib.test.parameter.mixformer_convmae_online as vot_params


class MIXFORMER(object):
    # def __init__(self, tracker,  
    #              refine_model_name='ARcm_coco_seg', threshold=0.6):
    def __init__(self, tracker_name, tracker_param, threshold=0.6):
        self.THRES = threshold
        params = vot_params.parameters(tracker_name, model=tracker_param)
        self.tracker = MixFormerOnline(params, "VOT20")
        '''create tracker'''
        
    def initialize(self, image, mask, i):
        region = rect_from_mask(mask)
        # init_info = {'init_bbox': region}
        # self.tracker.initialize(image, init_info)

        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize STARK for specific video'''
        init_info = {'init_bbox': list(gt_bbox_np)}
        self.tracker.initialize(image, init_info, i)

    def track(self, img_RGB, i, escape):
        '''TRACK'''
        '''base tracker'''
        outputs = self.tracker.track(img_RGB, i, escape)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['pred_score']
        escape = outputs['escape']
        
        if escape == True:
            return None, pred_score, escape
        '''Step2: Mask report'''
        return pred_bbox, pred_score, escape
    
def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_coord = [x1, y1, x2, y2]
    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def main(vis=False):

    '''sam setting'''
    sam_checkpoint = "/root/np_app_VOTS2023/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.cuda()
    segment_predictor = SamPredictor(sam)

    color = [(255, 0, 0),(0, 255, 0),(0, 0, 255),(255, 255, 0),(255, 0, 255),(0, 255, 255),(127, 127, 127),(127, 0, 255)]
    save_dir = '/root/vots2023_debug_sam_large'
    if vis:
        os.makedirs(save_dir, exist_ok=True)
    tracker_name = "baseline_large"
    tracker_param = "mixformer_convmae_large_online.pth.tar"
    
    handle = vot.VOT("mask", multiobject=True)
    objects = handle.objects()
    imagefile = handle.frame()
    if vis:
        seq_name = imagefile.split('/')[-3]
        save_img_dir = os.path.join(save_dir, seq_name)
    if not imagefile:
        sys.exit(0)
    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
    trackers = []
    for i, object in enumerate(objects):
        tracker = MIXFORMER(tracker_name=tracker_name, tracker_param=tracker_param)
        mask = make_full_size(object, (image.shape[1], image.shape[0]))
        tracker.initialize(image, mask, i)
        trackers.append(tracker)
    escapes = [False] * len(objects)
    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
        segment_predictor.set_image(image)

        outputs = [t.track(image, i, escapes[i]) for i, t in enumerate(trackers)]
        bbox = list(map(lambda x: x[0], outputs))
        score = list(map(lambda x: x[1], outputs))
        escapes = list(map(lambda x: x[2], outputs))
        for i in range(len(bbox)):
            for j in range(len(bbox)):
                if i != j:
                    if bbox[i] is None or bbox[j] is None:
                        continue
                    if IoU(bbox[i], bbox[j]) > 0.3:
                        if score[i] > score[j]:
                            bbox[j] = None
                        else:
                            bbox[i] = None
                else:
                    continue
        bbox_input = [[int(box[0]), int(box[1]), 
                            int(box[0] + box[2]), int(box[1] + box[3])] for box in bbox if box is not None]
        # with open('/root/check_large_sam.txt', 'a') as f:
        #     l = ''
        #     for s in score:
        #         l += f'{s}, '
        #     l += '\n'
        #     f.write(l)
        
        boxes = torch.tensor(bbox_input, device=sam.device)
        if len(boxes) > 0:
            transformed_boxes = segment_predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
            masks, _, _ = segment_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            masks = masks[:,0,:,:].cpu().numpy().astype(np.uint8)
            masks_out = []
            c = 0
            for box in bbox:
                if box is not None:
                    masks_out.append(masks[c])
                    c +=1
                else:
                    masks_out.append(np.zeros(image.shape[:2]).astype(np.uint8))
        else:
            masks_out = [np.zeros(image.shape[:2]).astype(np.uint8)] * len(objects)
        handle.report(masks_out)
        
        if vis:
            img_org = image[:, :, ::-1].copy()
            img_name = imagefile.split('/')[-1]
            save_path = os.path.join(save_dir, seq_name, img_name)
            img_m = img_org.copy().astype(np.float32)
            img_b = img_org.copy()

            for i, (b, m) in enumerate(zip(bbox, masks_out)):
                if b is not None:
                    cv2.rectangle(img_b, 
                                (int(b[0]), int(b[1])), 
                                (int(b[0] + b[2]), int(b[1] + b[3])),
                                (0, 0, 255), 2)
                    m3 = np.stack([m]*3, axis=2)/2
                    m3_inv = np.ones(m3.shape) - m3
                    img_m *= m3_inv
                    img_m += m3 * color[i]
                    contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    img_m = cv2.drawContours(img_m, contours, -1, color[i], 2)
                else:
                    continue

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