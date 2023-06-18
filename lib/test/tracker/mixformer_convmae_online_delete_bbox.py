from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
from lib.models.mixformer_convmae import build_mixformer_convmae_online_score
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box, box_iou

import numpy as np
import copy

class MixFormerOnline(BaseTracker):
    def __init__(self, params, dataset_name):
        super(MixFormerOnline, self).__init__(params)
        network = build_mixformer_convmae_online_score(params.cfg,  train=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        print(f"Load checkpoint {self.params.checkpoint} successfully!")
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.attn_weights = []

        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.debug = True
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
            self.online_sizes = self.cfg.TEST.ONLINE_SIZES[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
            self.online_size = 3
        self.update_interval = self.update_intervals[0]
        self.online_size = self.online_sizes[0]
        if hasattr(params, 'online_sizes'):
            self.online_size = params.online_sizes
        if hasattr(params, 'update_interval'):
            self.update_interval = params.update_interval
        if hasattr(params, 'max_score_decay'):
            self.max_score_decay = params.max_score_decay
        else:
            self.max_score_decay = 1.0
        if not hasattr(params, 'vis_attn'):
            self.params.vis_attn = 0
        print("Search scale is: ", self.params.search_factor)
        print("Online size is: ", self.online_size)
        print("Update interval is: ", self.update_interval)
        print("Max score decay is ", self.max_score_decay)

    def initialize_bbox(self, threshold_iou=0.5):
        self.threshold_iou = threshold_iou

        self.pre_bbox = None
    
            
    def initialize(self, image, info: dict):
        self.initialize_bbox()
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr)
        
        self.template = template
        self.online_template = template
        if self.online_size > 1:
            with torch.no_grad():
                self.network.set_online(self.template, self.online_template)

        self.online_state = info['init_bbox']
        
        self.online_image = image
        self.max_pred_score = -1.0
        self.online_max_template = template
        self.online_forget_id = 0
        
        # save states
        
        self.state = info['init_bbox']
        self.pre_xywh = info['init_bbox']
        
        self.pre_bbox = [(self.state[0] - 0.5 * self.state[2]), (self.state[1] - 0.5 * self.state[3]),
                        (self.state[0] + 0.5 * self.state[2]), (self.state[1] + 0.5 * self.state[3])]
        
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def calculate_iou(self):
        
        pred_box = [(self.state[0] - 0.5 * self.state[2]), (self.state[1] - 0.5 * self.state[3]),
                    (self.state[0] + 0.5 * self.state[2]), (self.state[1] + 0.5 * self.state[3])]
        
        # print(np.asarray([pred_box]),np.asarray([self.pre_bbox]))
        
        return box_iou(torch.Tensor(np.asarray([pred_box])),torch.Tensor(np.asarray([self.pre_bbox])))[0].cpu().item()

    def remove_bbox(self,img):
        
        
        pred_box = [int(self.state[0] - 0.5 * self.state[2]), int(self.state[1] - 0.5 * self.state[3]),
                    int(self.state[0] + 0.5 * self.state[2]), int(self.state[1] + 0.5 * self.state[3])]

        inter_section_xyxy = self.find_inter_section(torch.Tensor(np.asarray(pred_box)),
                                                    torch.Tensor(np.asarray(self.pre_bbox)))


        if pred_box[0] < inter_section_xyxy[0]:
            from_x = pred_box[0]
            to_x = inter_section_xyxy[0]
        else:
            from_x = inter_section_xyxy[2]
            to_x = pred_box[2]
        
        if pred_box[1] < inter_section_xyxy[1]:
            from_y = pred_box[1]
            to_y = inter_section_xyxy[1]
        else:
            from_y = pred_box[3]
            to_y = inter_section_xyxy[3]

        remove_img = copy.copy(img)
        # remove_img[pred_box[0]:pred_box[2], pred_box[1]:pred_box[3], :] = 0
        # remove_img[pred_box[1]:pred_box[3],pred_box[0]:pred_box[2], :] = 0
        remove_img[from_y:to_y,from_x:to_x, :] = 0
        # # debug
        cv2.imwrite(f'/root/vots2023_debug/remove_test/{self.frame_id}.jpg',remove_img)
        
        # 123 / 0
        return remove_img

    def find_inter_section(self, box1, box2):
        
        lt = torch.max(box1[:2], box2[:2])
        rb = torch.min(box1[2:], box2[2:])

        return [int(lt[0].cpu().item()),int(lt[0].cpu().item()),int(rb[0].cpu().item()),int(rb[1].cpu().item())]
        


    def update_pre_bbox(self):
        self.pre_bbox = [int(self.state[0] - 0.5 * self.state[2]), int(self.state[1] - 0.5 * self.state[3]),
                        int(self.state[0] + 0.5 * self.state[2]), int(self.state[1] + 0.5 * self.state[3])]
        self.pre_xywh = copy.copy(self.state)

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        # print(type(image), image.shape) <class 'numpy.ndarray'> (720, 1280, 3)
        # 123/0
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        with torch.no_grad():
            if self.online_size==1:
                out_dict, _ = self.network(self.template, self.online_template, search, run_score_head=True)
            else:
                out_dict, _ = self.network.forward_test(search, run_score_head=True)

        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        pred_score = out_dict['pred_scores'].view(1).sigmoid().item()
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
        iou = self.calculate_iou()

        if iou < self.threshold_iou:
            remove_image = self.remove_bbox(image)
            x_patch_arr, resize_factor, x_amask_arr = sample_target(remove_image, self.pre_xywh, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
            search = self.preprocessor.process(x_patch_arr)
            with torch.no_grad():
                if self.online_size==1:
                    out_dict, _ = self.network(self.template, self.online_template, search, run_score_head=True)
                else:
                    out_dict, _ = self.network.forward_test(search, run_score_head=True)

            pred_boxes = out_dict['pred_boxes'].view(-1, 4)
            pred_score = out_dict['pred_scores'].view(1).sigmoid().item()
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
                        

        if pred_score > 0,9:
            self.update_pre_bbox()
        else:
            self.pre_bbox
            self.pre_xywh
        
        self.max_pred_score = self.max_pred_score * self.max_score_decay
        # update template
        if pred_score > 0.5 and pred_score > self.max_pred_score:
            z_patch_arr, _, z_amask_arr = sample_target(image, self.state,
                                                        self.params.template_factor,
                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
            self.online_max_template = self.preprocessor.process(z_patch_arr)
            self.max_pred_score = pred_score
        if self.frame_id % self.update_interval == 0:
            if self.online_size == 1:
                self.online_template = self.online_max_template
            elif self.online_template.shape[0] < self.online_size:
                self.online_template = torch.cat([self.online_template, self.online_max_template])
            else:
                self.online_template[self.online_forget_id:self.online_forget_id+1] = self.online_max_template
                self.online_forget_id = (self.online_forget_id + 1) % self.online_size

            if self.online_size > 1:
                with torch.no_grad():
                    self.network.set_online(self.template, self.online_template)

            self.max_pred_score = -1
            self.online_max_template = self.template

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "pred_score": pred_score}
        else:
            return {"target_bbox": self.state, "pred_score": pred_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return MixFormerOnline
