from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import numpy as np
from lib.models.mixformer_convmae import build_mixformer_convmae_online_score
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box, box_iou

import copy
from lib.utils.kalman_filter import KalmanFilter

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
        self.debug = False
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
        ####### init ####### 
        self.initialize_term_values()
        ###################

        #### kalam filter ####
        self.kalman_filter = KalmanFilter()

        print("Search scale is: ", self.params.search_factor)
        print("Online size is: ", self.online_size)
        print("Update interval is: ", self.update_interval)
        print("Online long term size is: ", self.online_size_lt)
        print("Online short term size is: ", self.online_size_st)
        print("Update long term interval is: ", self.update_interval_lt)
        print("Update short term interval is: ", self.update_interval_st)
        print("Max score decay is ", self.max_score_decay)
    
    def initialize_bbox(self, threshold_iou=0.5):
        self.threshold_iou = threshold_iou

        self.pre_bbox = None

    def initialize_term_values(self):

        # change this values
        self.update_interval_lt = 200
        self.update_interval_st = 5
        #####################

        self.template_lt = None
        self.template_st = None

        self.online_template_lt = None
        self.online_template_st = None

        self.online_max_template_lt = None
        self.online_max_template_st = None

        self.online_size_lt = self.online_sizes[0]
        self.online_size_st = self.online_sizes[0]

        self.online_forget_id_lt = None
        self.online_forget_id_st = None

        self.max_pred_score_lt = None
        self.max_pred_score_st = None


    def initialize(self, image, info: dict, i=0):
        self.initialize_bbox()
        # id init
        self.object_id = i
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        # cv2.imwrite(f'/root/check/0_{i}_init_template.jpg', z_patch_arr)

        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        self.template_lt = template
        self.template_st = template
        self.online_template_lt = template
        self.online_template_st = template
        
        if self.online_size > 1:
            with torch.no_grad():
                self.network.set_online(self.template, self.online_template_lt)

        self.online_state = info['init_bbox']
        
        self.online_image = image
        self.max_pred_score_lt = -1.0
        self.max_pred_score_st = -1.0
        self.online_max_template_lt = template
        self.online_max_template_st = template
        self.online_forget_id_lt = 0
        self.online_forget_id_st = 0
        
        # save states
        self.state = info['init_bbox']
        self.mean, self.covariance = self.kalman_filter.initiate(info['init_bbox'])
        self.pre_xywh = info['init_bbox']
        
        self.pre_bbox = [(self.state[0] - 0.5 * self.state[2]), (self.state[1] - 0.5 * self.state[3]),
                        (self.state[0] + 0.5 * self.state[2]), (self.state[1] + 0.5 * self.state[3])]
        
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def calculate_iou(self, other_bbox):
        
        pred_box = [(self.state[0]), (self.state[1]),
                    (self.state[0] + self.state[2]), (self.state[1] + self.state[3])]
        other_box = [int(other_bbox[0]), int(other_bbox[1]),
                    int(other_bbox[0] + other_bbox[2]), int(other_bbox[1] + other_bbox[3])]
        # print(np.asarray([pred_box]),np.asarray([self.pre_bbox]))
        
        return box_iou(torch.Tensor(np.asarray([other_box])),torch.Tensor(np.asarray([pred_box])))[0].cpu().item()
    
    def IoU(self, box1, box2):
        # box = (x1, y1, x2, y2)
        box1 = [box1[0],box1[1],box1[0] + box1[2],box1[1] + box1[3]]
        box1 = [box2[0],box2[1],box2[0] + box2[2],box2[1] + box2[3]]

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

    def remove_intersection_bbox(self,img, other_bbox):
        if other_bbox[0] == -1:
            return img
        
        pred_box = [int(self.state[0]), int(self.state[1]),
                    int(self.state[0] + self.state[2]), int(self.state[1] + self.state[3])]

        other_box = [int(other_bbox[0]), int(other_bbox[1]),
                    int(other_bbox[0] + other_bbox[2]), int(other_bbox[1] + other_bbox[3])]

        # inter_section_xyxy = self.find_inter_section(torch.Tensor(np.asarray(pred_box)),
        #                                             torch.Tensor(np.asarray(other_box)))

        # if pred_box[0] < inter_section_xyxy[0]:
        #     from_x = pred_box[0]
        #     to_x = inter_section_xyxy[0]
        # else:
        #     from_x = inter_section_xyxy[2]
        #     to_x = pred_box[2]
        
        # if pred_box[1] < inter_section_xyxy[1]:
        #     from_y = pred_box[1]
        #     to_y = inter_section_xyxy[1]
        # else:
        #     from_y = pred_box[3]
        #     to_y = inter_section_xyxy[3]

        remove_img = copy.copy(img)
        # remove_img[pred_box[0]:pred_box[2], pred_box[1]:pred_box[3], :] = 0
        # remove_img[pred_box[1]:pred_box[3],pred_box[0]:pred_box[2], :] = 0
        
        cv2.imwrite(f"/root/vots2023_debug_sam_large/debug_image_{self.object_id}/{self.frame_id}_{self.object_id}_.jpg", remove_img)

        temp = copy.copy(remove_img[pred_box[1]:pred_box[3],pred_box[0]:pred_box[2], :])
        remove_img[other_box[1]:other_box[3],other_box[0]:other_box[2], :] = 0

        cv2.imwrite(f"/root/vots2023_debug_sam_large/debug_image_{self.object_id}/{self.frame_id}_{self.object_id}_pre.jpg", remove_img)
        remove_img[pred_box[1]:pred_box[3],pred_box[0]:pred_box[2], :] = temp
        cv2.imwrite(f"/root/vots2023_debug_sam_large/debug_image_{self.object_id}/{self.frame_id}_{self.object_id}_aft.jpg", remove_img)
        
        return remove_img

    def remove_bbox(self,img, pred_bbox , other_bbox, H, W):
        
        if other_bbox[0] == -1:
            return img
        pred_box = [int(pred_bbox[0]), int(pred_bbox[1]),
                    int(pred_bbox[0] + pred_bbox[2]), int(pred_bbox[1] + pred_bbox[3])]

        other_box = [int(other_bbox[0]), int(other_bbox[1]),
                    int(other_bbox[0] + other_bbox[2]), int(other_bbox[1] + other_bbox[3])]

 
        remove_img = copy.copy(img)

        # cv2.imwrite(f"/root/vots2023_debug_sam_large/debug_image_{self.object_id}/{self.frame_id}_{self.object_id}_.jpg", remove_img)

        from_y = min([max([pred_box[1] - 10,0]),H])
        to_y = min([max([pred_box[3] + 10,0]),H])
        from_x = min([max([pred_box[0] - 10,0]),W])
        to_x = min([max([pred_box[2] + 10,0]),W])
        
        temp = copy.copy(remove_img[from_y:to_y,from_x:to_x, :])
        
        remove_img[other_box[1]:other_box[3],other_box[0]:other_box[2], :] = 0

        # cv2.imwrite(f"/root/vots2023_debug_sam_large/debug_image_{self.object_id}/{self.frame_id}_{self.object_id}_pre.jpg", remove_img)
        remove_img[from_y:to_y,from_x:to_x, :] = temp
        # cv2.imwrite(f"/root/vots2023_debug_sam_large/debug_image_{self.object_id}/{self.frame_id}_{self.object_id}_aft.jpg", remove_img)
        
        return remove_img
    
    
    def find_inter_section(self, box1, box2):
        
        lt = torch.max(box1[:2], box2[:2])
        rb = torch.min(box1[2:], box2[2:])

        return [int(lt[0].cpu().item()),int(lt[0].cpu().item()),int(rb[0].cpu().item()),int(rb[1].cpu().item())]
    
    def hold_position(self, xywh, W, H):
        # cx = (xywh[0] + xywh[2]) / 2
        # cy = (xywh[1] + xywh[3]) / 2
        # w = xywh[2]
        # h = xywh[3]

        # output_xywh = [0,0,0,0]
        # if cx < (W / 2) and cy < (H / 2):
        #     cx += 1; cy += 1; w += 1; h += 1
        #     lt_x = cx - 0.5 * w;  lt_y = cy - 0.5 * h
        #     output_xywh = [lt_x,lt_y,w,h]
        # elif cx < (W / 2) and cy >= (H / 2):
        #     cx += 1; cy -= 1; w += 1; h += 1
        #     lt_x = cx - 0.5 * w;  lt_y = cy - 0.5 * h
        #     output_xywh = [lt_x,lt_y,w,h]
        # elif cx >= (W / 2) and cy < (H / 2):
        #     cx -= 1; cy += 1; w += 1; h += 1
        #     lt_x = cx - 0.5 * w;  lt_y = cy - 0.5 * h
        #     output_xywh = [lt_x,lt_y,w,h]
        # else:
        #     cx -= 1; cy -= 1; w += 1; h += 1
        #     lt_x = cx - 0.5 * w;  lt_y = cy - 0.5 * h
        #     output_xywh = [lt_x,lt_y,w,h]

        output_xywh = np.array(xywh) + np.array([-1,-1,2,2])
        # output_xywh = np.array(output_xywh)

        return output_xywh
    
    def move_position(self, xywh, W, H):
        output_xywh = np.array(xywh) + np.array([-1,-1,2,2])
        return output_xywh
    
    def debug_remove_bbox(self, img):
        img_b = copy.copy(img)
        cv2.imwrite(f"/root/vots2023_debug_sam_large/debug_image_{self.object_id}/{self.frame_id}_{self.object_id}_remove.jpg", img_b)

    def debug_state(self, img, b):
        img_b = copy.copy(img)
        
        cv2.rectangle(img_b, 
                    (int(b[0]), int(b[1])), 
                    (int(b[0] + b[2]), int(b[1] + b[3])),
                    (0, 0, 255), 2)
        return img_b

    def track(self, image, i=0, other_obj_bboxes=None, escape=False, info: dict = None):
    # def track(self, image, i=0, info: dict = None):
        other_bboxes = other_obj_bboxes
        H, W, _ = image.shape
        self.frame_id += 1
        
        #### delete other bbox
        remove_image = copy.copy(image)
        img = copy.copy(image)
        for o_id in other_bboxes.keys():
            # cur object
            if o_id == self.object_id:
                continue
            # other object
            else:
                remove_image = self.remove_bbox(remove_image, pred_bbox=self.state ,other_bbox=other_bboxes[o_id],H=H, W=W)        
                img = self.debug_state(img, other_bboxes[o_id])
        # cv2.imwrite(f"/root/vots2023_debug_sam_large/debug_image_{self.object_id}/{self.frame_id}_{self.object_id}.jpg", img)

        # self.debug_remove_bbox(remove_image)
    
        ######################
        # x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
        #                                                         output_sz=self.params.search_size)  # (x1, y1, w, h)
        x_patch_arr, resize_factor, x_amask_arr = sample_target(remove_image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)                                        
        search = self.preprocessor.process(x_patch_arr)
        with torch.no_grad():
            if self.online_size==1:
                out_dict, _ = self.network(self.template, self.online_template, search, run_score_head=True)
            else:
                out_dict, _ = self.network.forward_test(search, run_score_head=True)
        # cv2.imwrite(f'/root/check/{self.frame_id}_{i}_just_s.jpg', x_patch_arr)
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        pred_score = out_dict['pred_scores'].view(1).sigmoid().item()
        # add conf
        # conf_score = out_dict['conf_score'].view(1).sigmoid().item()
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        
    
        
        # get the final box result
        ####### pred_score ##################
        if pred_score < 0.1:
            # self.state = np.array(self.state) + np.array([-1,-1,2,2])
            self.state = self.hold_position(copy.copy(self.state), H=H, W=W)
            self.state = clip_box(self.state.tolist(), H, W, margin=5)
            escape = True
        elif pred_score > 0.9:
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
            escape = False
        
        else:
            if escape == False:
                self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
            else:
                # self.state = np.array(self.state) + np.array([-1,-1,2,2])
                self.state = self.hold_position(copy.copy(self.state), H=H, W=W)
                ###
                # mean, covariance = self.kalman_filter.predict(self.mean, self.covariance)
                # self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, mean[:4])
                # self.state = self.mean[:4]
                ###
                self.state = clip_box(self.state.tolist(), H, W, margin=5)
        ####################################
        self.max_pred_score_lt = self.max_pred_score_lt * self.max_score_decay
        self.max_pred_score_st = self.max_pred_score_st * self.max_score_decay
        
        # False positive delete & re-inference
        # if escape == False:        
        #     if pred_score < 0.9:
        #         if self.IoU(self.state, other_bboxes[self.object_id]) < 0.5:
        #             remove_image = self.remove_bbox(remove_image, pred_bbox=other_bboxes[self.object_id] ,other_bbox=self.state,H=H, W=W)
        #             # search_region = np.array(self.state) + np.array([-100,-100,100,100])
        #             # search_region = clip_box(search_region, H, W, margin=5)

        #             x_patch_arr, resize_factor, x_amask_arr = sample_target(remove_image, self.state, self.params.search_factor,
        #                                                             output_sz=self.params.search_size)  # (x1, y1, w, h)                                        
        #             search = self.preprocessor.process(x_patch_arr)
        #             with torch.no_grad():
        #                 if self.online_size==1:
        #                     out_dict, _ = self.network(self.template, self.online_template, search, run_score_head=True)
        #                 else:
        #                     out_dict, _ = self.network.forward_test(search, run_score_head=True)
        #             # cv2.imwrite(f'/root/check/{self.frame_id}_{i}_just_s.jpg', x_patch_arr)
        #             pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        #             pred_score = out_dict['pred_scores'].view(1).sigmoid().item()
        #             # add conf
        #             # conf_score = out_dict['conf_score'].view(1).sigmoid().item()
        #             # Baseline: Take the mean of all pred boxes as the final result
        #             pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        #             if pred_score > 0.95:
        #                 self.state = self.hold_position(copy.copy(self.state), H=H, W=W)
        #                 self.state = clip_box(pred_box, H, W, margin=5)
        #             # cv2.imwrite(f"/root/vots2023_debug_sam_large/false_positive/{self.frame_id}_{self.object_id}.jpg", remove_image)


        # update template
        # update long term template
        if pred_score > 0.5 and pred_score > self.max_pred_score_lt:
            z_patch_arr, _, z_amask_arr = sample_target(image, self.state,
                                                        self.params.template_factor,
                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
            self.online_max_template_lt = self.preprocessor.process(z_patch_arr)
            self.max_pred_score_lt = pred_score
        # update short term template
        if pred_score > 0.5 and pred_score > self.max_pred_score_st:
            z_patch_arr, _, z_amask_arr = sample_target(image, self.state,
                                                        self.params.template_factor,
                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
            self.online_max_template_st = self.preprocessor.process(z_patch_arr)
            self.max_pred_score_st = pred_score
        # cv2.imwrite(f'/root/check/{self.frame_id}_{i}_s.jpg', x_patch_arr)
            # cv2.imwrite(f'/root/check/{self.frame_id}_{i}_t.jpg', z_patch_arr)
            
        
        if self.frame_id % self.update_interval_lt == 0:
            if self.online_size_lt == 1:
                self.online_template_lt = self.online_max_template_lt
            elif self.online_template_lt.shape[0] < self.online_size_lt:
                self.online_template_lt = torch.cat([self.online_template_lt, self.online_max_template_lt])
            else:
                self.online_template_lt[self.online_forget_id_lt:self.online_forget_id_lt+1] = self.online_max_template_lt
                self.online_forget_id_lt = (self.online_forget_id_lt + 1) % self.online_size_lt

            # if self.online_size_lt > 1:
            #     with torch.no_grad():
            #         self.network.set_online(self.template, self.online_template_lt)
                    
            self.max_pred_score_lt = -1
            # self.online_max_template_lt = self.template

        # short term
        if self.frame_id % self.update_interval_st == 0:
            if self.online_size_st == 1:
                self.online_template_st = self.online_max_template_st
            elif self.online_template_st.shape[0] < self.online_size_st:
                self.online_template_st = torch.cat([self.online_template_st, self.online_max_template_st])
            else:
                self.online_template_st[self.online_forget_id_st:self.online_forget_id_st+1] = self.online_max_template_st
                self.online_forget_id_st = (self.online_forget_id_st + 1) % self.online_size_st

            if self.online_size_st > 1:
                with torch.no_grad():
                    self.network.set_online(self.template, self.online_template_st)
                    self.network.set_online(self.template, self.online_template_lt)
            self.max_pred_score_st = -1
            self.online_max_template_st = self.template
        

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
                    "pred_score": pred_score,
                    "escape": escape
                    # "conf_score": conf_score
                    }
        else:
            return {"target_bbox": self.state, "pred_score": pred_score, "escape": escape}
            # return {"target_bbox": self.state, "pred_score": pred_score}
            # return {"target_bbox": self.state, "pred_score": pred_score, "conf_score": conf_score}

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
