#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import copy
import math
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmdet.models import build_loss
from einops import rearrange
from mmdet.models.utils.transformer import inverse_sigmoid
from ..dense_heads.track_head_plugin import MemoryBank, QueryInteractionModule, Instances, RuntimeTrackerBase

import sys
import signal
import random
import time
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
import numpy as np
import cv2
import matplotlib.pyplot as plt
import rospy

def quit(signum, frame):
    print('')
    print('stop uniad_e2e images')
    sys.exit()

@DETECTORS.register_module()
class UniADTrack(MVXTwoStageDetector):
    """UniAD tracking part
    """
    def __init__(
        self, 
        use_grid_mask=False,
        img_backbone=None,
        img_neck=None,
        pts_bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        loss_cfg=None,
        qim_args=dict(
            qim_type="QIMBase",
            merger_dropout=0,
            update_query_pos=False,
            fp_ratio=0.3,
            random_drop=0.1,
        ),
        mem_args=dict(
            memory_bank_type="MemoryBank",
            memory_bank_score_thresh=0.0,
            memory_bank_len=4,
        ),
        bbox_coder=dict(
            type="DETRTrack3DCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            num_classes=10,
            score_threshold=0.0,
            with_nms=False,
            iou_thres=0.3,
        ),
        pc_range=None,
        embed_dims=256,
        num_query=900,
        num_classes=10,
        vehicle_id_list=None,
        score_thresh=0.2,
        filter_score_thresh=0.1,
        miss_tolerance=5,
        gt_iou_threshold=0.0,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_bn=False,
        freeze_bev_encoder=False,
        queue_length=3,
    ):
        super(UniADTrack, self).__init__(
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )

        # print(self.img_backbone)
        # print(self.img_neck)
        self.counter = 0
        self.pre_key_feats = []
        self.nusc = NuScenes(version='v1.0-mini', dataroot='/home/mobilitylab/v1.0-mini', verbose=True)
        self.camera_bboxes = {}  # Dictionary to store bounding boxes for each camera
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mask_roi_name = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.kf = 10

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.vehicle_id_list = vehicle_id_list
        self.pc_range = pc_range
        self.queue_length = queue_length
        if freeze_img_backbone:
            if freeze_bn:
                self.img_backbone.eval()
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        
        if freeze_img_neck:
            if freeze_bn:
                self.img_neck.eval()
            for param in self.img_neck.parameters():
                param.requires_grad = False

        # temporal
        self.video_test_mode = video_test_mode
        assert self.video_test_mode

        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.query_embedding = nn.Embedding(self.num_query+1, self.embed_dims * 2)   # the final one is ego query, which constantly models ego-vehicle
        self.reference_points = nn.Linear(self.embed_dims, 3)

        self.mem_bank_len = mem_args["memory_bank_len"]
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance,
        )  # hyper-param for removing inactive queries

        self.query_interact = QueryInteractionModule(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.memory_bank = MemoryBank(
            mem_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )
        self.mem_bank_len = (
            0 if self.memory_bank is None else self.memory_bank.max_his_length
        )
        self.criterion = build_loss(loss_cfg)
        self.test_track_instances = None
        self.l2g_r_mat = None
        self.l2g_t = None
        self.gt_iou_threshold = gt_iou_threshold
        self.bev_h, self.bev_w = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
        self.freeze_bev_encoder = freeze_bev_encoder

        signal.signal(signal.SIGINT, quit)
        signal.signal(signal.SIGTERM, quit)
    
    def add_padding_and_round_up(self, s, data_padding=100, devisible_size=32):
        return ((s + data_padding + devisible_size - 1) // devisible_size) * devisible_size
    
    def images_rois_spli_merge(self, imgs):
        # Define or calculate crop dimensions
        roi_name = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        imgs_feats = self.pre_key_feats
        # file = open('roi_time_records.txt', 'a+')
        cam_num = 0
        for i, img in enumerate(imgs[0]):
            
            t0 = time.time()
            # torch.cuda.synchronize()

            min_x, min_y, max_x, max_y = self.camera_bboxes[roi_name[i]]
            if max_x == 0 or max_y == 0 or max_x-min_x == 0 or max_y-min_y == 0:
                continue

            cam_num += 1

            crop_width = self.add_padding_and_round_up(max_x-min_x)
            crop_height = self.add_padding_and_round_up(max_y-min_y)
            min_x = max(min_x-100, 0)
            min_y = max(min_y-100, 0)
            crop_height = min(crop_height, 928-min_y)
            crop_width = min(crop_width, 1600-min_x)
            # print(min_x, min_y, crop_height, crop_width)
            # Apply cropping using the separate crop_images function
            img_cropped = img[..., min_y:min_y+crop_height, min_x:min_x+crop_width]
            # print(img_cropped.size())
            # img_cropped = self.crop_images(imgs, crop_x, crop_y, crop_width, crop_height)
            # Reshape for processing
            img_cropped = img_cropped.reshape(3, crop_height, crop_width)
            # Create a mini-batch as expected by the model
            input_batch = img_cropped.unsqueeze(0)

            t1 = time.time()
            torch.cuda.synchronize()

            if self.use_grid_mask:
                input_batch = self.grid_mask(input_batch)
            rois_feats = self.img_backbone(input_batch)

            t2 = time.time()
            torch.cuda.synchronize()

            x_p = int(min_x / 8)
            y_p = int(min_y / 8)
            # previous key features padding
            for j, feats in enumerate(rois_feats):
                target_shape = feats.shape[2:]  # Get the actual shape of the output feature map
                y_end = y_p + target_shape[0]
                x_end = x_p + target_shape[1]
                # imgs_feats[j][i][..., y_p:y_p+target_shape[0], x_p:x_p+target_shape[1]] = feats
                imgs_feats[j][i][..., y_p:y_end, x_p:x_end].copy_(feats[0])
                x_p = int(x_p / 2)
                y_p = int(y_p / 2)
            
            t3 = time.time()
            torch.cuda.synchronize()
            print(t1-t0, t2-t1, t3-t2, t3-t0)
            
        #     file.write("%s, %s, %s, %s, %s, %s, %s, %s, %s\n"% (roi_name[i],
        #         min_x, min_y, crop_height, crop_width, t1-t0, t2-t1, t3-t2, t3-t0))
        # file.close()
        
        return imgs_feats, cam_num
    
    def images_rois_spli_merge_unified(self, imgs):
        # Define or calculate crop dimensions
        roi_name = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        # imgs_feats = self.pre_key_feats
        imgs_feats = tuple(feat.clone() for feat in self.pre_key_feats)

        if self.command[0] == 2:  # forward
            self.mask_roi_name = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT']
        elif self.command[0] == 1:  # turn left
            self.mask_roi_name = ['CAM_FRONT', 'CAM_BACK_RIGHT', 'CAM_FRONT_LEFT']
        elif self.command[0] == 0:  # turn right
            self.mask_roi_name = ['CAM_FRONT', 'CAM_BACK_LEFT', 'CAM_FRONT_RIGHT']
        else:
            self.mask_roi_name = roi_name

        # Initialize variables for maximum bbox dimensions and minimum coordinates
        global_min_x = float('inf')
        global_min_y = float('inf')
        max_width = 0
        max_height = 0

        cam_num = 0

        # Calculate the largest valid bbox and the maximum of minimum coordinates across all cameras
        for name in self.mask_roi_name:
            min_x, min_y, max_x, max_y = self.camera_bboxes[name]
            if max_x == 0 or max_y == 0 or max_x - min_x == 0 or max_y - min_y == 0:
                continue
            global_min_x = min(global_min_x, min_x)
            global_min_y = min(global_min_y, min_y)
            width = self.add_padding_and_round_up(max_x - min_x)
            height = self.add_padding_and_round_up(max_y - min_y)
            max_width = max(max_width, width)
            max_height = max(max_height, height)

        # Adjust for padding and bounds
        crop_height = min(max_height, 928 - global_min_y)
        crop_width = min(max_width, 1600 - global_min_x)
        print(crop_height, crop_width)
        print(self.command, self.mask_roi_name)

        valid_crops = []

        # Crop each image according to the global parameters
        for i, name in enumerate(roi_name):
            if name in self.mask_roi_name:
                cam_num += 1
                img = imgs[0][i]
                img_cropped = img[..., global_min_y:global_min_y + crop_height, global_min_x:global_min_x + crop_width]
                
                # Reshape for processing and append to valid_crops list
                img_cropped = img_cropped.reshape(3, crop_height, crop_width)
                valid_crops.append(img_cropped.unsqueeze(0))

        # Concatenate all valid cropped images to form a batch
        if valid_crops:
            input_batch = torch.cat(valid_crops, dim=0)
            if self.use_grid_mask:
                input_batch = self.grid_mask(input_batch)
            rois_feats = self.img_backbone(input_batch)

            # Update the feature maps accordingly
            for i, name in enumerate(self.mask_roi_name):
                x_p = int(global_min_x / 8)
                y_p = int(global_min_y / 8)
                for j, feats in enumerate(rois_feats):
                    target_shape = feats.shape[2:]
                    x_end = x_p + target_shape[1]
                    y_end = y_p + target_shape[0]
                    
                    # Efficiently update imgs_feats with the new features
                    img_idx = roi_name.index(name)
                    imgs_feats[j][img_idx][..., y_p:y_end, x_p:x_end].copy_(feats[i])
                    x_p = int(x_p / 2)
                    y_p = int(y_p / 2)
        
        return imgs_feats, cam_num
    
    def images_rois_spli_merge_static(self, imgs):
        B, N, C, H, W = imgs.size()
        
        crop_x, crop_y, crop_width, crop_height = 0, 450, 1600, 450  # Example crop dimensions
        # Apply cropping using the separate crop_images function
        img_cropped = self.crop_images(imgs, crop_x, crop_y, crop_width, crop_height)
        # Reshape for processing
        img_cropped = img_cropped.reshape(B * N, C, crop_height, crop_width)
        if self.use_grid_mask:
            img_cropped = self.grid_mask(img_cropped)
        rois_feats = self.img_backbone(img_cropped)

        img_feats = self.pre_key_feats
        x_p = int(crop_x / 8)
        y_p = int(crop_y / 8)

        # print(rois_feats.shape)

        # previous key features to padding
        for i, feats in enumerate(rois_feats):
            # print(feats.shape)
            print(img_feats[i].shape)
            target_shape = feats.shape[2:]  # Get the actual shape of the output feature map
            # print(target_shape)
            # print(y_p, target_shape[0], x_p, target_shape[1])
            img_feats[i][..., y_p:y_p+target_shape[0], x_p:x_p+target_shape[1]] = feats
            x_p = int(x_p / 2)
            y_p = int(y_p / 2)
        
        return img_feats

    def crop_images(self, img, crop_x, crop_y, crop_width, crop_height):
        # Perform cropping
        return img[..., crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

    def extract_img_feat(self, img, len_queue=None):
        """Extract features of images."""
        # print(img.size())
        if img is None:
            return None
        print(img.dim(), img.size())
        assert img.dim() == 5
        B, N, C, H, W = img.size()
        # print(img.size()) # torch.Size([1, 6, 3, 928, 1600])
        # self.kf = 10

        t1 = time.time()
        print(self.counter)

        self.ROI = True
        self.STATIC = False

        cam_num = 6
        # file = open('logs/AR+FTS_feature.txt', 'a+')

        if self.counter != 0 and self.ROI:
            if not self.STATIC:
                # img_feats, cam_num = self.images_rois_spli_merge(img)
                img_feats, cam_num = self.images_rois_spli_merge_unified(img)
            else:
                img_feats = self.images_rois_spli_merge_static(img)
        else:
            img_cropped = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img_cropped = self.grid_mask(img_cropped)
            img_feats = self.img_backbone(img_cropped)
            self.pre_key_feats = img_feats
            self.mask_roi_name = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        
        # print(len(img_feats),img_feats[0].shape, img_feats[1].shape, img_feats[2].shape)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        torch.cuda.synchronize()
        t2 = time.time()
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        torch.cuda.synchronize()
        t3 = time.time()
        print(t2-t1, t3-t2, t3-t1, cam_num)
        # file.write("%s, %s, %s, %s\n"% (t2-t1, t3-t2, t3-t1, cam_num))
        # file.close()

        img_feats_reshaped = []
        for img_feat in img_feats:
            # print(img_feat.shape)
            _, c, h, w = img_feat.size()
            if len_queue is not None:
                img_feat_reshaped = img_feat.view(B//len_queue, len_queue, N, c, h, w)
            else:
                img_feat_reshaped = img_feat.view(B, N, c, h, w)
            img_feats_reshaped.append(img_feat_reshaped)
            # print(img_feat_reshaped.shape)
        # print(img_feats_reshaped.shape)
        self.counter += 1
        if self.counter % self.kf == 0:
            self.counter = 0
        return img_feats_reshaped

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape  # (300, 256 * 2)
        device = self.query_embedding.weight.device
        query = self.query_embedding.weight
        track_instances.ref_pts = self.reference_points(query[..., : dim // 2])

        # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz
        pred_boxes_init = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.query = query

        track_instances.output_embedding = torch.zeros(
            (num_queries, dim >> 1), device=device
        )

        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )
        track_instances.disappear_time = torch.zeros(
            (len(track_instances),), dtype=torch.long, device=device
        )

        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        # xy, wl, z, h, sin, cos, vx, vy, vz
        track_instances.pred_boxes = pred_boxes_init

        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros(
            (len(track_instances), mem_bank_len, dim // 2),
            dtype=torch.float32,
            device=device,
        )
        track_instances.mem_padding_mask = torch.ones(
            (len(track_instances), mem_bank_len), dtype=torch.bool, device=device
        )
        track_instances.save_period = torch.zeros(
            (len(track_instances),), dtype=torch.float32, device=device
        )

        return track_instances.to(self.query_embedding.weight.device)

    def velo_update(
        self, ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta
    ):
        """
        Args:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
            velocity (Tensor): (num_query, 2). m/s
                in lidar frame. vx, vy
            global2lidar (np.Array) [4,4].
        Outs:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
        """
        # print(l2g_r1.type(), l2g_t1.type(), ref_pts.type())
        time_delta = time_delta.type(torch.float)
        num_query = ref_pts.size(0)
        velo_pad_ = velocity.new_zeros((num_query, 1))
        velo_pad = torch.cat((velocity, velo_pad_), dim=-1)

        reference_points = ref_pts.sigmoid().clone()
        pc_range = self.pc_range
        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        reference_points = reference_points + velo_pad * time_delta

        ref_pts = reference_points @ l2g_r1 + l2g_t1 - l2g_t2

        g2l_r = torch.linalg.inv(l2g_r2).type(torch.float)

        ref_pts = ref_pts @ g2l_r

        ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]) / (
            pc_range[3] - pc_range[0]
        )
        ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]) / (
            pc_range[4] - pc_range[1]
        )
        ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]) / (
            pc_range[5] - pc_range[2]
        )

        ref_pts = inverse_sigmoid(ref_pts)

        return ref_pts

    def get_history_bev(self, imgs_queue, img_metas_list):
        """
        Get history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            # print(bs, len_queue)
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_img_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev, _ = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, 
                    img_metas=img_metas, 
                    prev_bev=prev_bev)
        self.train()
        return prev_bev

    # Generate bev using bev_encoder in BEVFormer
    def get_bevs(self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None):
        # print(prev_img_metas)
        if prev_img is not None and prev_img_metas is not None:
            assert prev_bev is None
            prev_bev = self.get_history_bev(prev_img, prev_img_metas)

        img_feats = self.extract_img_feat(img=imgs)
        if self.freeze_bev_encoder:
            with torch.no_grad():
                bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)
        else:
            bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)
        
        if bev_embed.shape[1] == self.bev_h * self.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)
        
        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        # print(bev_embed.shape)
        # print(bev_pos)
        return bev_embed, bev_pos

    def select_active_track_query(self, track_instances, active_index, img_metas, with_mask=True):
        result_dict = self._track_instances2results(track_instances[active_index], img_metas, with_mask=with_mask)
        result_dict["track_query_embeddings"] = track_instances.output_embedding[active_index][result_dict['bbox_index']][result_dict['mask']]
        result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[active_index][result_dict['bbox_index']][result_dict['mask']]
        return result_dict
    
    def select_sdc_track_query(self, sdc_instance, img_metas):
        out = dict()
        result_dict = self._track_instances2results(sdc_instance, img_metas, with_mask=False)
        out["sdc_boxes_3d"] = result_dict['boxes_3d']
        out["sdc_scores_3d"] = result_dict['scores_3d']
        out["sdc_track_scores"] = result_dict['track_scores']
        out["sdc_track_bbox_results"] = result_dict['track_bbox_results']
        out["sdc_embedding"] = sdc_instance.output_embedding[0]
        return out

    def upsample_bev_if_tiny(self, outs_track):
        if outs_track["bev_embed"].size(0) == 100 * 100:
            # For tiny model
            # bev_emb
            bev_embed = outs_track["bev_embed"] # [10000, 1, 256]
            dim, _, _ = bev_embed.size()
            w = h = int(math.sqrt(dim))
            assert h == w == 100

            bev_embed = rearrange(bev_embed, '(h w) b c -> b c h w', h=h, w=w)  # [1, 256, 100, 100]
            bev_embed = nn.Upsample(scale_factor=2)(bev_embed)  # [1, 256, 200, 200]
            bev_embed = rearrange(bev_embed, 'b c h w -> (h w) b c')
            outs_track["bev_embed"] = bev_embed

            # prev_bev
            prev_bev = outs_track.get("prev_bev", None)
            if prev_bev is not None:
                if self.training:
                    #  [1, 10000, 256]
                    prev_bev = rearrange(prev_bev, 'b (h w) c -> b c h w', h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(prev_bev)  # [1, 256, 200, 200]
                    prev_bev = rearrange(prev_bev, 'b c h w -> b (h w) c')
                    outs_track["prev_bev"] = prev_bev
                else:
                    #  [10000, 1, 256]
                    prev_bev = rearrange(prev_bev, '(h w) b c -> b c h w', h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(prev_bev)  # [1, 256, 200, 200]
                    prev_bev = rearrange(prev_bev, 'b c h w -> (h w) b c')
                    outs_track["prev_bev"] = prev_bev

            # bev_pos
            bev_pos  = outs_track["bev_pos"]  # [1, 256, 100, 100]
            bev_pos = nn.Upsample(scale_factor=2)(bev_pos)  # [1, 256, 200, 200]
            outs_track["bev_pos"] = bev_pos
        return outs_track

    def display_images_separately(self, images):
        for i, img in enumerate(images):
            # Convert from RGB to BGR for displaying with OpenCV
            img_bgr = cv2.cvtColor(img.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
            
            # Create a window for each image
            cv2.imshow(f'Camera {i+1}', img_bgr)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def _forward_single_frame_inference(
        self,
        img,
        img_metas,
        track_instances,
        prev_bev=None,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
    ):
        """
        img: B, num_cam, C, H, W = img.shape
        """

        """ velo update """

        print("_forward_single_frame_inference")
        # file = open('logs/AR+FTS_track.txt', 'a+')
        t0 = time.time()

        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            ref_pts = active_inst.ref_pts
            velo = active_inst.pred_boxes[:, -2:]
            ref_pts = self.velo_update(
                ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta
            )
            ref_pts = ref_pts.squeeze(0)
            dim = active_inst.query.shape[-1]
            active_inst.ref_pts = self.reference_points(active_inst.query[..., :dim//2])
            active_inst.ref_pts[...,:2] = ref_pts[...,:2]

        track_instances = Instances.cat([other_inst, active_inst])

        t1 = time.time()

        # print(img_metas[0]['lidar2img'])
        # print(img_metas[0]['cam_intrinsic'])
        scene_token = img_metas[0]['scene_token']
        scene_rec = self.nusc.get('scene', scene_token)
        scene_log = self.nusc.get('log', scene_rec['log_token'])
        # print(scene_log)
        # print(img_metas[0].keys())
        # print(img)

        # grid_order = [[2, 0, 1], [4, 3, 5]]  # Custom order specified
        # self.display_images_separately(img[0])

        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        bev_embed, bev_pos = self.get_bevs(img, img_metas, prev_bev=prev_bev)

        torch.cuda.synchronize()
        t2 = time.time()
        det_output = self.pts_bbox_head.get_detections(
            bev_embed, 
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )

        torch.cuda.synchronize()
        t3 = time.time()
        
        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        # for i in output_classes:
        #     print(i)

        high_score_indices = output_classes > 0.5

        # print(output_classes[high_score_indices])
        # print(output_coords[high_score_indices])
        # print(output_coords.shape, output_classes.shape)

        out = {
            "pred_logits": output_classes,
            "pred_boxes": output_coords,
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "query_embeddings": query_feats,
            "all_past_traj_preds": det_output["all_past_traj_preds"],
            "bev_pos": bev_pos,
        }

        """ update track instances with predict results """
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
        # each track will be assigned an unique global id by the track base.
        track_instances.scores = track_scores
        # track_instances.track_scores = track_scores  # [300]
        track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        track_instances.ref_pts = last_ref_pts[0]
        # hard_code: assume the 901 query is sdc query 
        track_instances.obj_idxes[900] = -2
        """ update track base """
        self.track_base.update(track_instances, None)
       
        active_index = (track_instances.obj_idxes>=0) & (track_instances.scores >= self.track_base.filter_score_thresh)    # filter out sleep objects
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        out.update(self.select_sdc_track_query(track_instances[track_instances.obj_idxes==-2], img_metas))

        """ update with memory_bank """
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        """  Update track instances using matcher """
        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances_fordet"] = track_instances
        out["track_instances"] = out_track_instances
        out["track_obj_idxes"] = track_instances.obj_idxes

        torch.cuda.synchronize()
        t4 = time.time()
        print(t1-t0, t2-t1, t3-t2, t4-t3, t4-t0)

        # file.write("%s, %s, %s, %s, %s\n"% (t1-t0, t2-t1, t3-t2, t4-t3, t4-t0))
        # file.close()
        return out

    def simple_test_track(
        self,
        img=None,
        l2g_t=None,
        l2g_r_mat=None,
        img_metas=None,
        timestamp=None,
        command=None
    ):
        """only support bs=1 and sequential input"""

        bs = img.size(0)
        self.command = command
        # img_metas = img_metas[0]

        """ init track instances for first frame """
        if (
            self.test_track_instances is None
            or img_metas[0]["scene_token"] != self.scene_token
        ):
            self.timestamp = timestamp
            self.scene_token = img_metas[0]["scene_token"]
            self.prev_bev = None
            track_instances = self._generate_empty_tracks()
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None
            
        else:
            track_instances = self.test_track_instances
            time_delta = timestamp - self.timestamp
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t
        
        """ get time_delta and l2g r/t infos """
        """ update frame info for next frame"""
        self.timestamp = timestamp
        self.l2g_t = l2g_t
        self.l2g_r_mat = l2g_r_mat

        """ predict and update """
        prev_bev = self.prev_bev
        frame_res = self._forward_single_frame_inference(
            img,
            img_metas,
            track_instances,
            prev_bev,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
        )

        self.prev_bev = frame_res["bev_embed"]
        track_instances = frame_res["track_instances"]
        track_instances_fordet = frame_res["track_instances_fordet"]
        # print(track_instances_fordet)

        self.test_track_instances = track_instances
        results = [dict()]
        get_keys = ["bev_embed", "bev_pos", 
                    "track_query_embeddings", "track_bbox_results", 
                    "boxes_3d", "scores_3d", "labels_3d", "track_scores", "track_ids"]
        if self.with_motion_head:
            get_keys += ["sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
        results[0].update({k: frame_res[k] for k in get_keys})
        results = self._det_instances2results(track_instances_fordet, results, img_metas)
        return results
    
    def _track_instances2results(self, track_instances, img_metas, with_mask=True):
        bbox_dict = dict(
            cls_scores=track_instances.pred_logits,
            bbox_preds=track_instances.pred_boxes,
            track_scores=track_instances.scores,
            obj_idxes=track_instances.obj_idxes,
        )
        # bboxes_dict = self.bbox_coder.decode(bbox_dict, with_mask=with_mask)[0]
        bboxes_dict = self.bbox_coder.decode(bbox_dict, with_mask=with_mask, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]
        bbox_index = bboxes_dict["bbox_index"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            bbox_index=bbox_index.cpu(),
            track_ids=obj_idxes.cpu(),
            mask=bboxes_dict["mask"].cpu(),
            track_bbox_results=[[bboxes.to("cpu"), scores.cpu(), labels.cpu(), bbox_index.cpu(), bboxes_dict["mask"].cpu()]]
        )
        return result_dict

    def _det_instances2results(self, instances, results, img_metas):
        """
        Outs:
        active_instances. keys:
        - 'pred_logits':
        - 'pred_boxes': normalized bboxes
        - 'scores'
        - 'obj_idxes'
        out_dict. keys:
            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - track_ids
            - tracking_score
        """
        # filter out sleep querys
        if instances.pred_logits.numel() == 0:
            return [None]
        bbox_dict = dict(
            cls_scores=instances.pred_logits,
            bbox_preds=instances.pred_boxes,
            track_scores=instances.scores,
            obj_idxes=instances.obj_idxes,
        )
        # print(instances.pred_boxes[0])
        bboxes_dict = self.bbox_coder.decode(bbox_dict, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        # print(bboxes[:,3:], len(bboxes))
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        # print(bboxes[0], len(bboxes))

        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]

        # print(img_metas[0])

        # print(img_metas)
        t1 = time.time()
        # self.bev_to_uv_coor(bboxes, labels, scores, img_metas[0]["sample_idx"])
        # if not self.STATIC and self.ROI:
            # self.bev_to_uv_coor(bboxes, labels, scores, img_metas)
        self.bev_to_uv_coor(bboxes, labels, scores, img_metas)
        t2 = time.time()
        print("bev_to_uv time: %s", str(t2-t1))

        induces = scores > 0.5

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = results[0]
        result_dict_det = dict(
            boxes_3d_det=bboxes.to("cpu"),
            scores_3d_det=scores.cpu(),
            labels_3d_det=labels.cpu(),
        )
        if result_dict is not None:
            result_dict.update(result_dict_det)
        else:
            result_dict = None

        return [result_dict]
    
    def calculate_ttc(self, x1, y1, vx, vy, v):
        """
        Calculate the Time to Collision (TTC) for the given object and ego vehicle parameters.

        Parameters:
        x1 (float): x-coordinate of the object's position.
        y1 (float): y-coordinate of the object's position.
        vx (float): x-component of the object's velocity.
        vy (float): y-component of the object's velocity.
        v (float): x-component of the ego vehicle's velocity (ego vehicle moves along x-axis).

        Returns:
        float: The time to collision (TTC) or infinity if no collision is expected.
        """
        # Relative velocities
        vx_r = vx - v
        vy_r = vy

        # Calculate potential collision times in x and y directions
        if vx_r != 0:
            t_x = -x1 / vx_r
        else:
            t_x = float('inf')  # No collision in x direction

        if vy_r != 0:
            t_y = -y1 / vy_r
        else:
            t_y = float('inf')  # No collision in y direction

        # Check if the times to collision are approximately equal
        if abs(t_x - t_y) < 1e-6 and t_x > 0:
            return t_x
        else:
            return float('inf')  # No collision
    
    def bev_to_uv_coor(self, bboxes, labels, scores, img_metas):
        sample_idx = img_metas[0]["sample_idx"]
        sample = self.nusc.get('sample', sample_idx)

        induces = scores > 0.5
        bboxes = bboxes[induces]
        labels = labels[induces]

        ego_velocity = img_metas[0]['can_bus'][-5:-2]
        # print(ego_velocity)

        ttc = 5
        for bbox in bboxes:
            # print("Center:", bbox[:3])
            x1, y1, _ = bbox[:3]
            # print("Velocities:", bbox[7:])
            vx, vy = bbox[7:]
            front_distance = bbox[0]
            ttc_res = self.calculate_ttc(x1, y1, vx, vy, ego_velocity[0])
            if ttc < 5:
                ttc = min(ttc_res, ttc)
                print(ttc)
        # print(ttc)
        # rospy.set_param("/synchronizer/slop", ttc/20)
        self.kf = (ttc - 1)*2

        pred_centers = bboxes.gravity_center.cpu().detach().numpy()
        pred_dims = bboxes.dims.cpu().detach().numpy()
        pred_yaw = bboxes.yaw.cpu().detach().numpy()

        # lidar_cs_record = {'translation': [0.943713, 0.0, 1.84023], 
        #                         'rotation': [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817]}

        lidar_cs_record = {'translation': [0.985793, 0.0, 1.84019], 
                                'rotation': [0.706749235646644, -0.015300993788500868, 0.01739745181256607, -0.7070846669051719]}
        # print("lidar_cs_record")
        # print(lidar_cs_record)
        # file_roi = open('roi_generator_time_records.txt', 'a+')

        for cam in self.mask_roi_name:

            t0 = time.time()
            # torch.cuda.synchronize()

            # print(cam)
            sample_token = sample['data'][cam]
            sample_data = self.nusc.get('sample_data', sample_token)
            # print(sample_data['filename'])
            # image_path = "/home/mobilitylab/v1.0-mini/" + sample_data['filename']  # Assuming sample_data['filename'] contains the path to the image
            # image = cv2.imread(image_path)
            sd_record = self.nusc.get('sample_data', sample_token)
            cs_record = self.nusc.get('calibrated_sensor',
                             sd_record['calibrated_sensor_token'])
            # print("cs_record")
            # if cam == 'CAM_FRONT':
            #     print(cs_record['translation'], cs_record['rotation'])
            sensor_record = self.nusc.get('sensor', cs_record['sensor_token'])
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])

            t1 = time.time()
            # torch.cuda.synchronize()

            box_list = []
            # file_box = open('box_convert_time_records.txt', 'a+')
            for i, box in enumerate(bboxes):
                # print(box)
                # print(pred_centers[i], pred_dims[i], pred_yaw[i])
                box = Box(pred_centers[i], pred_dims[i], Quaternion(axis=(0.0, 0.0, 1.0), radians=pred_yaw[i]),
                        name=labels[i], token='predicted')
                
                box.rotate(Quaternion(lidar_cs_record['rotation']))
                box.translate(np.array(lidar_cs_record['translation']))

                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

                if not box_in_image(box, cam_intrinsic, imsize):
                    continue
                # box_list.append(box)
                uv_box = box.render_convert(view=cam_intrinsic, normalize=True)
                # print(uv_box)

                # Correcting the order to [xmin, ymin, xmax, ymax]
                uv_box_corrected = [uv_box[0], uv_box[2], uv_box[1], uv_box[3]]

                box_list.append(uv_box_corrected)
            
            t2 = time.time()
            # torch.cuda.synchronize()

            # Compute the minimal bounding box to cover all detected bounding boxes
            minimal_bbox = self.compute_minimal_bounding_box(box_list)
            if minimal_bbox:
                self.camera_bboxes[cam] = minimal_bbox
                # print("Minimal bounding box for camera {}: {}".format(cam, minimal_bbox))
            else:
                self.camera_bboxes[cam] = [0, 0, 0, 0]
                # print("No valid bounding boxes found for camera {}".format(cam))

            # Draw a rectangle with the corrected and converted coordinates
            # for uv_box_int in box_list:
            #     cv2.rectangle(image, (uv_box_int[0], uv_box_int[1]), (uv_box_int[2], uv_box_int[3]), (0, 0, 255), 2)

            # Using OpenCV to display the image
            # cv2.imshow('Image with Bounding Box', image)
            # cv2.waitKey(0)  # Wait until a key press to close the window
            # cv2.destroyAllWindows()

            t3 = time.time()
            # torch.cuda.synchronize()

        #     file_roi.write("%s, %s, %s, %s, %s, %s\n"% (cam,
        #                     len(bboxes), t1-t0, t2-t1, t3-t2, t3-t0))
        # file_roi.close()

    
    def compute_minimal_bounding_box(self, box_list):
        if not box_list:
            return None  # Return None or an appropriate value if there are no boxes
        
        # Initialize min and max coordinates with extreme values
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        # Iterate through all boxes to find the extreme values
        for box in box_list:
            min_x = min(min_x, box[0])  # xmin
            min_y = min(min_y, box[1])  # ymin
            max_x = max(max_x, box[2])  # xmax
            max_y = max(max_y, box[3])  # ymax
        
        # Return the bounding box that covers all boxes
        return [min_x, min_y, max_x, max_y]

