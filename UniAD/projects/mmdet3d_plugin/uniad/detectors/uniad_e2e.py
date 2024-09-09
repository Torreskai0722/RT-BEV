#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
import copy
import os
from ..dense_heads.seg_head_plugin import IOU
from .uniad_track import UniADTrack
from mmdet.models.builder import build_head
import time
import sys
import signal
import rospy

def quit(signum, frame):
    print('')
    print('stop uniad_e2e images')
    sys.exit()

@DETECTORS.register_module()
class UniAD(UniADTrack):
    """
    UniAD: Unifying Detection, Tracking, Segmentation, Motion Forecasting, Occupancy Prediction and Planning for Autonomous Driving
    """
    def __init__(
        self,
        seg_head=None,
        motion_head=None,
        occ_head=None,
        planning_head=None,
        task_loss_weight=dict(
            track=1.0,
            map=1.0,
            motion=1.0,
            occ=1.0,
            planning=1.0
        ),
        **kwargs,
    ):
        super(UniAD, self).__init__(**kwargs)
        if seg_head:
            self.seg_head = build_head(seg_head)
        if occ_head:
            self.occ_head = build_head(occ_head)
        if motion_head:
            self.motion_head = build_head(motion_head)
        if planning_head:
            self.planning_head = build_head(planning_head)
        
        self.task_loss_weight = task_loss_weight
        assert set(task_loss_weight.keys()) == \
               {'track', 'occ', 'motion', 'map', 'planning'}
        
        signal.signal(signal.SIGINT, quit)
        signal.signal(signal.SIGTERM, quit)
        
        # self.latency_file = open('uniad-e2e-latency-cropped-500.log','w')

    @property
    def with_planning_head(self):
        return hasattr(self, 'planning_head') and self.planning_head is not None
    
    @property
    def with_occ_head(self):
        return hasattr(self, 'occ_head') and self.occ_head is not None

    @property
    def with_motion_head(self):
        return hasattr(self, 'motion_head') and self.motion_head is not None

    @property
    def with_seg_head(self):
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_test(self,
                     img=None,
                     img_metas=None,
                     l2g_t=None,
                     l2g_r_mat=None,
                     timestamp=None,
                     gt_lane_labels=None,
                     gt_lane_masks=None,
                     rescale=False,
                     # planning gt(for evaluation only)
                     sdc_planning=None,
                     sdc_planning_mask=None,
                     command=None,
 
                     # Occ_gt (for evaluation only)
                     gt_segmentation=None,
                     gt_instance=None, 
                     gt_occ_img_is_valid=None,
                     **kwargs
                    ):
        """Test function
        """

        torch.cuda.empty_cache()
        # file = open('logs/AR+FTS_e2e.txt', 'a+')

        t0 = time.time()

        # print(img_metas[0][0])
        # print(img_metas[0][0]['can_bus'])

        print(command[0].item())
        rospy.set_param("/synchronizer/command", command[0].item())


        # print(img_metas[0][0]['lidar2img'])
        # print(img_metas[0][0]['box_type_3d'])


        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        # print(tmp_angle)
        # print(tmp_pos)
        # first frame
        if self.prev_frame_info['scene_token'] is None:
            img_metas[0][0]['can_bus'][:3] = 0
            img_metas[0][0]['can_bus'][-1] = 0
        # following frames
        else:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle

        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        t1 = time.time()

        result = [dict() for i in range(len(img_metas))]
        # print(img_metas[0]['filename'])
        # print(timestamp)
        # print(type(img), img)
        result_track = self.simple_test_track(img, l2g_t, l2g_r_mat, img_metas, timestamp, command)

        torch.cuda.synchronize()
        t2 = time.time()

        # Upsample bev for tiny model        
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])
        
        bev_embed = result_track[0]["bev_embed"]

        torch.cuda.synchronize()
        t3 = time.time()

        if self.with_seg_head:
            result_seg =  self.seg_head.forward_test(bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale)

        torch.cuda.synchronize()
        t4 = time.time()

        if self.with_motion_head:
            result_motion, outs_motion = self.motion_head.forward_test(bev_embed, outs_track=result_track[0], outs_seg=result_seg[0])
            outs_motion['bev_pos'] = result_track[0]['bev_pos']

        torch.cuda.synchronize()
        t5 = time.time()

        outs_occ = dict()
        if self.with_occ_head:
            occ_no_query = outs_motion['track_query'].shape[1] == 0
            outs_occ = self.occ_head.forward_test(
                bev_embed, 
                outs_motion,
                no_query = occ_no_query,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
            )
            result[0]['occ'] = outs_occ
        
        torch.cuda.synchronize()
        t6 = time.time()
        
        if self.with_planning_head:
            planning_gt=dict(
                segmentation=gt_segmentation,
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command
            )
            result_planning = self.planning_head.forward_test(bev_embed, outs_motion, outs_occ, command)
            result[0]['planning'] = dict(
                planning_gt=planning_gt,
                result_planning=result_planning,
            )
        
        torch.cuda.synchronize()
        t7 = time.time()

        pop_track_list = ['prev_bev', 'bev_pos', 'bev_embed', 'track_query_embeddings', 'sdc_embedding']
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        if self.with_seg_head:
            result_seg[0] = pop_elem_in_result(result_seg[0], pop_list=['pts_bbox', 'args_tuple'])
        if self.with_motion_head:
            result_motion[0] = pop_elem_in_result(result_motion[0])
        if self.with_occ_head:
            result[0]['occ'] = pop_elem_in_result(result[0]['occ'],  \
                pop_list=['seg_out_mask', 'flow_out', 'future_states_occ', 'pred_ins_masks', 'pred_raw_occ', 'pred_ins_logits', 'pred_ins_sigmoid'])
        
        for i, res in enumerate(result):
            res['token'] = img_metas[i]['sample_idx']
            res.update(result_track[i])
            if self.with_motion_head:
                res.update(result_motion[i])
            if self.with_seg_head:
                res.update(result_seg[i])
        
        torch.cuda.synchronize()
        t8 = time.time()

        # file.write("%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (str(t1-t0),str(t2-t1),str(t3-t2),str(t4-t3),str(t5-t4),str(t6-t5),str(t7-t6),str(t8-t7),str(t8-t0)))
        # file.close()
        print(str(t1-t0),str(t2-t1),str(t3-t2),str(t4-t3),str(t5-t4),str(t6-t5),str(t7-t6),str(t8-t7),str(t8-t0))

        return result


def pop_elem_in_result(task_result:dict, pop_list:list=None):
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
            task_result.pop(k)
    
    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result
