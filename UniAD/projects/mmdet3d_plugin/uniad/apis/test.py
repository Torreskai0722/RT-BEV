import os
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from ..dense_heads.occ_head_plugin import IntersectionOverUnion, PanopticMetric
from ..dense_heads.planning_head_plugin import PlanningMetric

import mmcv
import numpy as np
import pycocotools.mask as mask_util
from mmcv.parallel import DataContainer as DC
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.core.bbox import LiDARInstance3DBoxes


def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def get_can_bus_info(nusc, nusc_can_bus, sample, scene_name):
    # scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0., 0.])
    return np.array(can_bus)

# def get_lidar2img_transforms(info):
#     lidar2img_rts = []
#     lidar2cam_rts = []
#     cam_intrinsics = []
#     for cam_type, cam_info in info['cams'].items():
#         # obtain lidar to image transformation matrix
#         lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
#         lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
#         lidar2cam_rt = np.eye(4)
#         lidar2cam_rt[:3, :3] = lidar2cam_r.T
#         lidar2cam_rt[3, :3] = -lidar2cam_t
#         intrinsic = cam_info['cam_intrinsic']
#         viewpad = np.eye(4)
#         viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
#         lidar2img_rt = (viewpad @ lidar2cam_rt.T)
#         lidar2img_rts.append(lidar2img_rt)
#         cam_intrinsics.append(viewpad)
#         lidar2cam_rts.append(lidar2cam_rt.T)
#     return lidar2img_rts, lidar2cam_rts, cam_intrinsics

def imgs_process_pipeline(results, img_root):
    images_multiView = []
    for img_path in results['img_filename']:
        img_path = os.path.join(img_root, img_path)
        img = mmcv.imread(img_path, 'unchanged')
        images_multiView.append(img)
    
    img = np.stack(images_multiView, axis=-1).astype(np.float32)
    results['img'] = [img[..., i] for i in range(img.shape[-1])]
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    
    mean = np.array([103.530, 116.280, 123.675], dtype=np.float32)
    std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    results['img'] = [mmcv.imnormalize(img, mean, std, False) for img in results['img']]
    results['img_norm_cfg'] = dict(mean=mean, std=std, to_rgb=False)
    
    size_divisor = 32
    padded_img = [mmcv.impad_to_multiple(img, size_divisor, pad_val=0) for img in results['img']]
    results['ori_shape'] = [img.shape for img in results['img']]
    results['img'] = padded_img
    results['img_shape'] = [img.shape for img in padded_img]
    results['pad_shape'] = [img.shape for img in padded_img]

    nusc = NuScenes(version='v1.0-mini', dataroot='/home/mobilitylab/v1.0-mini', verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot='/home/mobilitylab/v1.0-mini')

    sample_nokey = nusc.get('sample_data', results['sample_idx'])
    sample_key = sample_nokey['sample_token']
    sample = nusc.get('sample', sample_key)

    scene_name = nusc.get('scene', sample['scene_token'])['name']
    can_bus = get_can_bus_info(nusc, nusc_can_bus, sample_nokey, scene_name)

    lidar_token = sample['data']['LIDAR_TOP']
    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    l2e_t = lidar_cs_record['translation']
    l2e_r = lidar_cs_record['rotation']

    cs_record = nusc.get('calibrated_sensor', sample_nokey['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sample_nokey['ego_pose_token'])
    e2g_r = pose_record['rotation']
    e2g_t = pose_record['translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix
    l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T
    l2g_t = l2e_t @ e2g_r_mat.T + e2g_t

    print(cs_record)

    # lidar2img_rts, lidar2cam_rts, cam_intrinsics = get_lidar2img_transforms(cs_record)
    # Static lidar2img transformation matrices
    lidar2img_rts = [
        np.array([[ 1.24381651e+03,  8.39368107e+02,  3.42087518e+01, -5.57350178e+02],
                  [-1.66744603e+01,  5.37473878e+02, -1.22526326e+03, -7.52844510e+02],
                  [-1.07314065e-02,  9.98453425e-01,  5.45490220e-02, -6.66354968e-01],
                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        np.array([[ 1.36448369e+03, -6.20327836e+02, -3.96597524e+01, -3.46324266e+02],
                  [ 3.80615502e+02,  3.20626048e+02, -1.23935878e+03, -7.35374498e+02],
                  [ 8.43808544e-01,  5.35626281e-01,  3.30398047e-02, -7.10054208e-01],
                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        np.array([[ 3.41458798e+01,  1.50306933e+03,  7.85021326e+01, -7.44154062e+02],
                  [-3.87837708e+02,  3.21618006e+02, -1.23761331e+03, -7.47403616e+02],
                  [-8.22756209e-01,  5.66927050e-01,  4.08159246e-02, -6.95949948e-01],
                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        np.array([[-8.04228529e+02, -8.50477977e+02, -2.68547062e+01, -8.12459505e+02],
                  [-1.07097045e+01, -4.45116400e+02, -8.14991575e+02, -6.74168419e+02],
                  [-8.62041514e-03, -9.99189905e-01, -3.93093190e-02, -9.47818294e-01],
                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        np.array([[-1.18659082e+03,  9.23227786e+02,  5.32970683e+01, -6.26700423e+02],
                  [-4.62563185e+02, -1.02430100e+02, -1.25250923e+03, -5.61489117e+02],
                  [-9.47579232e-01, -3.19505916e-01,  3.09317810e-03, -4.32175821e-01],
                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        np.array([[ 2.84370869e+02, -1.46942079e+03, -5.99160491e+01, -7.64999470e+01],
                  [ 4.45387215e+02, -1.22643975e+02, -1.25017871e+03, -5.59920625e+02],
                  [ 9.23832924e-01, -3.82781620e-01, -3.31067171e-03, -4.13616207e-01],
                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    ]

    results['scene_token'] = sample['scene_token']
    results['timestamp'] = sample_nokey['timestamp']
    results['l2g_r_mat'] = l2g_r_mat
    results['l2g_t'] = l2g_t
    results['can_bus'] = can_bus
    results['lidar2img'] = lidar2img_rts
    # results['lidar2cam'] = lidar2cam_rts
    # results['cam_intrinsic'] = cam_intrinsics
    results['box_type_3d'] = LiDARInstance3DBoxes

    data = {}
    img_metas = {}
    meta_keys = [
        'filename', 'ori_shape', 'img_shape', 'lidar2img', 'lidar2cam', 
        'cam_intrinsic', 'depth2img', 'cam2img', 'pad_shape', 
        'img_norm_cfg', 'sample_idx', 'scene_token', 'can_bus', 'box_type_3d'
    ]
    keys = ["img", "timestamp", "l2g_r_mat", "l2g_t"]

    for key in meta_keys:
        if key in results:
            img_metas[key] = results[key]

    data['img_metas'] = DC(img_metas, cpu_only=True)
    for key in keys:
        data[key] = results[key]
    
    data_dict = {}
    for key, value in data.items():
        if 'l2g' in key:
            data_dict[key] = to_tensor(value[0])
        else:
            data_dict[key] = value

    data_dict = prepare_data_for_model(data_dict)
    return data_dict

def prepare_data_for_model(data):
    img_metas_list = [[data['img_metas'].data]]
    img_tensor = torch.stack([torch.from_numpy(img).float() for img in data['img']])
    
    print(f"Initial img_tensor shape: {img_tensor.shape}")  # Debug statement

    # Transpose to get (num_views, height, width, channels)
    img_tensor = img_tensor.permute(0, 3, 1, 2)  # (6, 3, 928, 1600)
    
    # Add batch dimension (batch_size, num_views, channels, height, width)
    img_tensor = img_tensor.unsqueeze(0)  # (1, 6, 3, 928, 1600)
    
    print(f"Processed img_tensor shape: {img_tensor.shape}")  # Debug statement

    formatted_data = {
        'img': [img_tensor],
        'img_metas': img_metas_list,
        'l2g_t': data['l2g_t'],
        'l2g_r_mat': data['l2g_r_mat'],
        'timestamp': [data['timestamp']]
    }
    return formatted_data

def custom_multi_gpu_test_images(model, data_loader, tmpdir=None, gpu_collect=False):
    
    img_root = '/home/mobilitylab/v1.0-mini/'
    results = {'img_filename': ['samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151604012404.jpg', 
                             'samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151604020482.jpg', 
                             'samples/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151604004799.jpg', 
                             'samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151604037558.jpg', 
                             'samples/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151604047405.jpg', 
                             'samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151604028370.jpg'],
                             'sample_idx': 'e3d495d4ac534d54b321f50006683844'}
    # e3d495d4ac534d54b321f50006683844
    # 'sample_idx': '3950bd41f74548429c0f7700ff3d8269'
    data = imgs_process_pipeline(results, img_root)
    
    model.eval()
        
    bbox_results = []
    mask_results = []
    
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    bbox_results.extend(result)

    ret_results = dict()
    ret_results['bbox_results'] = bbox_results

    if mask_results is not None:
        ret_results['mask_results'] = mask_results
    return ret_results

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()

    # Occ eval init
    eval_occ = hasattr(model.module, 'with_occ_head') \
                and model.module.with_occ_head
    if eval_occ:
        # 30mx30m, 100mx100m at 50cm resolution
        EVALUATION_RANGES = {'30x30': (70, 130),
                            '100x100': (0, 200)}
        n_classes = 2
        iou_metrics = {}
        for key in EVALUATION_RANGES.keys():
            iou_metrics[key] = IntersectionOverUnion(n_classes).cuda()
        panoptic_metrics = {}
        for key in EVALUATION_RANGES.keys():
            panoptic_metrics[key] = PanopticMetric(n_classes=n_classes, temporally_consistent=True).cuda()
    
    # Plan eval init
    eval_planning =  hasattr(model.module, 'with_planning_head') \
                      and model.module.with_planning_head
    if eval_planning:
        planning_metrics = PlanningMetric().cuda()
        
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(max(len(dataset), 100))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    num_occ = 0
    batches = list(iter(data_loader))
    for i, data in enumerate(data_loader):
    # for i in range(5):
        # data = batches[2]
        # if i > 100:
        #     break
        # print(data['img_metas'][0])
        # print(data.keys())
        # print(data['l2g_r_mat'], data['l2g_t'])
        # print(data['img_metas'][0].data[0][0])
        # print(data['img_metas'][0].data[0][0]['can_bus'])
        # print(data['img_metas'][0].data[0][0]['lidar2img'])
        # print(data['sweeps'])
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

            # EVAL planning
            if eval_planning:
                # TODO: Wrap below into a func
                segmentation = result[0]['planning']['planning_gt']['segmentation']
                sdc_planning = result[0]['planning']['planning_gt']['sdc_planning']
                sdc_planning_mask = result[0]['planning']['planning_gt']['sdc_planning_mask']
                pred_sdc_traj = result[0]['planning']['result_planning']['sdc_traj']
                result[0]['planning_traj'] = result[0]['planning']['result_planning']['sdc_traj']
                result[0]['planning_traj_gt'] = result[0]['planning']['planning_gt']['sdc_planning']
                result[0]['command'] = result[0]['planning']['planning_gt']['command']
                planning_metrics(pred_sdc_traj[:, :6, :2], sdc_planning[0][0,:, :6, :2], sdc_planning_mask[0][0,:, :6, :2], segmentation[0][:, [1,2,3,4,5,6]])

            # Eval Occ
            if eval_occ:
                occ_has_invalid_frame = data['gt_occ_has_invalid_frame'][0]
                occ_to_eval = not occ_has_invalid_frame.item()
                if occ_to_eval and 'occ' in result[0].keys():
                    num_occ += 1
                    for key, grid in EVALUATION_RANGES.items():
                        limits = slice(grid[0], grid[1])
                        iou_metrics[key](result[0]['occ']['seg_out'][..., limits, limits].contiguous(),
                                        result[0]['occ']['seg_gt'][..., limits, limits].contiguous())
                        panoptic_metrics[key](result[0]['occ']['ins_seg_out'][..., limits, limits].contiguous().detach(),
                                                result[0]['occ']['ins_seg_gt'][..., limits, limits].contiguous())

            # Pop out unnecessary occ results, avoid appending it to cpu when collect_results_cpu
            if os.environ.get('ENABLE_PLOT_MODE', None) is None:
                result[0].pop('occ', None)
                result[0].pop('planning', None)
            else:
                for k in ['seg_gt', 'ins_seg_gt', 'pred_ins_sigmoid', 'seg_out', 'ins_seg_out']:
                    if k in result[0]['occ']:
                        result[0]['occ'][k] = result[0]['occ'][k].detach().cpu()
                for k in ['bbox', 'segm', 'labels', 'panoptic', 'drivable', 'score_list', 'lane', 'lane_score', 'stuff_score_list']:
                    if k in result[0]['pts_bbox'] and isinstance(result[0]['pts_bbox'][k], torch.Tensor):
                        result[0]['pts_bbox'][k] = result[0]['pts_bbox'][k].detach().cpu()

            # encode mask results
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    bbox_results.extend(bbox_result)
                if 'mask_results' in result.keys() and result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        else:
            mask_results = None

    if eval_planning:
        planning_results = planning_metrics.compute()
        planning_metrics.reset()

    ret_results = dict()
    ret_results['bbox_results'] = bbox_results
    if eval_occ:
        occ_results = {}
        for key, grid in EVALUATION_RANGES.items():
            panoptic_scores = panoptic_metrics[key].compute()
            for panoptic_key, value in panoptic_scores.items():
                occ_results[f'{panoptic_key}'] = occ_results.get(f'{panoptic_key}', []) + [100 * value[1].item()]
            panoptic_metrics[key].reset()

            iou_scores = iou_metrics[key].compute()
            occ_results['iou'] = occ_results.get('iou', []) + [100 * iou_scores[1].item()]
            iou_metrics[key].reset()

        occ_results['num_occ'] = num_occ  # count on one gpu
        occ_results['ratio_occ'] = num_occ / len(dataset)  # count on one gpu, but reflect the relative ratio
        ret_results['occ_results_computed'] = occ_results
    if eval_planning:
        ret_results['planning_results_computed'] = planning_results

    if mask_results is not None:
        ret_results['mask_results'] = mask_results
    return ret_results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)