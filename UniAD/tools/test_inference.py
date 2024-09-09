#!/root/anaconda3/envs/uniad/bin/python3.8
import argparse
import cv2
import torch
import sklearn
import mmcv
import os
import warnings
from mmengine import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.uniad.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import rospy
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

warnings.filterwarnings("ignore")

class MMDetROSNode:
    def __init__(self):
        rospy.init_node('mmdet_ros_node', anonymous=True)  # Initialize ROS node
        self.args = self.parse_args()
        self.cfg = self.load_config(self.args.config)
        self.distributed = self.init_distributed()
        self.dataset, self.data_loader = self.build_dataloader(self.cfg.data.test)
        self.batches = list(iter(self.data_loader))
        self.model = self.build_model(self.cfg)
        self.rank, _ = get_dist_info()
        self.prev_param_value = None  # Initialize previous parameter value
        with torch.no_grad():
            data_init = self.batches[0]
            res_init = self.model(return_loss=False, rescale=True, **self.batches[0])
        # Clear CUDA cache to free memory
        del data_init
        del res_init
        torch.cuda.empty_cache()

        # Timer to check ROS parameter every second
        rospy.Timer(rospy.Duration(0.2), self.check_rosparam_and_infer)

    def parse_args(self):
        parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
        parser.add_argument('--config', default='/home/mobilitylab/UniAD/projects/configs/stage1_track_map/base_track_map.py', help='test config file path')
        parser.add_argument('--checkpoint', default='/home/mobilitylab/UniAD/ckpts/uniad_base_track_map.pth', help='checkpoint file')
        parser.add_argument('--out', default='output/results.pkl', help='output result file in pickle format')
        parser.add_argument('--fuse-conv-bn', action='store_true', help='Whether to fuse conv and bn, this will slightly increase the inference speed')
        parser.add_argument('--format-only', action='store_true', help='Format the output results without perform evaluation.')
        parser.add_argument('--eval', default='bbox', type=str, nargs='+', help='evaluation metrics, which depends on the dataset, e.g., "bbox", "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
        parser.add_argument('--show', action='store_true', help='show results')
        parser.add_argument('--show-dir', help='directory where results will be saved')
        parser.add_argument('--gpu-collect', action='store_true', help='whether to use gpu to collect results.')
        parser.add_argument('--tmpdir', help='tmp directory used for collecting results from multiple workers, available when gpu-collect is not specified')
        parser.add_argument('--seed', type=int, default=0, help='random seed')
        parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
        parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.')
        parser.add_argument('--options', nargs='+', action=DictAction, help='custom options for evaluation, the key-value pair in xxx=yyy format will be kwargs for dataset.evaluate() function (deprecate), change to --eval-options instead.')
        parser.add_argument('--eval-options', nargs='+', action=DictAction, help='custom options for evaluation, the key-value pair in xxx=yyy format will be kwargs for dataset.evaluate() function')
        parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
        parser.add_argument('--local_rank', type=int, default=0)
        args = parser.parse_args()
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(args.local_rank)

        if args.options and args.eval_options:
            raise ValueError('--options and --eval-options cannot be both specified, --options is deprecated in favor of --eval-options')
        if args.options:
            warnings.warn('--options is deprecated in favor of --eval-options')
            args.eval_options = args.options
        return args

    def load_config(self, config_path):
        cfg = Config.fromfile(config_path)
        if self.args.cfg_options is not None:
            cfg.merge_from_dict(self.args.cfg_options)
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])
        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                import importlib
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)
                else:
                    _module_dir = os.path.dirname(self.args.config)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)
        return cfg

    def init_distributed(self):
        if self.args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(self.args.launcher, **self.cfg.dist_params)
        return distributed

    def build_dataloader(self, data_cfg):
        dataset = build_dataset(data_cfg)
        samples_per_gpu = 1
        if isinstance(data_cfg, dict):
            data_cfg.test_mode = True
            samples_per_gpu = data_cfg.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                data_cfg.pipeline = replace_ImageToTensor(data_cfg.pipeline)
        elif isinstance(data_cfg, list):
            for ds_cfg in data_cfg:
                ds_cfg.test_mode = True
            samples_per_gpu = max([ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in data_cfg])
            if samples_per_gpu > 1:
                for ds_cfg in data_cfg:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            dist=self.distributed,
            shuffle=False,
            nonshuffler_sampler=self.cfg.data.nonshuffler_sampler,
        )
        return dataset, data_loader

    def build_model(self, cfg):
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, self.args.checkpoint, map_location='cpu')
        if self.args.fuse_conv_bn:
            model = fuse_conv_bn(model)
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = self.dataset.CLASSES
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        elif hasattr(self.dataset, 'PALETTE'):
            model.PALETTE = self.dataset.PALETTE
        if not self.distributed:
            model = MMDataParallel(model, device_ids=[0])
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
        return model

    def check_rosparam_and_infer(self, event):
        # Check the parameter value every second
        param_value = rospy.get_param('/synchronization/frame_id', None)
        if param_value is not None and param_value != self.prev_param_value:
            self.prev_param_value = param_value
            self.run_inference(param_value)

    def run_inference(self, frame_idx):
        try:
            frame_idx = int(frame_idx) % len(self.batches)
            data = self.batches[frame_idx]
            # Ensure no gradients are calculated
            with torch.no_grad():
                result = self.model(return_loss=False, rescale=True, **data)

            rospy.loginfo(f"Inference completed for frame index: {frame_idx}")
            
            # Clear CUDA cache to free memory
            del data
            del result
            torch.cuda.empty_cache()
        except Exception as e:
            rospy.logerr(f"Error processing frame index {frame_idx}: {e}")

    def run(self):
        assert self.args.out or self.args.eval or self.args.format_only or self.args.show or self.args.show_dir, (
            'Please specify at least one operation (save/eval/format/show the results / save the results) with the argument "--out", "--eval", "--format-only", "--show" or "--show-dir"')

        if self.args.eval and self.args.format_only:
            raise ValueError('--eval and --format_only cannot be both specified')

        if self.args.out is not None and not self.args.out.endswith(('.pkl', '.pickle')):
            raise ValueError('The output file must be a pkl file.')

        # Initialize ROS node
        rospy.init_node('mmdet_ros_node', anonymous=True)
        rospy.spin()

if __name__ == '__main__':
    node = MMDetROSNode()
    node.run()
