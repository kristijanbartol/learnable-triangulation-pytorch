from mmpose.core import wrap_fp16_model
from mmpose.models.detectors.top_down import TopDown


class HRNet(TopDown):

    def __init__(self,
                 backbone,
                 neck=None,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None):
        super().__init__()

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        pass


def get_pose_net(device='cuda:0'):
    config = Config.fromfile('./configs/hrnet.py')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = pose_builder.build_posenet(config.model)

    fp16_config = config.get('fp16', None)
    if fp16_config is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, config.checkpoint_path, map_location='cpu')

#    if args.fuse_conv_bn:
#        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    return model
