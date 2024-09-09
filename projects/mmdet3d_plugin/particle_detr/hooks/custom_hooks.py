from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time
import torch


@HOOKS.register_module()
class TransferWeight(Hook):
    
    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters
    
    def before_run(self, runner):
        """In case of using a pretrained model, set the weights here"""
        # Load the pretrained weights        
        if runner.model.module.use_pretrained_weights == True and \
            runner.model.module.using_pretrained_weights == False:
            runner.model.module.load_state_dict(torch.load("ckpts/diff_pretrained_tiny_24_epochs.pth"))

            # Set the necessary components to not require grads
            for param in runner.model.module.img_backbone.parameters():
                param.requires_grad = False
            for param in runner.model.module.img_neck.parameters():
                param.requires_grad = False
            for param in runner.model.module.pts_bbox_head.positional_encoding.parameters():
                param.requires_grad = False
            for param in runner.model.module.pts_bbox_head.bev_embedding.parameters():
                param.requires_grad = False
            for param in runner.model.module.pts_bbox_head.transformer.encoder.parameters():
                param.requires_grad = False
            for param in runner.model.module.pts_bbox_head.transformer.can_bus_mlp.parameters():
                param.requires_grad = False
            runner.model.module.pts_bbox_head.transformer.level_embeds.require_grad = False
            runner.model.module.pts_bbox_head.transformer.cams_embeds.require_grad = False
        
            runner.model.module.using_pretrained_weights = True