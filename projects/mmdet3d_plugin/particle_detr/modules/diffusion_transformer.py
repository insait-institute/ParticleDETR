# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from typing import Optional
import matplotlib.pyplot as plt

import torch.nn.functional as F
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import inverse_sigmoid
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from .transformer import PerceptionTransformer
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16
from collections import namedtuple
import math
import random

from detectron2.layers import batched_nms, batched_nms_rotated
from detectron2.structures import Boxes, BoxMode, ImageList, Instances, RotatedBoxes

from collections import namedtuple

AnnotObject = namedtuple("AnnotObject", ["gt_boxes", "gt_classes"]) # We keep just the (cx, cy, cz) point

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

@TRANSFORMER.register_module()
class DiffusionTransformer(PerceptionTransformer):

    def __init__(self, *args, **kwargs):
        self.DIFFUSION_CFG = kwargs['DIFFUSION_CFG']
        DIFFUSION_CFG = kwargs.pop("DIFFUSION_CFG", None)
        super(DiffusionTransformer, self).__init__(*args, **kwargs)
        # Setup the diffusion parameters
        self.num_classes = DIFFUSION_CFG.NUM_CLASSES
        self.num_proposals = DIFFUSION_CFG.NUM_PROPOSALS
        self.hidden_dim = DIFFUSION_CFG.HIDDEN_DIM
        self.num_heads = DIFFUSION_CFG.NUM_HEADS
        self.radial_suppression_radius = DIFFUSION_CFG.RADIAL_SUPPRESSION_RADIUS # A type of NMS
        self.ddim_query_type = DIFFUSION_CFG.DDIM_QUERY_TYPE # either 'both', or 'diffusion', or 'bevformer'
        if isinstance(self.radial_suppression_radius, list):
            self.radial_suppression_radius = self.radial_suppression_radius[0]
        self.box_renewal_threshold = DIFFUSION_CFG.BOX_RENEWAL_THRESHOLD
        timesteps = 1000
        sampling_timesteps = DIFFUSION_CFG.SAMPLE_STEP
        self.NMS_THRESHOLD = float(DIFFUSION_CFG.NMS_THRESHOLD)
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 0.
        self.self_condition = False
        self.scale = DIFFUSION_CFG.SNR_SCALE
        self.box_renewal = True
        self.use_ensemble = True
        self.use_nms = True

        # Diffusion buffers and parameters
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))
        
        # Diffusion reference point to positional encoding
        self.query_pos_net = nn.Linear(2, self.embed_dims)

        # This time embedding is at the level of the diffusion transformer
        self.time_dim = 4 * self.embed_dims
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(self.embed_dims), 
            nn.Linear(self.embed_dims, self.time_dim), nn.GELU(), nn.Linear(self.time_dim, self.time_dim),)
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(self.embed_dims * 4, self.embed_dims * 2))

        # BEVFormer positional encodings to reference points
        self.positional_encoding_to_reference_point = nn.Linear(self.embed_dims, 2)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
        xavier_init(self.query_pos_net)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embeds_bevformer', 'object_query_embeds_diffusion', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embeds_bevformer,
                object_query_embeds_diffusion,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                gt_bboxes_3d=None,
                gt_labels_3d=None,
                pc_range=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            bev_h, bev_w: int, the number of BEV cells in each direction
            grid_length: list[float, float], the spatial size of each BEV cell, i.e. 2*51.2/bev_h
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
            gt_bboxes_3d: list[LidarInstance3DBoxes], a list of the 3D bounding boxes
            gt_labels_3d: list[torch.Tensor], a list of the labels in each sample
            pc_range: list[float], the minimum and maximum point cloud distances
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        # Get the BEV
        bev_embed = self.get_bev_features(
            mlvl_feats=mlvl_feats,
            bev_queries=bev_queries,
            bev_h=bev_h,
            bev_w=bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)
        
        # Train/test logic branches here
        if (gt_bboxes_3d is None) and (gt_labels_3d is None):
            # Test logic - sample random reference points
            outputs_class, outputs_coord = self.ddim_sample(backbone_feats=bev_embed, 
                object_query_embeds_bevformer=object_query_embeds_bevformer,
                object_query_embeds_diffusion=object_query_embeds_diffusion,
                reg_branches=reg_branches, cls_branches=cls_branches, pc_range=pc_range,
                bev_h=bev_h, bev_w=bev_w, **kwargs)
            return bev_embed, outputs_class, outputs_coord
        else:
            # Training - prepare the targets
            targets, x_boxes, noises, t = self.prep_targets(gt_bboxes_3d, gt_labels_3d, bev_params=(bev_h, bev_w, grid_length, pc_range))
            bs = x_boxes.size(0)
            
            # Diffusion queries - convert the boxes coordinates into [0, 1] with (0, 0) being top-left and (1, 1) bottom-right.
            # First number is positive to the right of the car, second number is positive in front of the car.
            device = x_boxes.device
            x_boxes[..., 1] = 1 - x_boxes[..., 1]

            # Get the object queries as the BEV features in the noisy positions
            if object_query_embeds_diffusion is None:
                query_diffusion = torch.nn.functional.grid_sample(bev_embed.view(bs, bev_h, bev_w, self.embed_dims).permute(0, 3, 1, 2), 
                    2 * x_boxes[:, None, :, :].float() - 1, mode='bilinear', align_corners=False) # (B, C, 1, Q)
                query_diffusion = query_diffusion.squeeze(2).permute(0, 2, 1).detach() # (B, Q, C)
            else:
                query_diffusion = torch.nn.functional.grid_sample(object_query_embeds_diffusion[None].repeat(bs, 1, 1).permute(0, 2, 1).view(bs, self.embed_dims, 30, 30),
                    2 * x_boxes[:, None, :, :].float() - 1, mode='bilinear', align_corners=False) # (B, C, 1, Q)
                query_diffusion = query_diffusion.squeeze(2).permute(0, 2, 1) # (B, Q, C)
            
            # Handle the BEVFormer queries
            query_pos_bevformer, query_bevformer = torch.split(object_query_embeds_bevformer, self.embed_dims, dim=1)
            query_pos_bevformer = query_pos_bevformer[None].expand(bs, -1, -1)
            query_bevformer = query_bevformer[None].expand(bs, -1, -1)
            reference_points_bevformer = self.positional_encoding_to_reference_point(query_pos_bevformer).sigmoid()

            # Encode time and locations
            time_embedding = self.time_mlp(t)            
            reference_points_diffusion = x_boxes.to(query_diffusion.dtype)
            query_pos_diffusion = self.query_pos_net(reference_points_diffusion)

            # Format both sets of queries and locations
            query = torch.cat((query_bevformer, query_diffusion), dim=-2) # (B, 2Q, C)
            query_pos = torch.cat((query_pos_bevformer, query_pos_diffusion), dim=-2) # (B, 2Q, C)
            reference_points = torch.cat((reference_points_bevformer, reference_points_diffusion), dim=-2) # (B, 2Q, C)
            init_reference_out = reference_points
            query = query.permute(1, 0, 2)
            query_pos = query_pos.permute(1, 0, 2)
            bev_embed = bev_embed.permute(1, 0, 2)

            inter_states, inter_references = self.decoder(
                query=query,
                key=None,
                value=bev_embed,
                query_pos=query_pos,
                reference_points=reference_points,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                level_start_index=torch.tensor([0], device=query.device),
                **kwargs)

            # Since the time embedding is only one and belongs to the decoder itself,
            # we take the computation out over here.
            n_diffusion_queries = query_diffusion.size(1)
            n_bevformer_queries = query_bevformer.size(1)
            scale_shift = self.block_time_mlp(time_embedding)
            scale_shift = torch.repeat_interleave(scale_shift, n_diffusion_queries, dim=0)[None]
            scale, shift = scale_shift.chunk(2, dim=-1) # 2 x (Q, B, C)
            inter_states2 = torch.empty_like(inter_states)
            inter_states2[:, :n_bevformer_queries] = inter_states[:, :n_bevformer_queries]
            inter_states2[:, -n_diffusion_queries:] = inter_states[:, -n_diffusion_queries:] * (scale + 1) + shift
            inter_states = inter_states2

            inter_references_out = inter_references
            return bev_embed, inter_states, init_reference_out, inter_references_out
       
    def prep_targets(self, gt_bboxes_3d, gt_classes, bev_params=None):
        """gt_bboxes_3d: list[torch.Tensor] - the GT annotations.
        Each Tensor should be in absolute Lidar coordinates, relative to the ego vehicle, with size [Ni, 9]
        where Ni is the number of annotations in that sample and each annotation has 9 coordinates (cx, cy, z, dx, dy, dz, angle, vx, vy).

        bev_params = tuple[...], a tuple of the BEV specification
        """
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        bev_h, bev_w, grid_length, pc_range = bev_params
        
        gt_bboxes_3d = [i.tensor.cuda() for i in gt_bboxes_3d]
        gt_classes = [i.cuda() for i in gt_classes]
    
        device = gt_bboxes_3d[0].device
        coord_minimum = torch.Tensor([pc_range[0], pc_range[1]]).to(device)
        coord_range = torch.Tensor([pc_range[3] - pc_range[0], pc_range[4] - pc_range[1]]).to(device)

        targets = [AnnotObject(gt_boxes=i, gt_classes=j) for (i, j) in zip(gt_bboxes_3d, gt_classes)]

        for targets_per_image in targets:
            target = {}
            # Extract center
            gt_centers = targets_per_image.gt_boxes[:, [0, 1]]
            # gt_centers = torch.cat((gt_centers, targets_per_image.gt_boxes[:, [2]] + 0.5 * targets_per_image.gt_boxes[:, [5]]), dim=1) # (cx, cy, cz)
            
            # Normalize to [0, 1]
            gt_centers_normalized = (gt_centers - coord_minimum) / coord_range

            # Add diffusion
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_centers_normalized)
    
            # gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            target["labels"] = targets_per_image.gt_classes.to(device)
            target["centers"] = gt_centers.to(device)
            target['centers_normalized'] = gt_centers_normalized.to(device)
            new_targets.append(target)
        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)

    @torch.no_grad()
    def ddim_sample(self, backbone_feats, object_query_embeds_bevformer, object_query_embeds_diffusion, reg_branches=None, cls_branches=None, clip_denoised=True, **kwargs):
        """ Inference
        backbone_feats: torch.Tensor, multi-level features from the encoder
        clip_denoised: bool, whether to clip the denoised outputs
        """
        batch = backbone_feats.size(0)
        shape = (batch, self.num_proposals, 2)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        device = backbone_feats[0].device

        img = torch.randn(shape, device=device) # N(0, 1)
        ensemble_score, ensemble_label, ensemble_coord, outputs_coords_all, outputs_class_all = [], [], [], [], []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            # Get the predictions from the decoder
            preds, outputs_class, outputs_coord = self.model_predictions(backbone_feats, img, 
                object_query_embeds_bevformer, object_query_embeds_diffusion, time_cond, reg_branches, cls_branches, self_cond, clip_x_start=clip_denoised, **kwargs)

            if self.ddim_query_type == 'bevformer':
                # Do single NMS and return results
                boxes = self.prediction_to_box(outputs_coord)[-1, 0]
                scores, idxs = outputs_class[-1, 0, :, :].max(dim=1)
                keep = batched_nms_rotated(boxes=boxes, scores=scores, idxs=idxs, iou_threshold=self.NMS_THRESHOLD)
                selected_logits, selected_classes = outputs_class[-1, 0, keep, :].max(1) # (N, C)
                outputs_coord, outputs_class = self.average_boxes_in_radius(outputs_coord[-1, 0, keep, :], selected_logits, 
                outputs_class[-1, 0, keep, :], selected_classes, radius=self.radial_suppression_radius)
                return outputs_class[None], outputs_coord[None]

            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            # Filter using box renewal
            if self.box_renewal:
                score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
                
                if self.ddim_query_type == 'both':
                    n_bevformer_queries = object_query_embeds_bevformer.size(0)
                else:
                    n_bevformer_queries = 0
                
                score_per_image = score_per_image[n_bevformer_queries:]
                box_per_image = box_per_image[n_bevformer_queries:]
                threshold = self.box_renewal_threshold
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)
                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]
            if time_next < 0:
                img = x_start
                if self.use_ensemble and self.sampling_timesteps > 1:
                    outputs_coords_all.append(outputs_coord)
                    outputs_class_all.append(outputs_class)
                    bev_boxes = self.prediction_to_box(outputs_coord)[-1]
                    sigmoid_cls = torch.sigmoid(outputs_class[-1])
                    labels = torch.arange(self.num_classes, device=bev_boxes.device). \
                        unsqueeze(0).repeat(sigmoid_cls.size(1), 1).flatten(0, 1)
                    for i, (scores_per_image, box_pred_per_image) in enumerate(zip(sigmoid_cls, bev_boxes)):
                        scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                        labels_per_image = labels[topk_indices]
                        box_pred_per_image = box_pred_per_image.view(-1, 1, 5).repeat(1, sigmoid_cls.size(1), 1).view(-1, 5)
                        box_pred_per_image = box_pred_per_image[topk_indices]
                        ensemble_coord.append(box_pred_per_image)
                        ensemble_label.append(labels_per_image)
                        ensemble_score.append(scores_per_image)
                continue

            # Compute the next step
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img_old = img.clone()
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if self.box_renewal:  # filter
                img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 2, device=img.device)), dim=1)

            # Store DDIM outputs
            if self.use_ensemble and self.sampling_timesteps > 1:
                outputs_coords_all.append(outputs_coord)
                outputs_class_all.append(outputs_class)
                bev_boxes = self.prediction_to_box(outputs_coord)[-1]
                sigmoid_cls = torch.sigmoid(outputs_class[-1])
                labels = torch.arange(self.num_classes, device=bev_boxes.device). \
                    unsqueeze(0).repeat(sigmoid_cls.size(1), 1).flatten(0, 1)
                for i, (scores_per_image, box_pred_per_image) in enumerate(zip(sigmoid_cls, bev_boxes)):
                    scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(sigmoid_cls.size(1), sorted=False)
                    labels_per_image = labels[topk_indices]
                    box_pred_per_image = box_pred_per_image.view(-1, 1, 5).repeat(1, self.num_classes, 1).view(-1, 5)
                    box_pred_per_image = box_pred_per_image[topk_indices]
                    ensemble_coord.append(box_pred_per_image)
                    ensemble_label.append(labels_per_image)
                    ensemble_score.append(scores_per_image)
                
        # Ensemble NMS
        if self.use_ensemble and self.sampling_timesteps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)
            outputs_coords_all = torch.cat(outputs_coords_all, dim=2)
            outputs_class_all = torch.cat(outputs_class_all, dim=2)

            if self.use_nms:
                keep = batched_nms_rotated(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                outputs_coord = outputs_coords_all[:, :, keep, :]
                outputs_class = outputs_class_all[:, :, keep, :]
          
        # NMS - Non-maximum suppression
        boxes = self.prediction_to_box(outputs_coord)[-1, 0]
        scores, idxs = outputs_class[-1, 0, :, :].max(dim=1)
        keep = batched_nms_rotated(boxes=boxes, scores=scores, idxs=idxs, iou_threshold=self.NMS_THRESHOLD)
        
        # Finally, do radial suppression
        selected_logits, selected_classes = outputs_class[-1, 0, keep, :].max(1) # (N, C)
        outputs_coord, outputs_class = self.average_boxes_in_radius(outputs_coord[-1, 0, keep, :], selected_logits, 
                outputs_class[-1, 0, keep, :], selected_classes, radius=self.radial_suppression_radius)

        return outputs_class[None], outputs_coord[None]

    def prediction_to_box(self, preds):
        """Prediction to BEV projection
        Preds is Tensor[..., 10] with (cx, cy, log(w), log(h), cz, log(l), sin, cos, vx, cy)"""
        boxes = torch.cat((
            preds[..., [0, 1]], # Center (cx, cy)
            preds[..., [2, 3]].exp(), # (w, h)
            torch.rad2deg(torch.atan2(preds[..., [6]], preds[..., [7]])) # angle in degrees
        ), dim=-1)
        return boxes

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats, x, object_query_embeds_bevformer, object_query_embeds_diffusion, t, 
            reg_branches=None, cls_branches=None, x_self_cond=None, clip_x_start=False, **kwargs):
        """
        Prepare and call the decoder (diffusion head)
        backbone_feats: torch.Tensor, multilevel features, expected single level with shape [torch.Tensor[BCHW]]
        x: torch.Tensor of shape (B, N, 3), rotated boxes, coming from N(0, 1), (cx, cy, cz) semantics
        object_query: torch.Tensor (Q, C)
        """
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2 # This gets mapped to [0, 1]
        pc_range = kwargs['pc_range']
        bev_h = kwargs['bev_h']
        bev_w = kwargs['bev_w']

        x_boxes[..., 1] = 1 - x_boxes[..., 1]
        bs = x_boxes.size(0)

        # Handle the BEVFormer queries
        if self.ddim_query_type in ['both', 'bevformer']:
            query_pos_bevformer, query_bevformer = torch.split(object_query_embeds_bevformer, self.embed_dims, dim=1)
            query_pos_bevformer = query_pos_bevformer[None].expand(bs, -1, -1)
            query_bevformer = query_bevformer[None].expand(bs, -1, -1)
            reference_points_bevformer = self.positional_encoding_to_reference_point(query_pos_bevformer).sigmoid()

        # Handle diffusion queries
        if object_query_embeds_diffusion is None:
            query_diffusion = torch.nn.functional.grid_sample(backbone_feats.view(bs, bev_h, bev_w, self.embed_dims).permute(0, 3, 1, 2), 
                2 * x_boxes[:, None, :, :].float() - 1, mode='bilinear', align_corners=False) # (B, C, 1, Q)
            query_diffusion = query_diffusion.squeeze(2).permute(0, 2, 1).detach() # (B, Q, C)
        else:
            query_diffusion = torch.nn.functional.grid_sample(object_query_embeds_diffusion[None].repeat(bs, 1, 1).permute(0, 2, 1).view(bs, self.embed_dims, 30, 30),
                2 * x_boxes[:, None, :, :].float() - 1, mode='bilinear', align_corners=False) # (B, C, 1, Q)
            query_diffusion = query_diffusion.squeeze(2).permute(0, 2, 1) # (B, Q, C)

        # Encode time and locations
        if len(t.size()) == 1:
            t = t.view(1, 1)
        time_embedding = self.time_mlp(t)            
        reference_points_diffusion = x_boxes.to(query_diffusion.dtype)
        query_pos_diffusion = self.query_pos_net(reference_points_diffusion)

        # Format both sets of queries and locations
        if self.ddim_query_type == 'both':
            query = torch.cat((query_bevformer, query_diffusion), dim=-2) # (B, 2Q, C)
            query_pos = torch.cat((query_pos_bevformer, query_pos_diffusion), dim=-2) # (B, 2Q, C)
            reference_points = torch.cat((reference_points_bevformer, reference_points_diffusion), dim=-2) # (B, 2Q, C)
        elif self.ddim_query_type == 'diffusion':
            query = query_diffusion # (B, Q, C)
            query_pos = query_pos_diffusion # (B, Q, C)
            reference_points = reference_points_diffusion # (B, Q, C)
        else:
            query = query_bevformer
            query_pos = query_pos_bevformer
            reference_points = reference_points_bevformer

        init_reference_out = reference_points
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        backbone_feats = backbone_feats.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=backbone_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            time_embedding=time_embedding,
            **kwargs)

        if self.ddim_query_type == 'both':
            n_diffusion_queries = query_diffusion.size(1)
            n_bevformer_queries = query_bevformer.size(1)
            scale_shift = self.block_time_mlp(time_embedding)
            scale_shift = torch.repeat_interleave(scale_shift, n_diffusion_queries, dim=0)[None]
            scale, shift = scale_shift.chunk(2, dim=-1) # 2 x (Q, B, C)
            inter_states2 = torch.empty_like(inter_states)
            inter_states2[:, :n_bevformer_queries] = inter_states[:, :n_bevformer_queries]
            inter_states2[:, -n_diffusion_queries:] = inter_states[:, -n_diffusion_queries:] * (scale + 1) + shift
            inter_states = inter_states2
        elif self.ddim_query_type == 'diffusion':
            n_diffusion_queries = query_diffusion.size(1)
            scale_shift = self.block_time_mlp(time_embedding)
            scale_shift = torch.repeat_interleave(scale_shift, n_diffusion_queries, dim=0)[None]
            scale, shift = scale_shift.chunk(2, dim=-1) # 2 x (Q, B, C)
            inter_states = inter_states * (scale + 1) + shift
        else:
            pass

        inter_references_out = inter_references
        bev_embed, hs, init_reference, inter_references = backbone_feats, inter_states, init_reference_out, inter_references_out
        
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            outputs_class = cls_branches[lvl](hs[lvl])
            tmp = reg_branches[lvl](hs[lvl])
            assert reference.shape[-1] == 2
            tmp[..., 0:2] = (tmp[..., 0:2] + inverse_sigmoid(reference)).sigmoid()

            # Scale to output size
            if tmp.size(-1) > 10:
                tmp[..., [0, -6, -4, -2]] = (tmp[..., [0, -6, -4, -2]] * (pc_range[3] - pc_range[0]) + pc_range[0])
                tmp[..., [1, -5, -3, -1]] = (tmp[..., [1, -5, -3, -1]] * (pc_range[4] - pc_range[1]) + pc_range[1])
            else:
                tmp[..., [0]] = (tmp[..., [0]] * (pc_range[3] - pc_range[0]) + pc_range[0])
                tmp[..., [1]] = (tmp[..., [1]] * (pc_range[4] - pc_range[1]) + pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5].sigmoid() * (pc_range[5] - pc_range[2]) + pc_range[2])

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        if self.ddim_query_type == 'bevformer':
            return None, outputs_classes, outputs_coords

        # Now that we have the predictions, calculate diffusion quantities
        x_start = outputs_coords[-1, :, -n_diffusion_queries:, [0, 1]]  # (batch, num_proposals, 3) predict centers: absolute coordinates (cx, cy, cz)

        device = x_start.device
        coord_minimum = torch.Tensor([pc_range[0], pc_range[1]]).to(device)
        coord_range = torch.Tensor([pc_range[3] - pc_range[0], pc_range[4] - pc_range[1]]).to(device)
        x_start = (x_start - coord_minimum) / coord_range # (cx, cy) in [0, 1]

        x_start = (x_start * 2 - 1.) * self.scale # (-S, S)
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        t = t.squeeze(-1)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start), outputs_classes, outputs_coords

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: Tensor (N, 5). Elements are (cx, cy, cz), in [0, 1]
        :param num_proposals:

        Returns:
        :diff_boxes in (cx, cy, w, h, yaw)
        """
        device = gt_boxes.device
        t = torch.randint(0, self.num_timesteps, (1,), device=device).long()
        noise = torch.randn(self.num_proposals, 2, device=device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5]], dtype=torch.float, device=device)
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 2,
                                          device=device) / 6. + 0.5  # N(mu=1/2, sigma=1/6)
            # box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=0.)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = (x_start * 2. - 1.) * self.scale # This is applied also to yaw, because diffusion affects yaw
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.
        diff_boxes = x
        return diff_boxes, noise, t

    def average_boxes_in_radius(self, boxes, logits, logits_full, classes, radius=1.0):
        """
        Similar to NMS but we consider center-distances only (instead of IOUs) and we average the boxes according to confidence.

        boxes: torch.Tensor (B, 10)
        logits: torch.Tensor (B,)
        logits_full: torch.Tensor (B, K)
        classes = torch.Tensor (B,)
        """
        new_boxes = []
        new_logits_full = []
        for class_id in classes.unique():
            # Get those boxes which correspond to this class
            ids_for_that_class = torch.where(classes == class_id)[0]
            boxes_for_that_class = boxes[ids_for_that_class]
            logits_for_that_class = logits[ids_for_that_class].sigmoid()
            logits_full_for_that_class = logits_full[ids_for_that_class]
            
            # Sort boxes by descending confidence
            order = torch.argsort(logits_for_that_class, descending=True)
            keep = torch.ones_like(order, dtype=torch.bool)

            # Calculate pairwise distance
            pairwise_dist = torch.cdist(boxes_for_that_class[..., :2], boxes_for_that_class[..., :2], p=1)

            for i in order:
                if keep[i] == True:
                    indices = torch.nonzero(pairwise_dist[i] < radius)
                    if len(indices.size()) > 1:
                        indices = indices.squeeze()
                    if len(indices.size()) < 1:
                        continue
                    average_box =  (logits_for_that_class[indices][:, None] * boxes_for_that_class[indices]).sum(0) / logits_for_that_class[indices].sum()
                    average_conf =  (logits_for_that_class[indices][:, None] * logits_full_for_that_class[indices]).sum(0) / logits_for_that_class[indices].sum()

                    keep[indices] = False
                    keep[i] = True
                    boxes_for_that_class[i] = average_box
                    logits_full_for_that_class[i] = average_conf
                
            new_boxes.append(boxes_for_that_class[keep])
            new_logits_full.append(logits_full_for_that_class[keep])
        return torch.cat(new_boxes, dim=0), torch.cat(new_logits_full, dim=0)
