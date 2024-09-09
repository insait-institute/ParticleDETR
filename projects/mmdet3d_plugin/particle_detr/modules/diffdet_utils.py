# ========================================
# From the DiffusionDet paper
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import RotatedBoxes

from mmcv.cnn.bricks.registry import (CONV_LAYERS, ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


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


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)

# From here https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
class NeRFPositionalEncoding:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=5):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = NeRFPositionalEncoding(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim



@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DynamicHead(nn.Module):

    def __init__(self,
        ROI_CFG: dict,
        RCNN_CFG: dict,
        NUM_CLASSES = 10,
        NHEADS = 8, # In multi-head attention
        DROPOUT = 0.0,
        DIM_FEEDFORWARD = 2048,
        ACTIVATION = 'relu',
        HIDDEN_DIM = 256,
        NUM_HEADS = 6,
        DEEP_SUPERVISION = True,
        USE_FOCAL_LOSS = True,
        USE_FED_LOSS = False,
        FOCAL_LOSS_PRIOR_PROB = 0.01,
        ):

        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(ROI_CFG)
        self.box_pooler = box_pooler
        
        # Build heads.
        num_classes = NUM_CLASSES
        d_model = HIDDEN_DIM
        dim_feedforward = DIM_FEEDFORWARD
        nhead = NHEADS
        dropout = DROPOUT
        activation = ACTIVATION
        num_heads = NUM_HEADS
        rcnn_head = RCNNHead(RCNN_CFG, d_model, num_classes, dim_feedforward, nhead, dropout, activation)
        self.head_series = _get_clones(rcnn_head, num_heads)
        self.num_heads = num_heads
        self.num_layers = num_heads # Alias, for the upstream classes, because BEVFormerHead looks for num_layers
        self.return_intermediate = DEEP_SUPERVISION

        # Gaussian random feature embedding layer for time
        self.d_model = d_model
        time_dim = d_model * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Init parameters.
        self.use_focal = USE_FOCAL_LOSS
        self.use_fed_loss = USE_FED_LOSS
        self.num_classes = num_classes
        if self.use_focal or self.use_fed_loss:
            prior_prob = FOCAL_LOSS_PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss and fed loss.
            if self.use_focal or self.use_fed_loss:
                if p.shape[-1] == self.num_classes or p.shape[-1] == self.num_classes + 1:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(roi_cfg, input_features=None):
        # in_features = roi_cfg.IN_FEATURES
        pooler_resolution = roi_cfg.POOLER_RESOLUTION
        # pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = roi_cfg.POOLER_SAMPLING_RATIO
        pooler_type = roi_cfg.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        # in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        # assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=[1.0], # No scaling because in BEV we don't have multi-level features
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, init_bboxes, t, init_features):
        """
        IMPORTANT: init_bboxes here are in BEV grid coordinates, not in Lidar (real) coords."""
        # assert t shape (batch_size)
        time = self.time_mlp(t)

        inter_class_logits = []
        inter_pred_bboxes = [] # 3D predictions
        inter_pred_bboxes_bev = [] # BEV predictions

        bs = len(features[0])
        bboxes = init_bboxes
        num_boxes = bboxes.shape[1]

        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None
        
        for head_idx, rcnn_head in enumerate(self.head_series):
            class_logits, pred_bboxes, bbox_preds_3d, proposal_features = rcnn_head(features, bboxes, proposal_features, self.box_pooler, time)
            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(bbox_preds_3d)
                inter_pred_bboxes_bev.append(pred_bboxes)
            bboxes = pred_bboxes.detach() # Take only the BEV projection

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes), torch.stack(inter_pred_bboxes_bev)

        return class_logits[None], pred_bboxes[None], inter_pred_bboxes_bev[None]


class RCNNHead(nn.Module):

    def __init__(self, rcnn_cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(rcnn_cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # block time mlp
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model * 4, d_model * 2))

        # # positional coordinate for boxes
        # self.embed_fn, self.pos_enc_dim = get_embedder(multires=6, i=0, input_dims=5)

        # cls.
        num_cls = rcnn_cfg.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = rcnn_cfg.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)
        
        # pred.
        self.use_focal = rcnn_cfg.USE_FOCAL_LOSS
        self.use_fed_loss = rcnn_cfg.USE_FED_LOSS
        if self.use_focal or self.use_fed_loss:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 5) # The refinement happens in BEV
        self.bboxes_preds_3d = nn.Linear(d_model, 10) # Output for NuScenes
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pro_features, pooler, time_emb):
        """
        :param bboxes: (N, nr_boxes, 5) - last dimension is 5 because it contains the rotation, in degrees
        :param pro_features: (N, nr_boxes, d_model)

        Returns:
        logits (N, nr_boxes, K)
        pred_boxes_bev (N, nr_boxes, 5), with the format (cx, cy, dx, dy, theta), theta in degrees
        bbox_preds_3d (N, nr_boxes, 10), with the format (cx, cy, dx, dy, cz, dz, sin, cos, vx, vy)
        obj_features: features

        """
        N, nr_boxes = bboxes.shape[:2]

        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(RotatedBoxes(bboxes[b]))
        roi_features = pooler(features, proposal_boxes)

        if pro_features is None:
            pro_features = roi_features.view(N, nr_boxes, self.d_model, -1).mean(-1)

        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)

        # self_att.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)
        
        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)

        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift # (BP, C=256)

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()

        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature) # (BP, C)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature) # (BP, C)
        class_logits = self.class_logits(cls_feature) # (BP, K)

        # Get the final prediction - 3D coordinates
        coords_and_deltas = self.bboxes_preds_3d(reg_feature) # (BN, 10), real

        # Apply the deltas
        bbox_preds_3d = self.apply_deltas_custom(coords_and_deltas, bboxes.view(-1, 5))
        
        # Get the BEV of the predicted 3D coords
        pred_bboxes_bev = torch.cat((bbox_preds_3d[:, [0, 1, 2, 3]], 
            torch.rad2deg(torch.atan2(bbox_preds_3d[:, [6]], bbox_preds_3d[:, [7]]))), dim=1)
    
        return (class_logits.view(N, nr_boxes, -1), 
            pred_bboxes_bev.view(N, nr_boxes, -1), bbox_preds_3d.view(N, nr_boxes, -1), obj_features)

    def apply_deltas_with_rotation(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh, dtheta) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*10), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 5)
        
        boxes are represented as (cx, cy, dx, dy, angle)
        All of these are absolute coordinates. cx, cy is in [0, 200]
        [dx, dy] is in [0, 200]
        angle is in [-180, 180]
        """
        boxes = boxes.to(deltas.dtype)
        ctr_x = boxes[:, 0]
        ctr_y = boxes[:, 1]
        widths = boxes[:, 2]
        heights = boxes[:, 3]
        angles = boxes[:, 4]

        wx, wy, ww, wh, wtheta = self.bbox_weights
        dx = deltas[:, 0::5] / wx
        dy = deltas[:, 1::5] / wy
        dw = deltas[:, 2::5] / ww
        dh = deltas[:, 3::5] / wh
        dtheta = deltas[:, 4::5] / wtheta

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)
        dtheta = torch.clamp(dtheta, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        pred_theta = torch.exp(dtheta) * angles[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::5] = pred_ctr_x # xc
        pred_boxes[:, 1::5] = pred_ctr_y # yc
        pred_boxes[:, 2::5] = pred_w  # w
        pred_boxes[:, 3::5] = pred_h  # h
        pred_boxes[:, 4::5] = pred_theta  # theta

        return pred_boxes

    def apply_deltas_custom(self, deltas, boxes):
        """
        Apply transformation `deltas`.
         deltas is a 10-D vector of reals (Dcx, Dcy, Ddx, Ddy, cz, dz, Dsin, Dcos, vx, vy) where D stands for Delta.
         boxes is a 5-D vector of reals (cx, cy, dx, dy, theta), theta in degrees. Here the coordinates are in BEV grid coords.

        
        This function takes boxes in BEV coords and deltas in arbitrary units of measurement.
        It updates the boxes using the deltas, producing updated boxes in BEV coords. 

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*10), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 5)
        
        boxes are represented as (cx, cy, dx, dy, angle)
        All of these are absolute coordinates. cx, cy is in [0, bev_size]
        [dx, dy] is in [0, bev_size]
        angle is in [-180, 180]
        """
        boxes = boxes.to(deltas.dtype)
        ctr_x = boxes[:, 0]
        ctr_y = boxes[:, 1]
        widths = boxes[:, 2]
        heights = boxes[:, 3]
        angles = torch.deg2rad(boxes[:, 4]) 

        wx, wy, ww, wh, wtheta = self.bbox_weights
        dx = deltas[:, 0::10] / wx
        dy = deltas[:, 1::10] / wy
        dw = deltas[:, 2::10] / ww
        dh = deltas[:, 3::10] / wh
        cz = deltas[:, 4::10]
        dz = deltas[:, 5::10]
        dsin = deltas[:, 6::10] # Real
        dcos = deltas[:, 7::10] # Real
        vx = deltas[:, 8::10]
        vy = deltas[:, 9::10]

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        # The exponential scaling should happen wrt widths/heights that always stay horizontal/vertical
        real_w = widths * angles.cos() + heights * angles.sin()
        real_h = widths * angles.sin() + heights * angles.cos()
        pred_ctr_x = dx * real_w[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * real_h[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        pred_theta = angles[:, None] + torch.atan2(dsin, dcos) # in rad

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::10] = pred_ctr_x # cx, real coord in bev
        pred_boxes[:, 1::10] = pred_ctr_y # cy, real coord in bev
        pred_boxes[:, 2::10] = pred_w  # w, real coord in bev
        pred_boxes[:, 3::10] = pred_h  # h, real coord in bev
        pred_boxes[:, 4::10] = cz  # cz
        pred_boxes[:, 5::10] = dz  # dz
        pred_boxes[:, 6::10] = pred_theta.sin()  # sin
        pred_boxes[:, 7::10] = pred_theta.cos()  # cos
        pred_boxes[:, 8::10] = vx  # vx
        pred_boxes[:, 9::10] = vy  # vy
        return pred_boxes

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*5), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 5)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.hidden_dim = cfg.HIDDEN_DIM
        self.dim_dynamic = cfg.DIM_DYNAMIC
        self.num_dynamic = cfg.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
