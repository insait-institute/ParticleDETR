import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor



@HEADS.register_module()
class ParticleDETRHead(DETRHead):
    """
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.num_bevformer_queries = kwargs.pop('num_bevformer_queries')
        self.num_regularly_placed_queries = kwargs.pop('num_diffusion_queries')

        super(ParticleDETRHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head.
        We have copies for each decoder hidden layer, providing auxiliary losses.
        """
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding_bevformer = nn.Embedding(self.num_query, 2 * self.embed_dims)
            self.query_embedding_diffusion = nn.Embedding(self.num_query, self.embed_dims)


    def init_weights(self):
        """Initialize weights of the ParticleDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=False, 
        gt_bboxes_3d = None, gt_labels_3d = None):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: (list[dict]): metadata for each sample - images, shapes, paddings, lidar data, etc.
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """       
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds_bevformer = self.query_embedding_bevformer.weight.to(dtype)
        object_query_embeds_diffusion = self.query_embedding_diffusion.weight.to(dtype)

        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:  # only use encoder to obtain BEV features
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats=mlvl_feats,
                bev_queries=bev_queries,
                object_query_embeds_bevformer=object_query_embeds_bevformer,
                object_query_embeds_diffusion=object_query_embeds_diffusion,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,
                cls_branches=self.cls_branches if self.with_box_refine else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
                gt_bboxes_3d = gt_bboxes_3d,
                gt_labels_3d = gt_labels_3d,
                pc_range = self.pc_range,
        )
        
        if (gt_bboxes_3d is not None) and (gt_labels_3d is not None):
            bev_embed, hs, init_reference, inter_references = outputs
            hs = hs.permute(0, 2, 1, 3)
            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]

                outputs_class = self.cls_branches[lvl](hs[lvl])
                tmp = self.reg_branches[lvl](hs[lvl])
                assert reference.shape[-1] == 2
                tmp[..., 0:2] = (tmp[..., 0:2] + inverse_sigmoid(reference)).sigmoid()

                # Scale to output size
                if self.code_size > 10:
                    tmp[..., [0, -6, -4, -2]] = (tmp[..., [0, -6, -4, -2]] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
                    tmp[..., [1, -5, -3, -1]] = (tmp[..., [1, -5, -3, -1]] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
                else:
                    tmp[..., [0]] = (tmp[..., [0]] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
                    tmp[..., [1]] = (tmp[..., [1]] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
                tmp[..., 4:5] = (tmp[..., 4:5].sigmoid() * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

                outputs_coord = tmp
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)

            outputs_classes = torch.stack(outputs_classes)
            outputs_coords = torch.stack(outputs_coords)
        else:
            bev_embed, outputs_classes, outputs_coords = outputs

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None,
                           matching_type='1-1'):
        """Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Uses the Hungarian algorithm for matching.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore, matching_type=matching_type)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None,
                    matching_type='1-1'):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 10].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 9) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list, matching_type=matching_type)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None,
                    matching_type='1-1'):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 10].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 9) format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        assert matching_type in ['1-1', 'n-1']
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list, matching_type=matching_type)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)
                
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox 

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only passed when as_two_stage is
                    True, otherwise None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only
                    passed when as_two_stage is True, otherwise None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]
    
        num_boxes = all_bbox_preds.size(2)
        desired_num_boxes = 100
        if desired_num_boxes > 0:
            gt_bboxes_list_repeated = [i.repeat(desired_num_boxes // i.size(0) + 1, 1)[:desired_num_boxes] \
                          if desired_num_boxes > i.size(0) else i for i in gt_bboxes_list]
            gt_labels_list_repeated = [i.repeat(desired_num_boxes // i.size(0) + 1)[:desired_num_boxes] \
                          if desired_num_boxes > i.size(0) else i for i in gt_labels_list]
            
            all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
            all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
            all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

            all_gt_bboxes_list_repeated = [gt_bboxes_list_repeated for _ in range(num_dec_layers)]
            all_gt_labels_list_repeated = [gt_labels_list_repeated for _ in range(num_dec_layers)]
            all_gt_bboxes_ignore_list_repeated = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        else:
            all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
            all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
            all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
            all_gt_bboxes_list_repeated = all_gt_bboxes_list
            all_gt_labels_list_repeated = all_gt_labels_list
            all_gt_bboxes_ignore_list_repeated = all_gt_bboxes_ignore_list

        # BEVFormer loss
        num_queries = all_cls_scores.size(-2)
        num_bevformer_queries = num_queries // 2
        losses_cls_bevformer, losses_bbox_bevformer = multi_apply(
            self.loss_single, all_cls_scores[:, :, :num_bevformer_queries], all_bbox_preds[:, :, :num_bevformer_queries],
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list, matching_type='1-1')

        # Diffusion loss
        if desired_num_boxes > 0:
            losses_cls_diffusion, losses_bbox_diffusion = multi_apply(
                self.loss_single, all_cls_scores[:, :, num_bevformer_queries:], all_bbox_preds[:, :, num_bevformer_queries:],
                all_gt_bboxes_list_repeated, all_gt_labels_list_repeated,
                all_gt_bboxes_ignore_list_repeated, matching_type='1-1')
        else:
            losses_cls_diffusion, losses_bbox_diffusion = multi_apply(
                self.loss_single, all_cls_scores[:, :, num_bevformer_queries:], all_bbox_preds[:, :, num_bevformer_queries:],
                all_gt_bboxes_list_repeated, all_gt_labels_list_repeated,
                all_gt_bboxes_ignore_list_repeated, matching_type='n-1')            

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls_bevformer, enc_losses_bbox_bevformer = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)

            enc_loss_cls_diffusion, enc_losses_bbox_diffusion = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)

            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
        
        # Aggregate losses across BEVFormer and diffusion
        losses_cls = [0.5 * (i + j) for (i, j) in zip(losses_cls_bevformer, losses_cls_diffusion)]
        losses_bbox = [0.5 * (i + j) for (i, j) in zip(losses_bbox_bevformer, losses_bbox_diffusion)]

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list