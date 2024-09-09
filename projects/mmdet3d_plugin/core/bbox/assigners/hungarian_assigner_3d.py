import torch

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.models.utils.transformer import inverse_sigmoid
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.ops.box_iou_rotated import box_iou_rotated

from projects.mmdet3d_plugin.particle_detr.sim_ota_loss import HungarianMatcherDynamicK

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def prediction_to_box(preds):
    """Prediction to BEV projection
    Preds is Tensor[..., 10] with (cx, cy, log(w), log(h), cz, log(l), sin, cos, vx, cy)"""
    boxes = torch.cat((
        preds[..., [0, 1]], # Center (cx, cy)
        preds[..., [2, 3]].exp(), # (w, h)
        torch.atan2(preds[..., [6]], preds[..., [7]]) # angle in degrees
    ), dim=-1)
    return boxes


@BBOX_ASSIGNERS.register_module()
class HungarianAssigner3D(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.
    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 pc_range=None):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)

        self.use_sim_ota = True
        if self.use_sim_ota:
            self.dynamic_k_matcher = HungarianMatcherDynamicK()
        self.pc_range = pc_range

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes, 
               gt_labels,
               gt_bboxes_ignore=None,
               eps=1e-7,
               matching_type='1-1'):
        """Computes one-to-one matching based on the weighted costs.
        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
       
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
        reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])

        if matching_type == 'n-1':
            ious = box_iou_rotated(prediction_to_box(bbox_pred), prediction_to_box(normalized_gt_bboxes), mode='iou', aligned=False)
            #loss_iou = (1 - ious).log()
            cost = cls_cost + reg_cost #+ loss_iou

            # Modify the cost matrix based on foreground information
            #is_foreground = self.get_foreground_info(bbox_pred, normalized_gt_bboxes)
            #cost[~is_foreground] = cost[~is_foreground] + 10000.0
            (selected_queries, gt_indices), _ = self.dynamic_k_matcher.dynamic_k_matching(cost, ious, num_gt=normalized_gt_bboxes.size(0))
            
            assigned_gt_inds[:] = 0
            assigned_gt_inds[selected_queries] = gt_indices + 1
            assigned_labels[selected_queries] = gt_labels[gt_indices]
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        elif matching_type == '1-1':
            cost = cls_cost + reg_cost

            # 3. do Hungarian matching on CPU using linear_sum_assignment
            cost = cost.detach().cpu()
            if linear_sum_assignment is None:
                raise ImportError('Please run "pip install scipy" '
                                  'to install scipy first.')
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
            matched_row_inds = torch.from_numpy(matched_row_inds).to(
                bbox_pred.device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(
                bbox_pred.device)

            # 4. assign backgrounds and foregrounds
            # assign all indices to backgrounds first
            assigned_gt_inds[:] = 0
            # assign foregrounds based on matching results
            assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
            assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        else:
            print("Invalid matching type")
            assert False
    
    def get_foreground_info(self, box_preds, box_targets):
        """
        Find whether pred i is sufficiently close to target j.
        Sufficiently close here means that the center of box i is within some distance of the center of box j.
        We return a boolean mask of size (N,) where N is the number of predicted boxes.
        """

        pred_center_x = box_preds[:, [0]] # (N, 1)
        pred_center_y = box_preds[:, [1]] # (N, 1)
        target_center_x = box_targets[:, 0] # (M,)
        target_center_y = box_targets[:, 1] # (M,)
        center_radius = 4

        b_x = (pred_center_x[:, None] > (target_center_x - center_radius)[None, :, None]).squeeze(-1)
        t_x = (pred_center_x[:, None] < (target_center_x + center_radius)[None, :, None]).squeeze(-1)
        b_y = (pred_center_y[:, None] > (target_center_y - center_radius)[None, :, None]).squeeze(-1)
        t_y = (pred_center_y[:, None] < (target_center_y + center_radius)[None, :, None]).squeeze(-1)
        is_in_centers = ((b_x.long() + t_x.long() + b_y.long() + t_y.long()) == 4) # (N, M)
        is_in_centers_all = is_in_centers.sum(1) > 0 # (N,)
        return is_in_centers_all