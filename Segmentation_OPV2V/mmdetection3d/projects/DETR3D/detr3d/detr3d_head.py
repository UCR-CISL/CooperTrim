import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Linear
#from mmdet.models.dense_heads import DETRHead
from opv2v.opencood.models.heads.detr_head import DETRHead_old
from mmdet.models.layers import inverse_sigmoid
from mmdet.models.utils import multi_apply
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from mmengine.model import bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor
import inspect

from mmdet3d.registry import MODELS, TASK_UTILS
from .util import normalize_bbox


@MODELS.register_module()
class DETR3DHead(DETRHead_old):
    """Head of DETR3D.

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bbox_coder (obj:`ConfigDict`): Configs to build the bbox coder
        num_cls_fcs (int) : the number of layers in cls and reg branch
        code_weights (List[double]) : loss weights of
            (cx,cy,l,w,cz,h,sin(φ),cos(φ),v_x,v_y)
        code_size (int) : size of code_weights
    """

    def __init__(
            self,
            num_query=900,
            num_classes=10,
            in_channels=256,
            sync_cls_avg_factor=True,
            with_box_refine=False,
            as_two_stage=False,
            transformer=None,
            bbox_coder=None,
            num_cls_fcs=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            code_size=8,
            **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.code_size = code_size
        self.code_weights = code_weights

        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.num_cls_fcs = num_cls_fcs - 1
        super(DETR3DHead, self).__init__(
                 num_classes=num_classes,
                 in_channels=in_channels,
                 num_query=num_query,
                 num_reg_fcs=num_cls_fcs,
                 sync_cls_avg_factor=sync_cls_avg_factor,
            transformer=transformer, **kwargs)
        # DETR sampling=False, so use PseudoSampler, format the result
        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = TASK_UTILS.build(sampler_cfg)

        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False),
            requires_grad=False)

    # forward_train -> loss
    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
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
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, mlvl_feats: List[Tensor], img_metas: List[Dict],
                **kwargs) -> Dict[str, Tensor]:
        """Forward function.

        Args:
            mlvl_feats (List[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format
                (cx, cy, l, w, cz, h, sin(φ), cos(φ), vx, vy).
                Shape [nb_dec, bs, num_query, 10].
        """
        query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references = self.transformer(
            mlvl_feats,
            query_embeds,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            img_metas=img_metas,
            **kwargs)
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])  # shape: ([B, num_q, 10])
            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            tmp[..., 0:1] = \
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) \
                + self.pc_range[0]
            tmp[..., 1:2] = \
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) \
                + self.pc_range[1]
            tmp[..., 4:5] = \
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) \
                + self.pc_range[2]

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        return outs

    def _get_target_single(
            self,
            cls_score: Tensor,  # [query, num_cls]
            bbox_pred: Tensor,  # [query, 10]
            gt_instances_3d: InstanceList) -> Tuple[Tensor, ...]:
        """Compute regression and classification targets for a single image."""
        # turn bottm center into gravity center
        gt_bboxes = gt_instances_3d#.bboxes_3d  # [num_gt, 9]
        
        #gt_bboxes = torch.cat(
        #    (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)

        #gt_labels = gt_instances_3d.labels_3d  # [num_gt, num_cls]
        gt_labels = torch.ones(len(gt_bboxes), device=gt_bboxes[0].device, dtype=torch.long)

        # assigner and sampler: PseudoSampler   
        assign_result = self.assigner.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore=None)

        sampling_result = self.sampler.sample(
            assign_result, InstanceData(priors=bbox_pred),
            InstanceData(bboxes_3d=gt_bboxes))
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        num_bboxes = bbox_pred[0].size(0)
        # labels = gt_bboxes.new_full((num_bboxes, ),
        #                             self.num_classes,
        #                             dtype=torch.long)
        labels = torch.full(
            (num_bboxes,),                # Shape of the tensor
            self.num_classes,             # Fill value
            dtype=torch.long,             # Data type
            device=gt_bboxes[0].device       # Device matching `gt_bboxes`
        )
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # label_weights = gt_bboxes.new_ones(num_bboxes)
        label_weights = torch.ones(num_bboxes)

        # bbox targets
        # theta in gt_bbox here is still a single scalar
        bbox_targets = torch.zeros_like(bbox_pred[0])[:, :self.code_size-1]
        bbox_weights = torch.zeros_like(bbox_pred[0])[:, :self.code_size-1]

        # only matched query will learn from bbox coord
        bbox_weights[pos_inds] = 1.0

        # fix empty gt bug in multi gpu training
        if sampling_result.pos_gt_bboxes.shape[0] == 0:
            sampling_result.pos_gt_bboxes = \
                sampling_result.pos_gt_bboxes.reshape(0, self.code_size - 1)

        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes[:, :self.code_size-1]
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def get_targets(
            self,
            batch_cls_scores: List[Tensor],  # bs[num_q,num_cls]
            batch_bbox_preds: List[Tensor],  # bs[num_q,10]
            batch_gt_instances_3d: InstanceList) -> tuple():
        """"Compute regression and classification targets for a batch image for
        a single decoder layer.

        Args:
            batch_cls_scores (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            batch_bbox_preds (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx,cy,l,w,cz,h,sin(φ),cos(φ),v_x,v_y) and
                shape [num_query, 10]
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes_3d``、``labels_3d``.
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
        '''
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(self._get_target_single,
                                                     batch_cls_scores,
                                                     batch_bbox_preds,
                                                     batch_gt_instances_3d)
        '''
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,pos_inds_list, neg_inds_list = self._get_target_single(
                                                    batch_cls_scores,
                                                    batch_bbox_preds,
                                                    batch_gt_instances_3d)
        
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_by_feat_single(
        self,
        batch_cls_scores: Tensor,  # bs,num_q,num_cls
        batch_bbox_preds: Tensor,  # bs,num_q,10
        batch_gt_instances_3d: InstanceList
    ) -> Tuple[Tensor, Tensor]:
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
           batch_cls_scores (Tensor): Box score logits from a single
                decoder layer for batched images with shape [num_query,
                cls_out_channels].
            batch_bbox_preds (Tensor): Sigmoid outputs from a single
                decoder layer for batched images, with normalized coordinate
                (cx,cy,l,w,cz,h,sin(φ),cos(φ),v_x,v_y) and
                shape [num_query, 10]
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d. It usually has ``bboxes_3d``,``labels_3d``.
        Returns:
            tulple(Tensor, Tensor): cls and reg loss for outputs from
                a single decoder layer.
        """
        batch_size = batch_cls_scores.size(0)  # batch size
        cls_scores_list = [batch_cls_scores[i] for i in range(batch_size)]
        bbox_preds_list = [batch_bbox_preds[i] for i in range(batch_size)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances_3d)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        '''
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        '''
        labels = labels_list.to(batch_cls_scores.device)
        label_weights = label_weights_list.to(batch_cls_scores.device)
        bbox_targets = bbox_targets_list
        bbox_weights = bbox_weights_list

        # classification loss
        batch_cls_scores = batch_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                batch_cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            batch_cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        
        # regression L1 loss
        batch_bbox_preds = batch_bbox_preds.reshape(-1,
                                                    batch_bbox_preds.size(-1))

        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        # neg_query is all 0, log(0) is NaN
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights[:self.code_size-1]

        loss_bbox = self.loss_bbox(
            batch_bbox_preds[isnotnan, :self.code_size-1],
            normalized_bbox_targets[isnotnan, :self.code_size-1],
            bbox_weights[isnotnan, :self.code_size-1],
            avg_factor=num_total_pos)

        # get the x1, y1, x2, y2 coords
        iou_bbox_preds = convert_to_xyxy_rotated_custom(batch_bbox_preds[isnotnan, :self.code_size])
        iou_bbox_targets = convert_to_xyxy_rotated_custom(normalized_bbox_targets[isnotnan, :self.code_size])

        # det_polygon_list = list(convert_format(convert_to_corners(iou_bbox_preds)))
        # gt_polygon_list = list(convert_format(convert_to_corners(iou_bbox_targets)))
        # det_polygon_list = convert_to_corners(iou_bbox_preds).detach().cpu().numpy()
        # gt_polygon_list = convert_to_corners(iou_bbox_targets).detach().cpu().numpy()
        # save_bbox_plot(det_polygon_list, gt_polygon_list, output_path="/home/csmaj/jeli/MultiModalityPerception/bbox.png")
        # exit()
        # ious = []
        # ids = []
        # for det_polygon in det_polygon_list: 
        #     ious.append(np.max(compute_iou(det_polygon, gt_polygon_list)))
        #     ids.append(np.argmax(compute_iou(det_polygon, gt_polygon_list)))
        # print(list([np.max(compute_iou(det_polygon_list[0], gt_polygon_list))]))
        # ious = list([np.max(compute_iou(det_polygon, gt_polygon_list) for det_polygon in det_polygon_list)])
        # print(ids[:50])
        # # print(ious)
        # print(loss_cls.shape)
        # exit()

        loss_iou = self.loss_iou(
            iou_bbox_preds, 
            iou_bbox_targets, 
            bbox_weights[isnotnan, :4], 
            avg_factor=num_total_pos)
        # loss_iou = loss_iou/iou_bbox_preds.size(0)

        # loss_cls = torch.tensor([0.0])
        loss_bbox = loss_bbox.unsqueeze(0)
        loss_bbox = loss_bbox.unsqueeze(0)
        loss_iou = loss_iou.unsqueeze(0)
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        loss_iou = torch.nan_to_num(loss_iou)
        return loss_cls, loss_bbox, loss_iou

    # original loss()
    def loss_by_feat(
            self,
            batch_gt_instances_3d: InstanceList,
            preds_dicts: Dict[str, Tensor],
            batch_gt_instances_3d_ignore: OptInstanceList = None) -> Dict:
        """Compute loss of the head.

        Args:
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、`
                `labels_3d``、``depths``、``centers_2d`` and attributes.
                gt_instance.  It usually includes ``bboxes``、``labels``.
            batch_gt_instances_3d_ignore (list[:obj:`InstanceData`], Optional):
                NOT supported.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert batch_gt_instances_3d_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for batch_gt_instances_3d_ignore setting to None.'
        all_cls_scores = preds_dicts['all_cls_scores']  # num_dec,bs,num_q,num_cls
        all_bbox_preds = preds_dicts['all_bbox_preds']  # num_dec,bs,num_q,10
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        # calculate loss for each decoder layer
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_3d_list = [
            batch_gt_instances_3d for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox, losses_iou = multi_apply(self.loss_by_feat_single,
                                              all_cls_scores, all_bbox_preds,
                                              batch_gt_instances_3d_list)
        
        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            enc_loss_cls, enc_losses_bbox = self.loss_by_feat_single(
                enc_cls_scores, enc_bbox_preds, batch_gt_instances_3d_list)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    def predict_by_feat(self,
                        preds_dicts,
                        rescale=False) -> InstanceList:
        """Transform network output for a batch into bbox predictions.

        Args:
            preds_dicts (Dict[str, Tensor]):
                -all_cls_scores (Tensor): Outputs from the classification head,
                    shape [nb_dec, bs, num_query, cls_out_channels]. Note
                    cls_out_channels should includes background.
                -all_bbox_preds (Tensor): Sigmoid outputs from the regression
                    head with normalized coordinate format
                    (cx, cy, l, w, cz, h, rot_sine, rot_cosine, v_x, v_y).
                    Shape [nb_dec, bs, num_query, 10].
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                  (num_instances, C), where C >= 7.
        """
        # sinθ & cosθ ---> θ
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)  # batch size
        ret_list = []
        for i in range(num_samples):
            results = InstanceData()
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            #bboxes = img_metas[i]['box_type_3d'](bboxes, self.code_size - 1)

            results.bboxes_3d = bboxes
            results.scores_3d = preds['scores']
            results.labels_3d = preds['labels']
            ret_list.append(results)
        return ret_list


def convert_to_xyxy_rotated_custom(bboxes):
    """
    Convert bbox tensor from (cx, cy, w, l, cz, h, sin(rot), cos(rot))
    to (x1, y1, x2, y2) for IoU calculation, accounting for rotation.
    
    Args:
        bboxes (torch.Tensor): Tensor of shape (num_predictions, 8).
    
    Returns:
        torch.Tensor: Tensor of shape (num_predictions, 4) with (x1, y1, x2, y2).
    """
    # Extract bbox components
    cx, cy, w, l = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    sin_rot, cos_rot = bboxes[:, 6], bboxes[:, 7]
    
    # Reconstruct rotation angle
    rot = torch.atan2(sin_rot, cos_rot)
    
    # Compute rotated corners
    dx = l / 2
    dy = w / 2
    corners = torch.stack([
        torch.stack([-dx, -dy], dim=-1),
        torch.stack([ dx, -dy], dim=-1),
        torch.stack([ dx,  dy], dim=-1),
        torch.stack([-dx,  dy], dim=-1),
    ], dim=1)  # Shape: (num_predictions, 4, 2)

    # Rotation matrix components
    cos_rot = torch.cos(rot)
    sin_rot = torch.sin(rot)
    rotation_matrix = torch.stack([cos_rot, -sin_rot, sin_rot, cos_rot], dim=-1).view(-1, 2, 2)
    
    # Rotate corners
    rotated_corners = torch.einsum('nij,nkj->nki', rotation_matrix, corners)

    # Translate to center
    rotated_corners += torch.stack([cx, cy], dim=-1).unsqueeze(1)

    # Get (x1, y1, x2, y2)
    x1, _ = rotated_corners[..., 0].min(dim=1)
    y1, _ = rotated_corners[..., 1].min(dim=1)
    x2, _ = rotated_corners[..., 0].max(dim=1)
    y2, _ = rotated_corners[..., 1].max(dim=1)

    return torch.stack([x1, y1, x2, y2], dim=-1)


import numpy as np
from shapely.geometry import Polygon

def compute_iou(box, boxes):
    """
    Compute iou between box and boxes list
    Parameters
    ----------
    box : shapely.geometry.Polygon
        Bounding box Polygon.

    boxes : list
        List of shapely.geometry.Polygon.

    Returns
    -------
    iou : np.ndarray
        Array of iou between box and boxes.

    """
    # Calculate intersection areas
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]

    return np.array(iou, dtype=np.float32)


def convert_format(boxes_array):
    """
    Convert boxes array to shapely.geometry.Polygon format.
    Parameters
    ----------
    boxes_array : np.ndarray
        (N, 4, 2) or (N, 8, 3).

    Returns
    -------
        list of converted shapely.geometry.Polygon object.

    """
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in
                boxes_array]
    return np.array(polygons)


def convert_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert (N, 4) bounding boxes [x1, y1, x2, y2] to (N, 4, 2) corner points.

    Args:
        boxes (torch.Tensor): Shape (N, 4), where each row is [x1, y1, x2, y2]
    
    Returns:
        torch.Tensor: Shape (N, 4, 2), where each row contains 4 corner points [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # Define corner points
    top_left = torch.stack((x1, y1), dim=-1)
    top_right = torch.stack((x2, y1), dim=-1)
    bottom_right = torch.stack((x2, y2), dim=-1)
    bottom_left = torch.stack((x1, y2), dim=-1)

    # Stack into shape (N, 4, 2)
    corner_points = torch.stack((top_left, top_right, bottom_right, bottom_left), dim=1)
    return corner_points

import matplotlib.pyplot as plt
import os
def save_bbox_plot(pred_bboxes, gt_bboxes, output_path):
    """
    Plots predicted and ground truth bounding boxes on a 2D map and saves the image.
    
    Args:
        pred_bboxes (np.ndarray): Predicted bounding boxes of shape (N, 4, 2).
        gt_bboxes (np.ndarray): Ground truth bounding boxes of shape (N, 4, 2).
        output_path (str): File path (including filename) where the image will be saved.
    """
    
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot ground truth bounding boxes (green)
    for bbox in gt_bboxes:
        # Create a closed polygon from the 4 points
        polygon = plt.Polygon(bbox, closed=True, fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(polygon)
    
    # Plot predicted bounding boxes (red)
    for bbox in pred_bboxes:
        polygon = plt.Polygon(bbox, closed=True, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(polygon)
    
    # Calculate axis limits so all boxes are well-framed
    # Concatenate all points from both GT and predicted bboxes
    all_points = np.concatenate((gt_bboxes.reshape(-1, 2), pred_bboxes.reshape(-1, 2)), axis=0)
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    
    # Add a margin around the points
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)
    
    ax.set_aspect('equal')
    ax.axis('off')  # Turn off the axis

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the figure to disk without displaying it
    print(output_path)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()