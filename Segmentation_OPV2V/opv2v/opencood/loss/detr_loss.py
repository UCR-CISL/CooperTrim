import torch
import torch.nn as nn

from einops import rearrange


class DETRLoss(nn.Module):
    def __init__(self, args):
        super(DETRLoss, self).__init__()
        self.loss_dict = {}
        self.iou_loss_weight = args['iou_loss_weight']
        self.bbox_loss_weight = args['bbox_loss_weight']
        self.cls_loss_weight = args['cls_loss_weight']
        self.heatmap_loss_weight = args['heatmap_loss_weight']

    def calculate_iou(self, box1, box2):
        # Ensure box1 and box2 are 2D tensors
        if box1.dim() == 1:
            box1 = box1.unsqueeze(0)
        if box2.dim() == 1:
            box2 = box2.unsqueeze(0)

        box1_x_min = box1[:, 0] - box1[:, 3] / 2
        box1_x_max = box1[:, 0] + box1[:, 3] / 2
        box1_y_min = box1[:, 1] - box1[:, 4] / 2
        box1_y_max = box1[:, 1] + box1[:, 4] / 2

        box2_x_min = box2[:, 0] - box2[:, 3] / 2
        box2_x_max = box2[:, 0] + box2[:, 3] / 2
        box2_y_min = box2[:, 1] - box2[:, 4] / 2
        box2_y_max = box2[:, 1] + box2[:, 4] / 2

        inter_x_min = torch.max(box1_x_min.unsqueeze(1), box2_x_min.unsqueeze(0))
        inter_y_min = torch.max(box1_y_min.unsqueeze(1), box2_y_min.unsqueeze(0))
        inter_x_max = torch.min(box1_x_max.unsqueeze(1), box2_x_max.unsqueeze(0))
        inter_y_max = torch.min(box1_y_max.unsqueeze(1), box2_y_max.unsqueeze(0))

        inter_area = torch.clamp(inter_x_max - inter_x_min, min=0) * torch.clamp(inter_y_max - inter_y_min, min=0)

        box1_area = (box1_x_max - box1_x_min) * (box1_y_max - box1_y_min)
        box2_area = (box2_x_max - box2_x_min) * (box2_y_max - box2_y_min)

        union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - inter_area
        iou = inter_area / torch.clamp(union_area, min=1e-6)
        return iou

    def forward(self, output_dict, gt_dict):
        self.loss_dict = output_dict
        total_bbox_loss = torch.sum(torch.stack([value for loss, value in self.loss_dict.items() if "loss_bbox" in loss]))
        total_iou_loss = torch.sum(torch.stack([value for loss, value in self.loss_dict.items() if "loss_iou" in loss]))
        total_cls_loss = torch.sum(torch.stack([value for loss, value in self.loss_dict.items() if "loss_cls" in loss]))
        total_heatmap_loss = torch.sum(torch.stack([value for loss, value in self.loss_dict.items() if "loss_heatmap" in loss]))

        final_loss = (self.iou_loss_weight * total_iou_loss + 
                        self.bbox_loss_weight * total_bbox_loss +
                        self.cls_loss_weight * total_cls_loss +
                        self.heatmap_loss_weight * total_heatmap_loss)
        return final_loss

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        loss_bbox = self.loss_dict['loss_bbox']
        loss_iou = self.loss_dict['loss_iou']
        loss_cls = self.loss_dict['loss_cls']
        loss_heatmap = self.loss_dict['loss_heatmap']

        if pbar is None:
            print("[epoch %d][%d/%d], || BBox Loss: %.4f || IoU Loss: %.4f || heatmap Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    loss_bbox.item(), loss_iou.item(), loss_cls.item(), loss_heatmap.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || BBox Loss: %.4f || IoU Loss: %.4f || cls Loss: %.4f || heatmap Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    loss_bbox.item(), loss_iou.item(), loss_cls.item(), loss_heatmap.item()))

        writer.add_scalar('bbox_loss', loss_bbox.item(),
                          epoch*batch_len + batch_id)

        writer.add_scalar('iou_loss', loss_iou.item(),
                          epoch*batch_len + batch_id)
        
        writer.add_scalar('cls_loss', loss_cls.item(),
                          epoch*batch_len + batch_id)

        writer.add_scalar('heatmap_loss', loss_heatmap.item(),
                          epoch*batch_len + batch_id)