import numpy as np
import torch
import torch.nn as nn

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    temp1 = torch.unsqueeze(a[:, 2], dim=1)
    temp2 = b[:, 2]
    iw2 = torch.min(temp1, temp2)
    iw1 = torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    iw =  iw2 - iw1
    ih2 = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3])
    ih1 = torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    ih = ih2 - ih1

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

def Focal_Loss(classifications, regressions, anchors, annotations):
    alpha = 0.25
    gamma = 2.0
    batch_size = classifications.shape[0]
    classification_losses = []
    regression_losses = []

    anchor = anchors[0, :, :]
    anchor_width    = anchor[:, 2] - anchor[:, 0]
    anchor_height   = anchor[:, 3] - anchor[:, 1]
    anchor_cx       = anchor[:, 0] + 0.5*anchor_width
    anchor_cy       = anchor[:, 1] + 0.5*anchor_height

    for j in range(batch_size):
        classification = classifications[j, :, :]
        regression = regressions[j, :, :]

        bbox_annotation = annotations[j, :, :]
        bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
        if bbox_annotation.shape[0] == 0:
            regression_losses.append(torch.tensor(0).float().cuda())
            classification_losses.append(torch.tensor(0).float().cuda())
            continue
        classification = torch.clamp(classification, 1e-4, 1.0-1e-4)
        Iou = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])
        Iou_max, Iou_argmax = torch.max(Iou, dim=1)

        # 类别损失
        targets = torch.ones(classification.shape) * -1
        targets = targets.cuda()
        targets[torch.lt(Iou_max, 0.4), :] = 0
        positive_indices = torch.ge(Iou_max, 0.5)
        num_positive_anchors = positive_indices.sum()
        assigned_annotations = bbox_annotation[Iou_argmax, :]
        targets[positive_indices, :] = 0
        targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
        alpha_factor = torch.ones(targets.shape).cuda() * alpha
        alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1.-alpha_factor)
        focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
        focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

        bce = -(targets * torch.log(classification) + (1.0 - targets)*torch.log(1.0-classification))

        cls_loss = focal_weight * bce
        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
        classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

        # 回归检测框损失
        if positive_indices.sum() > 0:
            assigned_annotations = assigned_annotations[positive_indices, :]
            anchor_width_pi     = anchor_width[positive_indices]
            anchor_height_pi    = anchor_height[positive_indices]
            anchor_cx_pi        = anchor_cx[positive_indices]
            anchor_cy_pi        = anchor_cy[positive_indices]

            gt_width    = assigned_annotations[:, 2] - assigned_annotations[:, 0]
            gt_height   = assigned_annotations[:, 3] - assigned_annotations[:, 1]
            gt_cx       = assigned_annotations[:, 0] + 0.5*gt_width
            gt_cy       = assigned_annotations[:, 1] + 0.5*gt_height

            gt_width = torch.clamp(gt_width, min=1)
            gt_height = torch.clamp(gt_height, min=1)

            targets_dx = (gt_cx - anchor_cx_pi)/anchor_width_pi
            targets_dy = (gt_cy - anchor_cy_pi)/anchor_height_pi
            targets_dw = torch.log(gt_width/anchor_width_pi)
            targets_dh = torch.log(gt_height/anchor_height_pi)

            targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
            targets = targets.t()

            targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

            negative_indices = 1 - positive_indices
            regression_diff = torch.abs(targets - regression[positive_indices, :])
            regression_loss = torch.where(
                torch.le(regression_diff, 1.0/9.0),
                0.5 * 9.0 * torch.pow(regression_diff, 2),
                regression_diff - 0.5/9.0
            )
            regression_losses.append(regression_loss.mean())
        else:
            regression_losses.append(torch.tensor(0).float().cuda())
    c_losses = torch.stack(classification_losses).mean(dim=0, keepdim=True)
    r_losses = torch.stack(regression_losses).mean(dim=0, keepdim=True)
    return c_losses, r_losses




