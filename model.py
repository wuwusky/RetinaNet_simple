import torch.nn as nn
import torch
import math
# import time
import numpy as np
import torch.utils.model_zoo as model_zoo
import loss
import anchors



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        else:
            self.std = std

    def forward(self, boxes, deltas):

        widths  = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        ctr_y   = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes

class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
      
        return boxes

class PFN(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, F_size=256):
        super(PFN, self).__init__()
        #根据C5得到P5,通过上采样后面与C4相加
        self.P5_1 = nn.Conv2d(C5_size, F_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(F_size, F_size, kernel_size=3, stride=1, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2.)
        #根据C4得到P4,通过上采样后后面与C3相加
        self.P4_1 = nn.Conv2d(C4_size, F_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(F_size, F_size, kernel_size=3, stride=1, padding=1)
        #根据C3得到P3
        self.P3_1 = nn.Conv2d(C3_size, F_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(F_size, F_size, kernel_size=3, stride=1, padding=1)
        #根据C5得到P6
        self.P6 = nn.Conv2d(C5_size, F_size, kernel_size=3, stride=2, padding=1)
        #根据P6得到P7
        self.P7 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(F_size, F_size, kernel_size=3, stride=2, padding=1)
        )
    def forward(self, inputs):
        C3, C4, C5 = inputs
        P5 = self.P5_1(C5)
        P5_up = self.up(P5)
        P5 = self.P5_2(P5)

        P4 = self.P4_1(C4)
        P4 = P5_up + P4
        P4_up = self.up(P4)
        P4 = self.P4_2(P4)

        P3 = self.P3_1(C3)
        P3 = P4_up + P3
        P3 = self.P3_2(P3)

        P6 = self.P6(C5)

        P7 = self.P7(P6)

        return P3, P4, P5, P6, P7

class BoxDetect(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, F_size=256):
        super(BoxDetect, self).__init__()
        self.detect = nn.Sequential(
            nn.Conv2d(num_features_in, F_size, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(F_size, F_size, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(F_size, F_size, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(F_size, F_size, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        self.out = nn.Conv2d(F_size, num_anchors*4, kernel_size=3, padding=1)
    def forward(self, x):
        out = self.detect(x)
        out = self.out(out)
        # 此时输出为 B C W H ,需要整形为 B W H (num_anchors*4)
        out = out.permute(0, 2, 3, 1)
        out = out.contiguous().view(out.shape[0], -1, 4)
        return out

class Classification(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes = 1000, prior=0.01, F_size=256):
        super(Classification, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.detect = nn.Sequential(
            nn.Conv2d(num_features_in, F_size, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(F_size, F_size, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(F_size, F_size, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(F_size, F_size, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        self.out = nn.Conv2d(F_size, num_anchors*num_classes, kernel_size=3, padding=1)
        self.out_sig = nn.Sigmoid()

    def forward(self, x):
        out = self.detect(x)
        out = self.out(out)
        out = self.out_sig(out)
        # 此时输出为 B C W H ,需要整形为 B W H (num_anchors*num_classes)
        out = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out.shape
        out = out.view(batch_size, width, height, self.num_anchors, self.num_classes)
        out = out.contiguous().view(batch_size, -1, self.num_classes)
        return out

def nms(dets, thresh, mode='union'):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2-x1+1) * (y2-y1+1)
    order = scores.sort(0, descending=True)[1]
    keep = []
    while order.numel() > 0:
        if len(order.shape) > 0:
            i = order[0].data.item()
        else:
            i = order.data.item()
        keep.append(i)

        if order.numel() == 1:
            break
        
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        inter = w*h

        if mode == 'union':
            ovr = inter/(areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter/areas[order[1:]].clamp(max = areas[i])
        
        ids = (ovr <= thresh).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)

class RetinaNet(nn.Module):
    def __init__(self, num_classes, block, layers):
        super(RetinaNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [128, 256, 512]
        elif block == Bottleneck:
            fpn_sizes = [512, 1024, 2048]
        
        self.fpn = PFN(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        self.regression = BoxDetect(256)
        self.classification = Classification(256, num_classes=num_classes)

        self.anchors = anchors.Anchors()
        self.boxs_regression = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.prior = 0.01
        self.classification.out.weight.data.fill_(0)
        self.classification.out.bias.data.fill_(-math.log((1.0-self.prior)/self.prior))

        self.regression.out.weight.data.fill_(0)
        self.regression.out.bias.data.fill_(0)
        self.freeze_bn

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
            annotations = annotations.cuda()
        else:
            img_batch = inputs
        
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        features = self.fpn([c3, c4, c5])

        list_regressions = []
        list_classifications = []
        for feature in features:
            temp_regression = self.regression(feature)
            list_regressions.append(temp_regression)
            temp_classification = self.classification(feature)
            list_classifications.append(temp_classification)
        regressions = torch.cat(list_regressions, dim=1)
        classifications = torch.cat(list_classifications, dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            c_losses, r_losses = loss.Focal_Loss(classifications, regressions, anchors, annotations)
            return c_losses, r_losses
        else:
            transformed_anchors = self.boxs_regression(anchors, regressions)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)
            scores = torch.max(classifications, dim=2, keepdim=True)[0]
            scores_over_thresh = (scores > 0.05)[0, :, 0]
            if scores_over_thresh.sum() == 0:
                # 没有满足条件的检测框进行不NMS，直接返回
                nms_scores = torch.zeros(0)
                nms_class = torch.zeros(0)
                transformed_anchors = torch.zeros(0, 4)
                return nms_scores, nms_class, transformed_anchors
            else:
                # 满足条件的检测框进行NMS
                classifications = classifications[:, scores_over_thresh, :]
                transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
                scores = scores[:, scores_over_thresh, :]

                anchors_with_scores = torch.cat([transformed_anchors, scores], dim=2)
                anchors_with_scores = anchors_with_scores[0, :, :]
                anchors_nms_idx = nms(anchors_with_scores, 0.5)
                nms_scores, nms_class = classifications[0, anchors_nms_idx, :].max(dim=1)
                return nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]

def retinanet_50(num_classes, pretrained=False, **kwargs):
    model = RetinaNet(num_classes, Bottleneck, [3,4,6,3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model
def retinanet_18(num_classes, pretrained=False, **kwargs):
    model = RetinaNet(num_classes, BasicBlock, [2,2,2,2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


model_test = retinanet_50(num_classes=99, pretrained=False)
model_test = model_test.eval().cuda()

optimizer = torch.optim.SGD(model_test.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
print(optimizer.param_groups[0]['lr'])  
input_tensor = torch.rand(2, 3, 224, 224).cuda()
out = model_test(input_tensor)


