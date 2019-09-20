# from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# import numpy as np
import torch
# import os
import json

def eval_coco(dataset, model, threshold=0.5):
    model.eval()

    with torch.no_grad():
        print('开始验证')

        results = []
        image_ids = []

        for i in range(len(dataset)):
            data = dataset[i]
            scale = data['scale']
            data_tensor = data['img'].permute(2, 0, 1).cuda().float()
            data_tensor = data_tensor.unsqueeze(dim=0)
            out = model(data_tensor)
            scores, labels, boxes = out
            
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()

            boxes = boxes/scale

            if boxes.shape[0] > 0:
                # 转换检测框格式，从xy-xy转换为xy-wh
                boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    if score < threshold:
                        break
                    
                    image_result = {
                        'image_id' : dataset.image_ids[i],
                        'category_id' : dataset.label_to_coco_label(label),
                        'score' : float(score),
                        'bbox'  : box.tolist(),
                    }
                    
                    results.append(image_result)
            image_ids.append(dataset.image_ids[i])

            print('{}/{}'.format(i+1, len(dataset)), end='\r')
        
        if len(results) < 0:
            return
        
        json.dump(results, open('{}_results.json'.format(dataset.set_name), 'w'), indent=4)

        coco_true = dataset.coco
        coco_pred = coco_true.loadRes('{}_results.json'.format(dataset.set_name))

        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return


