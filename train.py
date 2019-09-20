import os
import numpy as np
import torch
from torchvision import datasets, models, transforms
import model
from torch.utils.data import Dataset, DataLoader
from dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
import eval

print('pytorch version:', torch.__version__)

def main():
    data_type = 'coco'
    data_root_dir = '/data/data_coco/'
    # model_depth = 50
    epoch_max = 100
    batch_size = 8

    if data_type == 'coco':
        dataset_train = CocoDataset(data_root_dir, set_name='train2017', transform=transforms.Compose([Normalizer(),Augmenter(), Resizer()]))
        dataset_val   = CocoDataset(data_root_dir, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()])) 
    else:
        print('暂不支持')
    
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=batch_size, drop_last=True)
    loader_train = DataLoader(dataset_train, num_workers=8, collate_fn=collater, batch_sampler=sampler)
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=batch_size, drop_last=True)
    loader_val = DataLoader(dataset_val, num_workers=8, collate_fn=collater, batch_sampler=sampler_val)

    retinanet = model.retinanet_50(dataset_train.num_classes(), pretrained=True)

    retinanet = retinanet.cuda()
    optimizer = torch.optim.Adam(retinanet.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(retinanet.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.5)

    model_pretrain_dir = './model/model_final.pt'
    if os.path.exists(model_pretrain_dir):
        print('pretrain model exist!')
        retinanet = torch.load(model_pretrain_dir)

    print('train images num: {}'.format(len(loader_train)*batch_size))
    for epoch_num in range(epoch_max):
        retinanet.train()
        epoch_loss = []
        for iter_num, data in enumerate(loader_train):
            optimizer.zero_grad()
            input_tensor = [data['img'].cuda().float(), data['annot']]
            classification_loss, regression_loss = retinanet(input_tensor)
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss

            epoch_loss.append(float(loss))

            if loss.item() == 0:
                continue
            
            loss.backward()
            optimizer.step()

            print('Epoch:{}/{} | Iters:{}/{} | C loss:{:.4f} | R loss:{:.4f} | Current loss:{:.4f} | Current LR:{:.7f}'.format(epoch_num+1, epoch_max, iter_num+1, len(loader_train), float(classification_loss), float(regression_loss), np.mean(epoch_loss), optimizer.param_groups[0]['lr']))
            del classification_loss
            del regression_loss

        # 每个epoch 进行验证一次
        eval.eval_coco(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))
        torch.save(retinanet, './model/{}_retinanet_{}.pt'.format(data_type, epoch_num+1))
    retinanet.eval()
    torch.save(retinanet, './model/model_final.pt')

main()


