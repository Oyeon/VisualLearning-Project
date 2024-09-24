import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn

import cv2
import numpy as np
from tqdm import tqdm

from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

from torchvision.transforms import v2
from torchvision import tv_tensors, datasets

from torchvision.utils import save_image
from torchinfo import summary
from torchvision.utils import draw_segmentation_masks



transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomPhotometricDistort(p=1),
        v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
        v2.RandomIoUCrop(),
        v2.RandomHorizontalFlip(p=1),
        v2.SanitizeBoundingBoxes(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

transforms_test = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

trainset = CocoDetection(root='/home/baekw92/ai_editor/train2017', annFile='/home/baekw92/ai_editor/annotations/instances_train2017_adj.json', transforms=transforms)
trainset = datasets.wrap_dataset_for_transforms_v2(trainset, target_keys=("boxes", "labels", "masks"))


validset = CocoDetection(root='/home/baekw92/ai_editor/val2017', annFile='/home/baekw92/ai_editor/annotations/instances_val2017.json', transforms=transforms_test)
validset = datasets.wrap_dataset_for_transforms_v2(validset, target_keys=("boxes", "labels", "masks"))



def collate_fn(batch):
    return tuple(zip(*batch))
trainloader = DataLoader(trainset, batch_size=2, shuffle=True, collate_fn=collate_fn)
validloader = DataLoader(validset, batch_size=2, shuffle=True, collate_fn=collate_fn)

num_epochs = 60
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

model = maskrcnn_resnet50_fpn(pretrained=True, progress=False).to(device)
summary(model)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=3e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 50], gamma=0.1)

for epoch in range(num_epochs):
    model.train()
    loader = tqdm(trainloader)
    loss_sum = 0.0
    loss_mask_sum = 0.0
    for i, (images, targets) in enumerate(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        loss_mask = losses['loss_mask'].item()
        loss_sum += loss.item()
        loss_mask_sum += loss_mask

        loader.set_description('train {}/{}: loss: {:.3f}, mask_loss: {:.3f}'.format(epoch, num_epochs, loss_sum/(i+1), loss_mask_sum/(i+1)))

                    
    scheduler.step()

    model.eval()
    loader = tqdm(validloader)
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            
            outputs = model(images)
            candidate_idx = torch.where(outputs[0]['scores'] > .9)[0]
            a = draw_segmentation_masks(images[0], torch.where(outputs[0]['masks'][candidate_idx].squeeze(1)>0.8, 1, 0).type(torch.BoolTensor))
            save_image(a,'result.png')
        


            









