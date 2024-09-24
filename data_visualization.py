import torch
import random
from torchvision.datasets import CocoDetection

import matplotlib.pyplot as plt


trainset = CocoDetection(root='/home/baekw92/ai_editor/train2017', annFile='/home/baekw92/ai_editor/annotations/instances_train2017.json')
# validset = CocoDetection(root='val2017', annFile='annotations/instances_val2017.json')


for i in range(10):
    rnd = random.randint(0,len(trainset)-1)
    img, annos = trainset.__getitem__(rnd)
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.axis('off')
    trainset.coco.showAnns(annos)
    fig.tight_layout()
    plt.savefig('visualization_{}.png'.format(i))
    plt.close()
