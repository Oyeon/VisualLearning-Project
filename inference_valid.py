import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import v2
import torchvision.transforms as T
from torchvision import tv_tensors, datasets


from torchvision.utils import save_image
from torchinfo import summary
from torchvision.utils import draw_segmentation_masks

import glob


transforms_test = T.Compose(
    [
        T.ToTensor(),
    ]
)

class CustomDataset(Dataset):
    def __init__(self, files, transforms):
        self.files = files
        self.transforms = transforms

    def __getitem__(self,idx):
        fname = self.files[idx]
        img = Image.open(fname)
        inputs = self.transforms(img)
        return inputs
    def __len__(self):
        return len(self.files)


# files = glob.glob('/home/baekw92/K*.*')
# testset = CustomDataset(files, transforms_test)

transforms_test = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

validset = CocoDetection(root='/home/baekw92/ai_editor/val2017', annFile='/home/baekw92/ai_editor/annotations/instances_val2017.json', transforms=transforms_test)
validset = datasets.wrap_dataset_for_transforms_v2(validset, target_keys=("boxes", "labels", "masks"))

validloader = DataLoader(validset, batch_size=1, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

model = maskrcnn_resnet50_fpn(pretrained=True, progress=False).to(device)
summary(model)

model.eval()
loader = tqdm(validloader)
with torch.no_grad():
    for i, (images, targets) in enumerate(loader):
        if i == 10:
            break
        images = list(image.to(device) for image in images)
        outputs = model(images)
        candidate_idx = torch.where(outputs[0]['scores'] > .5)[0]
        a = draw_segmentation_masks(images[0], torch.where(outputs[0]['masks'][candidate_idx].squeeze(1)>0.8, 1, 0).type(torch.BoolTensor))
        save_image(a,'result.png')
    


        









