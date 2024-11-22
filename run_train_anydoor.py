import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.ytb_vos import YoutubeVOSDataset
from datasets.ytb_vis import YoutubeVISDataset
from datasets.saliency_modular import SaliencyDataset
from datasets.vipseg import VIPSegDataset
from datasets.mvimagenet import MVImageNetDataset
from datasets.sam import SAMDataset
from datasets.uvo import UVODataset
from datasets.uvo_val import UVOValDataset
from datasets.mose import MoseDataset
from datasets.vitonhd import VitonHDDataset
from datasets.fashiontryon import FashionTryonDataset
from datasets.lvis import LvisDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from torch.utils.data import ConcatDataset
from cldm.hack import disable_verbosity, enable_sliced_attention
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

import warnings
warnings.filterwarnings("ignore")

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

# Configs
resume_path = 'models/epoch=1-step=8687-pruned.ckpt'
batch_size = 2
logger_freq = 1000
learning_rate = 1e-5
sd_locked = False
only_mid_control = False
n_gpus = 1
accumulate_grad_batches=1

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./configs/anydoor.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Datasets torch.amp.autocast('cuda', **ctx.gpu_autocast_kwargs)

#DConf = OmegaConf.load('./configs/datasets.yaml')
# dataset1 = YoutubeVOSDataset(**DConf.Train.YoutubeVOS)  
# dataset2 =  SaliencyDataset(**DConf.Train.Saliency) 
# dataset3 = VIPSegDataset(**DConf.Train.VIPSeg) 
# dataset4 = YoutubeVISDataset(**DConf.Train.YoutubeVIS) 
# dataset5 = MVImageNetDataset(**DConf.Train.MVImageNet)
# dataset6 = SAMDataset(**DConf.Train.SAM)
# dataset7 = UVODataset(**DConf.Train.UVO.train)
# dataset8 = VitonHDDataset(**DConf.Train.VitonHD)
# dataset9 = UVOValDataset(**DConf.Train.UVO.val)
# dataset10 = MoseDataset(**DConf.Train.Mose)
#dataset11 = FashionTryonDataset(**DConf.Train.FashionTryon)
# dataset12 = LvisDataset(**DConf.Train.Lvis)

# image_data = [dataset2, dataset6, dataset12]
# video_data = [dataset1, dataset3, dataset4, dataset7, dataset9, dataset10 ]
# tryon_data = [dataset8, dataset11]
# threed_data = [dataset5]
data_path = '/home/sjohnny/gill/VPR/VisualLearning-Project/data/train'
dataset = FashionTryonDataset(data_path)

# Define a TensorBoard logger
#logger = TensorBoardLogger("logs", name="my_model")
# The ratio of each dataset is adjusted by setting the __len__ 
# dataset = ConcatDataset( image_data + video_data + tryon_data +  threed_data + video_data + tryon_data +  threed_data  )
dataloader = DataLoader(dataset, num_workers=1, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
 
# trainer = pl.Trainer( strategy="ddp", precision=16, accelerator="gpu", callbacks=[logger], enable_progress_bar=True,accumulate_grad_batches=accumulate_grad_batches)
trainer = pl.Trainer( strategy="ddp_find_unused_parameters_true", precision=16, accelerator="gpu", callbacks=[logger], enable_progress_bar=True,accumulate_grad_batches=accumulate_grad_batches)

# Train!
trainer.fit(model, dataloader)





# ############################################################
# ############################################################
# ############################################################
# ## Dummy and Simple Training Code

# import pytorch_lightning as pl
# from torch.utils.data import DataLoader, Dataset, ConcatDataset
# import torch
# import torch.nn as nn

# # Create a dummy dataset that generates random data
# class DummyDataset(Dataset):
#     def __init__(self, length=1000, input_size=(3, 224, 224)):
#         self.length = length
#         self.input_size = input_size

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         # Generate random input and target tensors
#         input_tensor = torch.randn(self.input_size)
#         target_tensor = torch.randint(0, 10, (1,))
#         return {'input': input_tensor, 'target': target_tensor}

# # Define a simple model for testing
# class SimpleModel(pl.LightningModule):
#     def __init__(self, learning_rate=1e-5):
#         super(SimpleModel, self).__init__()
#         self.learning_rate = learning_rate
#         self.model = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(3 * 224 * 224, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10)
#         )
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         inputs = batch['input']
#         targets = batch['target'].squeeze()
#         outputs = self(inputs)
#         loss = self.criterion(outputs, targets)
#         self.log('train_loss', loss)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#         return optimizer

# # Configurations
# batch_size = 16
# learning_rate = 1e-5
# n_gpus = 1 if torch.cuda.is_available() else 0
# accumulate_grad_batches = 1

# # Instantiate the model
# model = SimpleModel(learning_rate=learning_rate)

# # Create dummy datasets to mimic your original datasets
# dataset1 = DummyDataset()
# dataset2 = DummyDataset()
# dataset3 = DummyDataset()
# dataset4 = DummyDataset()
# dataset5 = DummyDataset()
# dataset6 = DummyDataset()
# dataset7 = DummyDataset()
# dataset8 = DummyDataset()
# dataset9 = DummyDataset()
# dataset10 = DummyDataset()
# dataset11 = DummyDataset()
# dataset12 = DummyDataset()

# # Group datasets as in your original code
# image_data = [dataset2, dataset6, dataset12]
# video_data = [dataset1, dataset3, dataset4, dataset7, dataset9, dataset10]
# tryon_data = [dataset8, dataset11]
# threed_data = [dataset5]

# # Concatenate datasets
# dataset = ConcatDataset(image_data + video_data + tryon_data + threed_data + video_data + tryon_data + threed_data)

# # Create DataLoader
# dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)

# # Set up the trainer
# trainer = pl.Trainer(
#     gpus=n_gpus,
#     precision=16 if n_gpus > 0 else 32,
#     accumulate_grad_batches=accumulate_grad_batches,
#     max_epochs=10  # Set to 1 for testing purposes
# )

# # Start training
# trainer.fit(model, dataloader)


############################################################
############################################################
############################################################
## Original Main Training Code
