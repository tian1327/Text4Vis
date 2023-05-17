import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision
import time
from utils.utils import init_distributed_mode, AverageMeter, reduce_tensor, accuracy
import clip

import yaml
from dotmap import DotMap

from datasets import Video_dataset
from datasets.transforms import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, GroupNormalize, GroupOverSample, GroupFullResSample
from modules.video_clip import video_header, VideoCLIP
from modules.text_prompt import text_prompt



# load the torch tensor of the label embeddings
label_embeddings = torch.load('text_feats_anet_ViT-L14.pt')
print('label_embeddings.shape:', label_embeddings.shape)

# load the text class labels from lists/anet/anet1.3_labels.csv into dataframe
import pandas as pd
df = pd.read_csv('lists/anet1.3_labels.csv')
print(df.info())

# calculate the correlation of each row vector in the label embeddings
# with each other row vector in the label embeddings
corr_matrix = np.corrcoef(label_embeddings)
print('corr_matrix.shape:', corr_matrix.shape)

# plot the correlation matrix, set colorbar to 'hot' for better visualization
import matplotlib.pyplot as plt
# plt.imshow(corr_matrix, cmap='hot', vmin=0.6, vmax=1)
plt.imshow(corr_matrix, vmin=0.6, vmax=1)
plt.xticks(np.arange(0, 210, 10), np.arange(0, 210, 10), rotation=90)
plt.yticks(np.arange(0, 210, 10), np.arange(0, 210, 10))
plt.tight_layout()
plt.colorbar()
plt.show()


# plot the correlation labels from 80 to 130
plt.imshow(corr_matrix[80:131, 80:131], vmin=0.6, vmax=1)
plt.xticks(np.arange(0, 51, 5), np.arange(80, 131, 5), rotation=90)
plt.yticks(np.arange(0, 51, 5), np.arange(80, 131, 5))
plt.tight_layout()
plt.colorbar()
plt.show()
