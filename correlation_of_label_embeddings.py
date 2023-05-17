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

# calculate the correlation matrix of the label embeddings
corr_matrix = torch.matmul(label_embeddings, label_embeddings.T)
print('corr_matrix.shape:', corr_matrix.shape)

# plot the correlation matrix
import matplotlib.pyplot as plt
plt.imshow(corr_matrix)
plt.colorbar()
plt.show()

