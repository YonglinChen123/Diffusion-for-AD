import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import time, datetime
import pdb, traceback

import cv2
# import imagehash
from PIL import Image

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b4')

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from resnet import Resnet18
from ViTnet import ViT

class QRDataset(Dataset):
    def __init__(self, train_jpg, transform=None):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        start_time = time.time()
        img = Image.open(self.train_jpg[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.from_numpy(np.array(int('AD' in self.train_jpg[index])))

    def __len__(self):
        return len(self.train_jpg)


#---------------------------------
class VisitNet(nn.Module):
    def __init__(self):
        super(VisitNet, self).__init__()
        model1 = Resnet18(3)
        self.resnet1 = model1
    def forward(self, img):
        out = self.resnet1(img)
        return out
#---------------------------------

def predict(test_loader, model, tta=10):
    # switch to evaluate mode
    model.eval()

    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(test_loader):
                input = input.cuda()
                target = target.cuda()

                # compute output
                output = model(input)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


test_jpg = [r'Test400\{0}.png'.format(x) for x in range(1, 401)]
test_jpg = np.array(test_jpg)

test_pred = None

for model_path in ['CNN-VIT(NEW).pt']:
    test_loader = torch.utils.data.DataLoader(
        QRDataset(test_jpg,
                  transforms.Compose([
                      transforms.Resize((96, 96)),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])
                  ), batch_size=10, shuffle=False, num_workers=0, pin_memory=True
    )


###########################################multi-attention mechanism
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.ReLU(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.avg_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        return out

class VisitNet(nn.Module):
    def __init__(self):
        super(VisitNet, self).__init__()
        model1 = Resnet18(3)
        model2 = ViT(
            image_size=96,
            patch_size=32,
            num_classes=3,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.resnet1 = model1
        self.dsvit = model2

    def forward(self, img):
        out2 = self.resnet1(img)
        out1 = self.dsvit(img, out2)
        Foutput = torch.mean(torch.stack([out1, out2]), 0)
        cbam = CBAM(channel=150)
        out = cbam(Foutput)
        return out
    model = VisitNet().cuda()
    model_path = os.path.join("model", model_path)
    model.load_state_dict(torch.load(model_path))
    #model2.load_state_dict(torch.load(model_path))
    # model = nn.DataParallel(model).cuda()
    if test_pred is None:
        test_pred = predict(test_loader, model, 1)
    else:
        test_pred += predict(test_loader, model, 1)

preds_cache_all = test_pred  # (2000,3) * 15
np.save(f'./npy/CNN-VIT(NEW).npy', preds_cache_all)  # 保存时更改 'se101', 'b5', 'b6'
test_csv = pd.DataFrame()
test_csv['uuid'] = list(range(1, 401))
test_csv['label'] = np.argmax(test_pred, 1)
test_csv['label'] = test_csv['label'].map({1: 'AD', 0: 'NC', 2: 'MCI'})
test_csv.to_csv('tmp.csv', index=None)

