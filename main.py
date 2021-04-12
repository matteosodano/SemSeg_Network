'''
IMPLEMENTATION OF A SEMANTIC SEGMENTATION NETWORK
Source: https://medium.com/analytics-vidhya/pytorch-implementation-of-semantic-segmentation-for-single-class-from-scratch-81f96643c98c

Implementation is divided into 4 pipelines:
1 Data Preprocessing: Converting train_mask images from .gif to .png, then we will convert both train and train mask
  images(.png) from their original dimension to new dimension[128,128]
2. Dataloader: fetch images in batches apply transforms to them & then returns dataloders for train and validation
3. Scores: calculating the required score
4. Training: Final pipeline where training begins, loss are calculated & parameters are updated

We will be using the Unet Architecture. For this we will use an high level API provided by segmentation_models_pytorch
'''

# visualization library
import cv2
from matplotlib import pyplot as plt

# data storing library
import numpy as np
import pandas as pd

# torch libraries
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from torchvision.transforms import ToTensor as tensorize

# architecture and data split library
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

# augmentation library
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor

# others
import os
import pdb
import time
import warnings
import random
from tqdm.notebook import tqdm as tqdm
import concurrent.futures
from pathlib import Path
import PIL
from PIL import Image

# warning print supression
warnings.filterwarnings("ignore")

# *****************to reproduce same results fixing the seed and hash*******************
seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

'''
1. DATA PREPROCESSING PIPELINE
'''
PATH = Path('/home/matteo/Code/SemSeg/carvana')

# using fastai below lines convert the gif image to pil image.
(PATH/'train_masks_png').mkdir(exist_ok=True)
def convert_img(fn):
    fn = fn.name
    PIL.Image.open(PATH/'train_masks'/fn).save(PATH/'train_masks_png'/f'{fn[:-4]}.png') # opening and saving image as png
files = list((PATH/'train_masks').iterdir())
#with concurrent.futures.ThreadPoolExecutor(8) as e: e.map(convert_img, files)  # uses multi thread for fast conversion

print('Converted train masks from gif to png.')

# we convert the high resolution image mask to 128*128 for starting for the masks.
(PATH/'train_masks-128').mkdir(exist_ok=True)
def resize_mask(fn):
    PIL.Image.open(fn).resize((128,128)).save((fn.parent.parent)/'train_masks-128'/fn.name)

files = list((PATH/'train_masks_png').iterdir())
#with concurrent.futures.ThreadPoolExecutor(8) as e: e.map(resize_mask, files)

print('Resized train masks to 128*128.')

# # # we convert the high resolution input image to 128*128
(PATH/'train-128').mkdir(exist_ok=True)
def resize_img(fn):
    PIL.Image.open(fn).resize((128,128)).save((fn.parent.parent)/'train-128'/fn.name)

files = list((PATH/'train').iterdir())
#with concurrent.futures.ThreadPoolExecutor(8) as e: e.map(resize_img, files)

print('Resized input images to 128*128.')

(PATH/'test-128').mkdir(exist_ok=True)
def resize_img(fn):
    PIL.Image.open(fn).resize((128,128)).save((fn.parent.parent)/'test-128'/fn.name)

files = list((PATH/'test').iterdir())
#with concurrent.futures.ThreadPoolExecutor(8) as e: e.map(resize_img, files)

print('Resized test images to 128*128.')


# load the train_masks.csv in dataframe just for getting the image names
df=pd.read_csv('/home/matteo/Code/SemSeg/carvana/train_masks.csv')

# locations of original and mask image
img_fol = '/home/matteo/Code/SemSeg/carvana/train-128'
mask_fol = '/home/matteo/Code/SemSeg/carvana/train_masks-128'
test_fol = 'home/matteo/Code/SemSeg/carvana/test-128'

# imagenet mean/std will be used as the resnet backbone is trained on imagenet stats
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

'''
2. DATALOADER PIPELINE
We will implement custom transforms , dataset and dataloader.
if “train” then we will use horizontal flip along with Normalize and ToTensor 
If “val” then we will only be using Normalize and ToTensor.
'''

# during traning/val phase make a list of transforms to be used.
# input-->"phase",mean,std
# output-->list
def get_transform(phase, mean, std):
    list_trans = []
    if phase == 'train':
        list_trans.extend([HorizontalFlip(p=0.5)])
    list_trans.extend([Normalize(mean=mean, std=std, p=1), ToTensor()])  # normalizing the data & then converting to tensors
    list_trans = Compose(list_trans)
    return list_trans

'''
We will create a custom dataset class named CarDataset: we fetch the original image and mask using the index id from 
dataloader and then apply transformation. Output from this class is image tensor of shape [3,128,128] and 
mask tensor [1,128,128] (one channel because we have a single class)
Then using CarDataloader function we split the input dataframe into train dataframe and valid dataframe
'''
class CarDataset(Dataset):
    def __init__(self, df, img_fol, mask_fol, mean, std, phase):
        self.fname = df['img'].values.tolist()
        self.img_fol = img_fol
        self.mask_fol = mask_fol
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transform = get_transform(phase, mean, std)

    def __getitem__(self, idx):
        name = self.fname[idx]
        img_name_path = os.path.join(self.img_fol, name)
        mask_name_path = img_name_path.split('.')[0].replace('train-128', 'train_masks-128') + '_mask.png'
        img = cv2.imread(img_name_path)
        mask = cv2.imread(mask_name_path, cv2.IMREAD_GRAYSCALE)
        augmentation = self.transform(image=img, mask=mask)
        img_aug = augmentation['image']                           # [3,128,128] type:Tensor
        mask_aug = augmentation['mask']                           # [1,128,128] type:Tensor
        return img_aug, mask_aug

    def __len__(self):
        return len(self.fname)

# divide data into train and val and return the dataloader depending upon train or val phase
def CarDataloader(df, img_fol, mask_fol, mean, std, phase, batch_size, num_workers):
    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=69)
    df = df_train if phase == 'train' else df_valid
    for_loader = CarDataset(df, img_fol, mask_fol, mean, std, phase)
    dataloader = DataLoader(for_loader, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return dataloader

'''
3. SCORES PIPELINE
To tackle the problem of class imbalance we use Soft Dice Score
Inside every epoch for all the batch we calculate the dice score & append in a empty list. At the end of epoch we 
calculate the mean of dice scores which represent dice score for that particular epoch
'''
'''calculates dice scores when Scores class for it'''
def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

# initialize a empty list when Scores is called, append the list with dice scores for every batch,
# at the end of epoch calculates mean of the dice scores
class Scores:
    def __init__(self, phase, epoch):
        self.base_dice_scores = []

    def update(self, targets, outputs):
        probs = outputs
        dice= dice_score(probs, targets)
        self.base_dice_scores.append(dice)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        return dice

# return dice score for epoch when called
def epoch_log(epoch_loss, measure):
    # logging the metrics at the end of an epoch
    dices = measure.get_metrics()
    dice = dices
    print("Loss: %0.4f |dice: %0.4f" % (epoch_loss, dice))
    return dice

'''
4. TRAINING PIPELINE
Create a trainer class by initializing most of the values
'''
class Trainer(object):
    def __init__(self, model):
        # The num_workers attribute tells the data loader instance how many sub-processes to use for data loading.
        # By default, the num_workers value is set to zero, and a value of zero tells the loader to load the data
        # inside the main process. This means that the training process will work sequentially inside the main process.
        self.num_workers = 4
        self.batch_size = {'train':4, 'val':1}
        self.accumulation_steps = 4//self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = 20
        self.phases = ['train', 'val']
        self.best_loss = float('inf')
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model.to(self.device)
        cudnn.benchmark = True # cudnn will look for the optimal set of algorithms for that particular configuration
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, verbose=True)
        self.dataloaders = {phase: CarDataloader(df, img_fol,
                                               mask_fol, mean, std,
                                               phase=phase, batch_size=self.batch_size[phase],
                                               num_workers=self.num_workers) for phase in self.phases}

        self.losses = {phase:[] for phase in self.phases}
        self.dice_score = {phase:[] for phase in self.phases}

    def forward(self, inp_images, tar_mask):
        inp_images = inp_images.to(self.device)
        tar_mask = tar_mask.to(self.device)
        pred_mask = self.net(inp_images)
        loss = self.criterion(pred_mask, tar_mask)
        return loss, pred_mask

    def iterate(self, epoch, phase):
        measure = Scores(phase, epoch)
        print (f"Starting epoch: {epoch} | phase:{phase}")
        batch_size = self.batch_size[phase]
        self.net.train(phase=="train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, mask_target = batch
            # the forward method computes the loss, that is then divided by accumulated steps and finally added to the running loss
            loss, pred_mask = self.forward(images, mask_target)
            loss = loss/self.accumulation_steps
            if phase=='train':
                loss.backward()
                # optimization step and zero the gradient if accumulation steps are reached
                if (itr+1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            pred_mask = pred_mask.detach().cpu()
            measure.update(mask_target, pred_mask)
        epoch_loss = (running_loss*self.accumulation_steps)/total_batches
        dice = epoch_log(epoch_loss, measure)
        self.losses[phase].append(epoch_loss)
        self.dice_score[phase].append(dice)
        torch.cuda.empty_cache()
        return epoch_loss

    '''
    In the start method for every epoch first we will call iterate method for training then iterate method for 
    validation. If the current validation loss is less than previous one then we save the model parameters 
    '''
    def start(self):
        for epoch in range (self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss=self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model_office.pth")
            print ()


# load the UNet architecture, using resnet18 as backbone. Number of classes = 1 as our mask dimension is [1,128,128]
model = smp.Unet("resnet18", encoder_weights="imagenet", classes=1, activation=None)
model_trainer = Trainer(model)
model_trainer.start()

'''
5. INFERENCE PHASE
'''
print('Starting inference...')

# test_dataloader = CarDataloader(df, img_fol, mask_fol, mean, std, 'val', 1, 4)
test_dataloader = CarDataloader(df, img_fol, mask_fol, mean, std, 'val', 1, 4)

ckpt_path = './model_office.pth'
print('Loaded model.')

device = torch.device("cuda")
model = smp.Unet("resnet18", encoder_weights=None, classes=1, activation=None)
model.to(device)
model.eval()
state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])

print('Starting prediction...')
# start prediction
predictions = []

'''
for i, batch in enumerate(tqdm(test_dataloader)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    fig.suptitle('predicted_mask//original_mask')
    plt.ion() # necessary for changing image
    images, mask_target = batch
    batch_preds = torch.sigmoid(model(images.to(device)))
    batch_preds = batch_preds.detach().cpu().numpy()
    ax1.imshow(np.squeeze(batch_preds), cmap='gray')
    ax2.imshow(np.squeeze(mask_target), cmap='gray')
    plt.show()
    plt.pause(5.0)
    #_ = input("Press [enter] to continue.")
    plt.close()
'''

image = Image.open('/home/matteo/Code/SemSeg/carvana/test-128/0a0e3fb8f782_01.jpg')
img = tensorize()(image).unsqueeze(0)
pred = torch.sigmoid(model(img.to(device)))
pred = pred.detach().cpu().numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
fig.suptitle('predicted_mask//original_img')
ax1.imshow(np.squeeze(pred), cmap='gray')
ax2.imshow(image)
plt.show()

print('Completed.')
