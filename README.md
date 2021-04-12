# Implement a Single Class Semantic Segmentation Network from scratch
Code fully commented.
Source: https://medium.com/analytics-vidhya/pytorch-implementation-of-semantic-segmentation-for-single-class-from-scratch-81f96643c98c

Dataset: Carvana (https://www.kaggle.com/c/carvana-image-masking-challenge)
Structure: ./carvana/<train, train_masks, train_masks.csv, test, metadata.csv>

Requisites:
- Python 3.8
- CUDA 11.2
- Pytorch
- CV2 4.5.1
- numpy 1.19.5
- pandas 1.2.3
- torch 1.7.1
- sklearn 0.24.1
- albumentations 0.5.2

Possibile modifications:
- lines 70 - 116: dataset, folders, ...
- lines 220 - 240: trainer parameters
- lines 329 - end: validation set (with groundtruth mask) vs single image from test folder (with original image)
