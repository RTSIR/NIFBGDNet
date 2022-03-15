#import packages
import os

#default number of epochs
epochs=40

#default batch size for training
batch_size=32

#noise-level for training
sigma=25.0  #change it according to noise level in your dataset

#path to generate the data
genDataPath='./data/'

#path to save the genPatches
save_dir='./data/'

#path to training data
data='./data/clean_pats_rgb.npy'

#variables required to generate patch
pat_size=30
stride=10
step=0

