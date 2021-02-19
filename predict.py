# Imports here
import numpy as np
import pandas as pd
import time
import os
from os.path import isdir
import json
# import pyplot & torch
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets, transforms, utils, models
from collections import OrderedDict
from torch import nn, optim
import torch.nn.functional as F
import argparse


def args_pass():
    args = argparse.ArgumentParser(description='predict Network settings')
    args.add_argument('input_img', default='/home/workspace/ImageClassifier/flowers/test/1/image_06752.jpg', nargs='*', action="store",
                    type=str)
    args.add_argument('checkpoint', default='/home/workspace/ImageClassifier/img_classifier_checkpoint.pth', nargs='*', action="store",
                    type=str)
    args.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)

    # parse args
    pass_args = args.parse_args()
    return pass_args


def load_checkpoint_rebuild_model():
    """
    Load checkpoint and rebuild model
    """
    # Load the saved checkpoint file
    checkpoint = torch.load("img_classifier_checkpoint.pth")
    # Download pretrained model
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    # Load from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(input_img):
    test_image = Image.open(input_img)
    #
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = adjustments(test_image)

    return img_tensor

def predict_image(image_tensor, model, cat_to_name):
    # function predict image
    #
    top_k = 5
    # mode eval & convert image numpy to torch
    model.eval()
    torch_image = torch.from_numpy(np.expand_dims(image_tensor, axis=0)).type(torch.FloatTensor)
    model = model.cpu()
    # Find probabilities
    log_probs = model.forward(torch_image)
    linear_probs = torch.exp(log_probs)
    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)

    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]

    # Convert to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]

    return top_probs, top_labels, top_flowers


# Function for the main program to predict model
def main_prog_predict():
    # define command line arguments for prediction
    args = args_pass()
    # Load the category json file
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Using checkpoint from train.py, load the trained model
    model = load_checkpoint_rebuild_model()

    # process image
    image_tensor = process_image(args.input_img)

    # check for cuda\cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device);

    # Use processed image to predict, A common practice is to predict the top 5 or so (usually called top-K) most probable classes
    top_probs, top_labels, top_flowers = predict_image(image_tensor, model, cat_to_name)
    # Print the probabilities
    print('Top flower', top_flowers)
    print('Top prob', top_probs)


# Project : train an image classifier to recognize different species of flowers
# Run the main program to predict model
if __name__ == '__main__':
    main_prog_predict()
