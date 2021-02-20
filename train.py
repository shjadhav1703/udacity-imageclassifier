# Imports here
import numpy as np
import pandas as pd
import time
import os
from os.path import isdir
# import pyplot & torch
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets, transforms, utils, models
from collections import OrderedDict
from torch import nn, optim
import torch.nn.functional as F
import argparse


def arg_pass():
    args = argparse.ArgumentParser(description='Train the Image classifier model')
    args.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args.add_argument('--save_dir', dest="save_dir", action="store", default=os.getcwd())
    args.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    args.add_argument('--epochs', dest="epochs", action="store", type=int, default=5)
    args.add_argument('--arch', dest="arch", action="store", default="vgg16", type=str)
    args.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)
    # parse args
    pass_args = args.parse_args()
    return pass_args

def load_data():
    # TODO: Define your transforms for the training, validation, and testing sets
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    #
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    return train_dataloaders, valid_dataloaders, test_dataloaders, train_datasets

def initial_classifier(model, hidden_units):
    if type(hidden_units) == type(None):
        hidden_units = 4096

    #
    input_features = model.classifier[0].in_features

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, 102, bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    return classifier


def validation(model, test_dataloaders, criterion, device):
    # function for validation pass to calculate the loss & accuracy
    test_loss = 0
    accuracy = 0

    for ii, (inputs, labels) in enumerate(test_dataloaders):
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

def train_network_model(model, train_dataloaders, valid_dataloaders, device,
                                      criterion, optimizer, epochs):
    # Train classifier layer
    steps = 0
    print_every = 30
    print("Training process started..!")
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        for ii, (inputs, labels) in enumerate(train_dataloaders):
            steps += 1
            # move input , labels to default device
            inputs, labels = inputs.to(device), labels.to(device)
            # reset optimizer
            optimizer.zero_grad()

            # forward pass
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            # backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    # calculate loss  & accuracy using validation data
                    valid_loss, accuracy = validation(model, valid_dataloaders, criterion, device)

                print("Epoch: {}/{} | ".format(epoch + 1, epochs),
                      "Training Loss: {:.3f} | ".format(running_loss / print_every),
                      "Validation Loss: {:.3f} | ".format(valid_loss / len(valid_dataloaders)),
                      "Validation Accuracy: {:.3f}".format(accuracy / len(valid_dataloaders)))
                #
                running_loss = 0
                model.train()
    print("\nTraining process completed..!")
    return model

def test_model(model, test_dataloaders, device):
    # validate the train model
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in test_dataloaders:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def pre_train_model(model_arch):
    if model_arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = model_arch
    elif model_arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.name = model_arch
    else:
        print("Model architecture is not valid model")

    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    return model

def create_checkpoint(model,train_data, save_dir):
    # save model
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'architecture': model.name,
                      'classifier': model.classifier,
                      'class_to_idx': model.class_to_idx,
                      'state_dict': model.state_dict()}

    torch.save(checkpoint, 'img_classifier_checkpoint.pth')
    print("Save  checkpoint completed")

# main_prog function
def main_prog():
    # command line arguments
    args = arg_pass()
    # Load data using image dataset
    train_dataloaders, valid_dataloaders, test_dataloaders, train_data = load_data()
    # Load pre-train model
    model = pre_train_model(model_arch=args.arch )

    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    model.classifier = initial_classifier(model, hidden_units=args.hidden_units)
    # check for cuda\cpu
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device);
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    # Train the classifier
    trained_model = train_network_model(model, train_dataloaders, valid_dataloaders, device, criterion, optimizer, args.epochs)

    # Test your trained_model network and save model to the checkpoint
    test_model(trained_model, test_dataloaders, device)
    # save the model
    create_checkpoint(trained_model,train_data, args.save_dir)


# Project : train an image classifier to recognize different species of flowers
# Run the main program to train model
if __name__ == '__main__':
    main_prog()
