import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from PIL import Image
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from functools import lru_cache
import os
import time

import copy

import requests
import zipfile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def download_data(url, folder="data"):
    
    if "data.zip" not in os.listdir():
        r = requests.get(url)

        with open("data.zip", "wb") as f:
            f.write(r.content)
            
        with zipfile.ZipFile("data.zip", "r") as zip_ref:
            zip_ref.extractall(folder)

def get_dataloaders(path):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    data_dir = path
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=4, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    return (dataloaders, dataset_sizes, class_names)


def train_model(model, criterion, optimizer, scheduler, loaders, num_epochs=10):
    dataloaders, dataset_sizes, class_names = loaders

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "best_model.pth")
    return model


def load_pretrained_model():
    model_conv = torchvision.models.resnet18(pretrained=True)

    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    
    model_conv = model_conv.to(device)
    """
    if weights:
        pass
    """
    return model_conv


def main(args):
    print(args)

    if args.download_data:
        print("downloading data")
        download_data()

    if args.train:
        
        if args.epochs:
            epochs = args.epochs
        else:
            epochs = 2

        if args.optimizer:
            optimizer = args.optimizer
        else:
            optimizer = "sgd"

        model_conv = load_pretrained_model()
        #model_conv = load_pretrained_model(args.load_weights)

        # load data into dataloader, get dataset size and classes 
        loaders = get_dataloaders(args.data_path)
        print(loaders)

        criterion = nn.CrossEntropyLoss()
        # choose optimizer by args
        if optimizer == "adam":
            optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01, esp=1e-5)
        elif optimizer == "sgd":
            optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=3, gamma=0.1)

        start = time.perf_counter()

        model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, loaders, epochs)

        total = time.perf_counter() - start

        print(f"Total time {total}")


    if args.epochs:
        if not args.train:
            print("You will need to pass the '--train' argument to train the model")
            exit()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data_path", help="The path of the downloaded data", type=str)
    parser.add_argument("--epochs", help="The number of epochs your model should be trained.", type=int)
    parser.add_argument("--optimizer", help="The optimizer you would like to use.", type=str)
    parser.add_argument("--load-weights", action="store_true")

    args = parser.parse_args()
    main(args)
    
    # --download-data
    # --train --epochs INT (it should have a default)
    # --optimizer STRING
    # --load-weights STRING
    
    # extra otion
    # load the data URL from an environment variable

# python3 train.py --download-data --train --epochs 3 --optimizer sgd --load-weights my_weigths.pth