import numpy as np
import time
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.nn.modules.distance import PairwiseDistance
from losses.triplet_loss import TripletLoss
from dataset.TripletLossDataset import TripletFaceDataset
from tqdm import tqdm
from models.mobilenetv2_tripletloss import MobileNetV2Triplet

parser = argparse.ArgumentParser(description="Training a FaceNet facial recognition model using Softmax.")
parser.add_argument('--dataroot', '-d', type=str, required=True,
                    help="(REQUIRED) Absolute path to the training dataset folder"
                    )
parser.add_argument('--lfw', type=str, required=True,
                    help="(REQUIRED) Absolute path to the labeled faces in the wild dataset folder"
                    )
parser.add_argument('--training_dataset_csv_path', type=str, default='datasets/webface.csv',
                    help="Path to the csv file containing the image paths of the training dataset"
                    )
parser.add_argument('--epochs', default=100, type=int,
                    help="Required training epochs (default: 100)"
                    )
parser.add_argument('--iterations_per_epoch', default=5000, type=int,
                    help="Number of training iterations per epoch (default: 5000)"
                    )
parser.add_argument('--embedding_dimension', default=512, type=int,
                    help="Dimension of the embedding vector (default: 512)"
                    )
parser.add_argument('--num_human_identities_per_batch', default=32, type=int,
                    help="Number of set human identities per generated triplets batch. (Default: 32)."
                    )
parser.add_argument('--batch_size', default=128, type=int,
                    help="Batch size (default: 544)"
                    )
parser.add_argument('--num_workers', default=4, type=int,
                    help="Number of workers for data loaders (default: 4)"
                    )
parser.add_argument('--optimizer', type=str, default="adagrad", choices=["sgd", "adagrad", "rmsprop", "adam"],
                    help="Required optimizer for training the model: ('sgd','adagrad','rmsprop','adam'), (default: 'adagrad')"
                    )
parser.add_argument('--learning_rate', default=0.075, type=float,
                    help="Learning rate for the optimizer (default: 0.075)"
                    )
parser.add_argument('--image_size', default=140, type=int,
                    help='Input image size (default: 140 (140x140))'
                    )
parser.add_argument('--use_semihard_negatives', default=False, type=bool,
                    help="If True: use semihard negative triplet selection. Else: use hard negative triplet selection (Default: False)"
                    )
args = parser.parse_args()

def set_model_architecture(model_architecture, pretrained, embedding_dim):
    model = MobileNetV2Triplet(
        embedding_dim=embedding_dim,
        pretrain= pretrained
    )
    print("Using {} model architecture.".format(model_architecture))

    return model

def set_model_gpu_mode(model):
    flag_train_gpu = torch.cuda.is_available()
    flag_train_multi_gpu = False

    if flag_train_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()
        flag_train_multi_gpu = True
        print('Using multi-gpu training.')

    elif flag_train_gpu and torch.cuda.device_count() == 1:
        model.cuda()
        print('Using single-gpu training.')

    return model, flag_train_multi_gpu

def set_optimizer(optimizer, model, learning_rate):
    if optimizer == "sgd":
        optimizer_model = optim.SGD(
            params=model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            dampening=0,
            nesterov=False,
            weight_decay=1e-5
        )

    elif optimizer == "adagrad":
        optimizer_model = optim.Adagrad(
            params=model.parameters(),
            lr=learning_rate,
            lr_decay=0,
            initial_accumulator_value=0.1,
            eps=1e-10,
            weight_decay=1e-5
        )

    elif optimizer == "rmsprop":
        optimizer_model = optim.RMSprop(
            params=model.parameters(),
            lr=learning_rate,
            alpha=0.99,
            eps=1e-08,
            momentum=0,
            centered=False,
            weight_decay=1e-5
        )

    elif optimizer == "adam":
        optimizer_model = optim.Adam(
            params=model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=False,
            weight_decay=1e-5
        )

    return optimizer_model


def main():
    dataroot = args.dataroot
    training_dataset_csv_path = args.training_dataset_csv_path
    epochs = args.epochs
    iterations_per_epoch = args.iterations_per_epoch
    model_architecture = args.model_architecture
    pretrained = args.pretrained
    embedding_dimension = args.embedding_dimension
    num_human_identities_per_batch = args.num_human_identities_per_batch
    batch_size = args.batch_size
    resume_path = args.resume_path
    num_workers = args.num_workers
    optimizer = args.optimizer
    learning_rate = args.learning_rate
    image_size = args.image_size
    start_epoch = 0

    # Define image data pre-processing transforms
    #   ToTensor() normalizes pixel values between [0, 1]
    #   Normalize(mean=[0.6071, 0.4609, 0.3944], std=[0.2457, 0.2175, 0.2129]) according to the calculated glint360k
    #   dataset with tightly-cropped faces dataset RGB channels' mean and std values by
    #   calculate_glint360k_rgb_mean_std.py in 'datasets' folder.
    data_transforms = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6071, 0.4609, 0.3944],
            std=[0.2457, 0.2175, 0.2129]
        )
    ])

    # Instantiate model
    model = set_model_architecture(
        model_architecture=model_architecture,
        pretrained=pretrained,
        embedding_dimension=embedding_dimension
    )

    # Load model to GPU or multiple GPUs if available
    model, flag_train_multi_gpu = set_model_gpu_mode(model)
    # Re-instantiate training dataloader to generate a triplet list for this training epoch
    train_dataloader = torch.utils.data.DataLoader(
        dataset= datasets.ImageFolder(dataroot, transforms = data_transforms),
        batch_size=batch_size,
        shuffle=False  # Shuffling for triplets with set amount of human identities per batch is not required
    )


    # set losses softmax
    criterion = nn.CrossEntropyLoss()
    # Set optimizer
    optimizer_model = set_optimizer(
        optimizer=optimizer,
        model=model,
        learning_rate=learning_rate
    )

    # Resume from a model checkpoint
    if resume_path:
        if os.path.isfile(resume_path):
            print("Loading checkpoint {} ...".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch'] + 1
            optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])

            # In order to load state dict for optimizers correctly, model has to be loaded to gpu first
            if flag_train_multi_gpu:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded: start epoch from checkpoint = {}".format(start_epoch))
        else:
            print("WARNING: No checkpoint found at {}!\nTraining from scratch.".format(resume_path))

    start_epoch = start_epoch

    print(f"Training using softmax starting for {epochs - start_epoch} epochs:\n")

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        train_total_loss = 0.0
        total = 0
        since = time.time()
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            train_total_loss += loss.item() * batch_size
            total += batch_size
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        train_total_loss = train_total_loss / total
        time_elapsed = time.time() - since

        print(f'total_loss: {train_total_loss:.4f} time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # Save model checkpoint
        state = {
            'epoch': epoch,
            'embedding_dimension': embedding_dimension,
            'batch_size_training': batch_size,
            'model_state_dict': model.state_dict(),
            'model_architecture': model_architecture,
            'optimizer_model_state_dict': optimizer_model.state_dict(),
        }

        # For storing data parallel model's state dictionary without 'module' parameter
        if flag_train_multi_gpu:
            state['model_state_dict'] = model.module.state_dict()

        # Save model checkpoint
        torch.save(state, 'model_training_checkpoints/model_{}_triplet_epoch_{}.pt'.format(
                model_architecture,
                epoch
            )
        )


if __name__ == '__main__':
    main()