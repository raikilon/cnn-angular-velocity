import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional
import torch.optim
import torch.utils.data
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import numpy as np
import wandb
import os
import datetime
from cnn_regressor import CNNRegressor
from obstacle_position_dataset import ObstaclePositionDataset

parser = argparse.ArgumentParser(description='PyTorch Rotation Training')
parser.add_argument('data', metavar='PATH', help='path to dataset')
parser.add_argument('--epochs', default=1000, type=int,
                    help='number of total epochs to run (early stopping is used to stop before this limit)')
parser.add_argument('--batch_size', default=32, type=int, help='mini-batch size (default: 400)')
parser.add_argument('--patience', default='25', type=int, help='patience of early stopping (default 50)')
parser.add_argument('--feature_extraction', type=int, help='Do feature extraction (train only classifier)')
parser.add_argument('--optimizer', default='SGD', help='model optimizer (default: SGD)')
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate (default: 0.01)')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')


def main():
    args = parser.parse_args()

    if args.feature_extraction == 1:
        feature = True
    else:
        feature = False
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    model = CNNRegressor(2, feature)

    # Multi GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # Send model to GPU or keep it to the CPU
    model = model.to(device=args.device)

    criterion = nn.MSELoss().to(device=args.device)

    if args.optimizer == "ADAM":
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), args.learning_rate,
                         weight_decay=args.weight_decay)
    elif args.optimizer == "ADAGRAD":
        optimizer = Adagrad(filter(lambda p: p.requires_grad, model.parameters()), args.learning_rate,
                            weight_decay=args.weight_decay)
    else:
        optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), args.learning_rate,
                        weight_decay=args.weight_decay)

    dataset = ObstaclePositionDataset(args.data)

    test_size = int((len(dataset) / 100) * 10)
    train_set, val_set, test_set = random_split(dataset, [len(dataset) - (test_size * 2), test_size, test_size])

    # val_size = int((len(train_set) / 100) * 20)
    # train_set, val_set = random_split(dataset, [len(train_set) - val_size, val_size])

    # Final test set will be used only on the final architecture
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                            drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             drop_last=True)

    # Track model in wandb
    wandb.init(project="RoboticsProject", config=args)

    wandb.watch(model)

    args.name = 'model_best_{}.pth.tar'.format(datetime.now().strftime("%d-%b-%Y (%H:%M:%S)"))

    train(model, criterion, optimizer, train_loader, val_loader, args)

    checkpoint = torch.load(args.name)
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint
    torch.cuda.empty_cache()

    test_loss = validate(test_loader, model, criterion, args)
    wandb.run.summary["Test loss"] = test_loss

    os.remove(args.name)


def train(model, criterion, optimizer, train_loader, val_loader, args):
    # Best prediction
    best_loss = np.inf

    epoch_no_improve = 0

    for epoch in range(args.epochs):

        losses = []

        # switch to train mode
        model.train()

        for input_val, target_val in train_loader:
            target_val = target_val.to(device=args.device, non_blocking=True)
            input_val = input_val.to(device=args.device, non_blocking=True)

            output = model(input_val)

            loss = criterion(output, target_val)

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().numpy())
            # TODO define better way to evaluate model ? some kind of accuracy?
            del loss
            torch.cuda.empty_cache()

        val_loss = validate(val_loader, model, criterion, args)

        wandb.log({"Loss": np.mean(losses)}, step=epoch)
        wandb.log({"Val loss": val_loss}, step=epoch)

        #  Save best model and best prediction
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'state_dict': model.state_dict()
            }, args.name)
            epoch_no_improve = 0
        else:
            # Early stopping
            epoch_no_improve += 1
            if epoch_no_improve == args.patience:
                wandb.run.summary["Best val loss"] = best_loss
                wandb.run.summary["Best val loss epoch"] = epoch - args.patience
                return


def validate(test_loader, model, criterion, args):
    losses = []

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for batch_idx, (input_val, target_val) in enumerate(test_loader):
            target_val = target_val.to(device=args.device, non_blocking=True)
            input_val = input_val.to(device=args.device, non_blocking=True)
            output = model(input_val)
            loss = criterion(output, target_val)
            losses.append(loss.detach().cpu().numpy())
            del loss
            torch.cuda.empty_cache()

    return np.mean(losses)


if __name__ == '__main__':
    main()
