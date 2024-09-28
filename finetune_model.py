
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from functools import partial
import time
from tqdm import tqdm

import argparse
import matplotlib.pyplot as plt
import matplotlib

import myutils
import Mymodel

from A_CLIP.losses import ACLIPLoss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main(args):

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    print("cuda available:", use_cuda)
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")

    # Get data loader
    train_loader, val_loader, test_loader = myutils.getdata(batch_size=args.batch_size, m_type=args.type, data_path=args.data_dir)

    model = Mymodel.load_model(args.type, device)

    # Observe that all parameters are being optimized
    if args.type == 'clip':
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    elif args.type == 'aclip':
        '''
        loss_function = ACLIPLoss(0.1).to(device)
        p_wd, p_non_wd = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)

        optim_params = [{"params": p_wd, "weight_decay": 0.1},
                        {"params": p_non_wd, "weight_decay": 0},
                        ]

        optimizer = torch.optim.AdamW(optim_params, lr=args.learning_rate, betas=(0.9, 0.98),
                                     eps=1e-8, weight_decay=0.1) 
        '''
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    elif args.type == 'aclip_rand':
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    elif args.type == 'test_aclip':
        loss_function = nn.CrossEntropyLoss().to(device)
        init_lr = args.learning_rate * int(args.batch_size) / 256
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        #assert len(parameters) == 2  # weight, bias

        optimizer = torch.optim.SGD(parameters, init_lr,
                                    momentum=0.9,
                                    weight_decay=0)
    elif args.type == 'siglip':
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        loss_function = AverageMeter()
        #loss_function = nn.BCEWithLogitsLoss()
    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    best_acc = 0.0
    train_loss = []
    val_loss = []
    for epoch in range(args.num_epochs):

        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)
        if args.type == 'test_aclip':
            train_epoch_loss, train_epoch_acc = myutils.train_aclip_model(model, train_loader, optimizer, loss_function, device, args.type)
        else:
            train_epoch_loss, train_epoch_acc = myutils.train_model(model, train_loader, optimizer, loss_function, device, args.type)
        train_loss.append(train_epoch_loss)
        val_epoch_loss, val_epoch_acc = myutils.val_model(model, val_loader, loss_function, device, args.type)
        val_loss.append(val_epoch_loss)
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), f'{args.log_dir}/{args.num_epochs}_{args.type}_best_model.pth') 

    try:
        # Plot the loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, args.num_epochs + 1), train_loss, marker='o', linestyle='-', color='b')
        plt.plot(range(1, args.num_epochs + 1), val_loss, marker='o', linestyle='-', color='g')
        plt.title(f'Training and validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.grid(True)
        plt.savefig(f"{args.log_dir}/{args.num_epochs}_{args.type}_vitb16_loss.png")
        # plt.show()
        plt.close()

    except Exception as e:
        print(f"Error plotting acc and loss E: {e}")

    # run test data
    
    #best_model = myutils.myDinoV2(os.path.join(args.log_dir, 'best_model.pth'))
    test_loss, test_acc = myutils.test_model(model, test_loader, loss_function, device, args.type)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune CLIP on human action')
    parser.add_argument('--type', default='aclip_rand',
                        help='model type')
    parser.add_argument('--batch_size', '-b', default=4096, type=int, metavar='N',
                        help='mini-batch size (default: 128)')
    parser.add_argument('--log_dir', default='./myoutput', type=str, metavar='PATH',
                        help='path to directory where to log')
    parser.add_argument('--data_dir', default="/home/cap6411.student1/CVsystem/assignment/hw5/human-action-recognition-dataset/Structured/", type=str,
                        help='path to the dataset')
    parser.add_argument('--learning_rate',
                        type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=10,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--weight_path', 
                        default='./pretrain/dinov2_vits14_pretrain.pth',
                        type=str,
                        help='path to the model weight')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    main(args)

