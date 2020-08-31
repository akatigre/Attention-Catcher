#!/usr/bin/env python3

import torch.utils.data
import wandb
from torch import optim
import utils
from model import Net
from train import train, evaluate
from data import data_balancer
import numpy as np
import os

architecture = "CNN"
dataset_id = "cifar-10"
BATCH_SIZE = 64
SAMPLE_SIZE = 1000
EARLY_STOPPING = 10
N_EPOCHS = 100
PATH = "/content/gdrive/My Drive/SeqBoost-image/"
P = 5
M = 0.1
E = "1:7"


config = dict(
    learning_rate=0.001,
    weight_decay=0.00001,
    epoch=N_EPOCHS,
    momentum=0.9,
    architecture="CNN",
    dataset_id="cifar-10",
    batch_size=BATCH_SIZE,
    sample_size=SAMPLE_SIZE,
    early_stopping=EARLY_STOPPING
)

if __name__ == '__main__':

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset, train_loader, valid_loader, test_loader, trainA_loader, validA_loader, trainB_loader, validB_loader = data_balancer(batch_size=BATCH_SIZE)
    loaders = [train_loader, valid_loader, test_loader, trainA_loader, trainB_loader, validA_loader, validB_loader]
    names = ['train_loader','valid_loader', 'test_loader',"trainA_loader", "trainB_loader", "validA_loader", "validB_loader"]
    for loader, name in zip(loaders, names):
        train_iter = iter(loader)
        for _ in range(5):
            _, target = train_iter.next()
            print(f'{name}', ': Classes {}, counts: {}'.format(
                *np.unique(target.numpy(), return_counts=True)))

    ##############################
    #########Seq Boost############
    ##############################
    
    model = Net()
    utils.xavier_initialize(model)
    model = model.to(device)

    for name, param in model.named_parameters():
        if param.device.type != 'cuda':
            print('param {}, not on GPU'.format(name))

    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
    mid_model, train_loss, valid_loss = train(model, trainA_loader, validA_loader, batch_size=BATCH_SIZE,
                                              wandb_log=False, patience=EARLY_STOPPING, consolidate=False,
                                              n_epochs=config['epoch'])
    wandb.init(
        project='CIFAR10',
        config=config,
        name='SeqBoost(EWC) p={} mu={} eta={} sample size = {}'.format(P, M, E, SAMPLE_SIZE))
    model, train_loss, valid_loss = train(mid_model, trainB_loader, validB_loader, batch_size = BATCH_SIZE, fisher_estimation_sample_size = SAMPLE_SIZE,patience = EARLY_STOPPING, consolidate = True, n_epochs=config['epoch'])

    model.load_state_dict(torch.load(os.path.join(PATH, 'checkpoint.pt')))
    evaluate(model, test_loader, batch_size = BATCH_SIZE)

    ##############################
    #########Base Line############
    ##############################
    # model = Net()
    # utils.xavier_initialize(model)
    # model = model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # wandb.init(
    #     project='CIFAR10',
    #     config=config,
    #     name="Baseline  p={} mu={} eta={}".format(P,M,E))

    # model, train_loss, valid_loss = train(model, train_loader, valid_loader, batch_size=BATCH_SIZE, wandb_log=True,
    #                                       consolidate=False, patience=EARLY_STOPPING, n_epochs=config['epoch'])
    # evaluate(model, test_loader, batch_size = BATCH_SIZE)

