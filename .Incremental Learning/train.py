from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import torch
import torch.utils.data
import torch.utils.data
from torch.nn import functional as F
from torch.autograd import Variable
import wandb
from torch import nn
from torch import optim
from pytorchtools import EarlyStopping
from tqdm import tqdm


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()
        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = variable(input)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, valid_loader, batch_size, fisher_estimation_sample_size=100, wandb_log = True, consolidate = False, nb_class = 10, patience = 10, n_epochs =100, lr = 1e-3, weight_decay = 1e-5):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        # data_stream = tqdm(enumerate(train_loader, 1))
        for batch, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data_size = len(data)
            dataset_size = len(train_loader.dataset)
            dataset_batches = len(train_loader)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            criterion = nn.CrossEntropyLoss()
            ce_loss = criterion(output, target)
            e_loss = 0
            if consolidate:
                e_loss = model.ewc_loss()
            loss = ce_loss + e_loss
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

            # Predictions
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # Calculate global accuracy
        try: 
          accuracy = 100 * correct / total
          if wandb_log:
            wandb.log({"accuracy": accuracy})

        except: pass

        if consolidate:
            model.consolidate(model.estimate_fisher(
                train_loader, fisher_estimation_sample_size, batch_size
            ))

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        correct = 0
        total = 0
        confusion_matrix = torch.zeros(nb_class, nb_class)
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())
            # Predicted results
            _, preds = torch.max(output, 1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            for t, p in zip(target, preds):
                confusion_matrix[t, p] += 1

        # Model Performance Statistics
        accuracy = 100 * correct / total
        class_correct = confusion_matrix.diag()
        class_total = confusion_matrix.sum(1)
        class_accuracies = class_correct / class_total

        TP = np.zeros(nb_class)
        TN = np.zeros(nb_class)
        FP = np.zeros(nb_class)
        FN = np.zeros(nb_class)
        precision = np.zeros(nb_class)
        recall = np.zeros(nb_class)
        f1 = np.zeros(nb_class)

        # Normalize confusion matrix
        for i in range(nb_class):
            TP[i] = confusion_matrix[i][i]
            TN[i] = confusion_matrix[-i][-i]
            FP[i] = confusion_matrix[i][-i]
            FN[i] = confusion_matrix[-i][i]
            precision[i] = TP[i] / (TP[i] + FP[i])
            recall[i] = TP[i] / (TP[i] + FN[i])
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
            confusion_matrix[i] = confusion_matrix[i] / confusion_matrix[i].sum()

        # print training/validation statistics
        # calculate average loss over an epoch

        print("f1 vector : ", f1)
        macro_f1 = reduce(lambda a, b: a + b, f1) / len(f1) * 100
        macro_f1 = reduce(lambda a, b: a + b, f1) / len(f1) * 100
        precision = reduce(lambda a, b: a + b, precision) / len(precision) * 100
        recall = reduce(lambda a, b: a + b, recall) / len(recall) * 100
        
        print("Macro f1 score is {:.3f}%".format(macro_f1))
        print('Accuracy of the model {:.3f}%'.format(accuracy))

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)
        if wandb_log:
            wandb.log({"Precision(valid)": precision, "Recall(valid)": recall, "F1 Score(valid)": macro_f1,
                   "Accuracy(valid)": accuracy}, step=epoch)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model

    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, avg_train_losses, avg_valid_losses


def evaluate(model, test_loader, batch_size, wandb_log = True, nb_classes = 10, class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')):
    model.eval()
    correct_list = list(0. for i in range(10))
    total_list = list(0. for i in range(10))
    correct = 0
    total = 0
    test_loss = 0.0
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for data, target in test_loader:
            if len(target.data) != batch_size:
                break
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += batch_size
            correct += (predicted == target).sum().item()
            for i in range(batch_size):
                label = target.data[i]
            for t, p in zip(target, predicted):
                confusion_matrix[t, p] += 1

    # Calculate global accuracy
    if wandb_log:
        try:
          accuracy = 100 * correct / total
          wandb.log({"accuracy": accuracy})
        except:
          pass

    # F1 score 직접 계산하기
    print(confusion_matrix)
    class_correct = confusion_matrix.diag()
    class_total = confusion_matrix.sum(1)
    class_accuracies = class_correct / class_total

    TP = np.zeros(nb_classes)
    TN = np.zeros(nb_classes)
    FP = np.zeros(nb_classes)
    FN = np.zeros(nb_classes)
    precision = np.zeros(nb_classes)
    recall = np.zeros(nb_classes)
    f1 = np.zeros(nb_classes)

    # Normalize confusion matrix
    print("confusion_matrix", confusion_matrix)
    for i in range(nb_classes):
        TP[i] = confusion_matrix[i][i]
        TN[i] = confusion_matrix[-i][-i]
        FP[i] = confusion_matrix[i][-i]
        FN[i] = confusion_matrix[-i][i]
        precision[i] = TP[i] / (TP[i] + FP[i])
        recall[i] = TP[i] / (TP[i] + FN[i])
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        confusion_matrix[i] = confusion_matrix[i] / confusion_matrix[i].sum()

    # Print statistics
    macro_f1 = reduce(lambda a, b: a + b, f1) / len(f1) * 100
    precision = reduce(lambda a, b: a + b, precision) / len(precision) * 100
    recall = reduce(lambda a, b: a + b, recall) / len(recall) * 100
    print("Macro f1 score is {:.3f}%".format(macro_f1))
    try:
      print('Accuracy of the model {:.3f}%'.format(accuracy))
    except:
      pass
    for i in range(nb_classes):
        print('Accuracy for {}: {:.3f}%'.format(
            class_names[i], 100 * class_accuracies[i]))
        if wandb_log:
            wandb.log({f"Accuracy of class {class_names[i]}": class_accuracies[i] * 100})
    if wandb_log:
        try:
            wandb.log(
              {"Precision(test)": precision, "Recall(test)": recall, "F1 Score(test)": macro_f1, "Accuracy(test)": accuracy})
        except:
            wandb.log(
              {"Precision(test)": precision, "Recall(test)": recall, "F1 Score(test)": macro_f1})
        


    # Plot confusion matrix
    f = plt.figure()
    ax = f.add_subplot(111)
    cax = ax.imshow(confusion_matrix.numpy(), interpolation='nearest')
    f.colorbar(cax)
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)
    if wandb_log:
        wandb.log({"Confusion Matrix": plt})
