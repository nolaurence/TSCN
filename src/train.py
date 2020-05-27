from model import TSCN
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data as Data
from torch import optim, nn
import numpy as np
np.random.seed(1)
torch.manual_seed(7)

def train(args, data, gpu):
    n_item = data[0]
    items = data[1]
    n_user = data[2]
    adj_item, adj_adam = data[3], data[4]
    train_data, test_data = data[5], data[6]
    user2item = data[7]

    # detect devices
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # tensorboard initialization
    writer = SummaryWriter('../runs')

    # load data
    train_loader = Data.DataLoader(train_data, args.batch_size, True)
    test_loader = Data.DataLoader(test_data, args.batch_size, True)

    model = TSCN(args, n_item, n_user, adj_item, adj_adam, user2item)
    if gpu:
        model.to(device)
    else:
        model.to('cpu')

    # summary(model, input_size=(args.batch_size, 2))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    loss_fn = nn.CrossEntropyLoss()  # 不带softmax激活函数的CrossEntropy函数

    # train
    print('training ...')
    idx = 0
    for epoch in range(args.n_epochs):

        running_loss = 0
        n_batch = 0
        for i, data in enumerate(train_loader):
            idx += 1
            optimizer.zero_grad()
            inputs, labels = data

            # if epoch == 0 and i == 0:
            #     if torch.cuda.is_available() and gpu:
            #         writer.add_graph(model, input_to_model=inputs.cuda(), verbose=False)
            #     else:
            #         writer.add_graph(model, input_to_model=inputs, verbose=False)
            if inputs.shape[0] < args.batch_size:
                continue
            if torch.cuda.is_available() and gpu:
                outputs = model(inputs.cuda())
                loss = loss_fn(outputs, labels.cuda().long())
            else:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batch += 1

            # loss visualization
            # writer.add_scalar('loss', loss.item(), global_step=idx)
            print('\r', 'epoch {} batch {} loss: {:.4f}'.format(epoch + 1, i, loss.item()), end='')
        writer.add_scalar('epoch loss', running_loss / n_batch, global_step=epoch + 1)
        print('\nepocn {} loss:{:.4f}'.format(epoch + 1, running_loss / n_batch))

    torch.save(model, '../weights/TSCN_' + args.dataset + '.pth')


def load_data(path):
    print('loading data ...')
    user2item = np.load(path + 'user2item.npy', allow_pickle=True).item()
    n_user = len(user2item)
    # users = set(user2item.keys())
    items = list(np.load(path + 'items.npy', allow_pickle=True))
    n_item = len(items)
    items = list(range(len(items)))
    # n_user = len(users)

    adj_item = np.load(path + 'adj_item.npy', allow_pickle=True)
    adj_adam = np.load(path + 'adj_adam.npy', allow_pickle=True)
    # user2item = np.load('newdata/user2item.npy', allow_pickle=True).item()
    train_data = train_set(path)
    test_data = test_set(path)
    return n_item, items, n_user, adj_item, adj_adam, train_data, test_data, user2item


def load_train(path):
    train_data = np.load(path + 'train_data.npy', allow_pickle=True)
    train_x = train_data[:,0:2]
    train_y = train_data[:, 2]
    return train_x, train_y


class train_set(Data.Dataset):
    def __init__(self, path, loader=load_train):
        self.x, self.y = loader(path)

    def __getitem__(self, item):
        data = self.x[item]
        label = self.y[item]
        return data, label

    def __len__(self):
        return self.x.shape[0]


def load_test(path):
    test_data = np.load(path + 'test_data.npy', allow_pickle=True)
    test_x = test_data[:, 0:2]
    test_y = test_data[:, 2]
    return test_x, test_y


class test_set(Data.Dataset):
    def __init__(self, path, loader=load_test):
        self.x, self.y = loader(path)

    def __getitem__(self, item):
        data = self.x[item]
        label = self.y[item]
        return data, label

    def __len__(self):
        return self.x.shape[0]
