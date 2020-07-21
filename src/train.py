import torch
import logging
import os
from time import time
from torch.utils.tensorboard import SummaryWriter
from torch import optim, nn
from eval import evaluation
from model import TSCN
from preprocess import prepare_batch_input
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
    # item_embedding_matrix = data[8]
    print(args)

    # detect devices
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # tensorboard initialization
    tb_dir = '../runs'
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    writer = SummaryWriter(tb_dir)

    weights_dir = '../weights'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # log
    log_dir = os.path.join('../log', args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'TSCN.log')
    if logging.root.handlers:
        logging.root.handlers = []
    logging.basicConfig(format='%(asctime)s : %(levelname)s: %(message)s',
                        level=logging.INFO,
                        filename=log_file)

    model = TSCN(args, n_item, n_user, adj_item, adj_adam)
    if gpu:
        model.to(device)
    else:
        model.to('cpu')

    # visualization of computation graph
    # input_to = torch.zeros((args.batch_size, args.dim + 1), dtype=torch.float32)
    # if torch.cuda.is_available():
    #     input_to = input_to.cuda()
    # writer.add_graph(model, input_to_model=input_to)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    loss_fn = nn.BCELoss()  # 不带softmax激活函数的CrossEntropy函数

    # train
    print('training ...')
    idx = 0
    # for epoch in range(args.n_epochs):
    for epoch in range(args.n_epochs):
        running_loss = 0
        start_time = time()
        for i in range(len(train_data)):
            optimizer.zero_grad()

            data = train_data[i]
            data = np.array(data)
            np.random.shuffle(data)
            user_list = data[:, 0]
            item_list = data[:, 1]
            labels = torch.from_numpy(data[:, 2])
            # prepare user input
            user_input, n_idxs = prepare_batch_input(user_list, item_list, user2item, n_item)

            if torch.cuda.is_available() and gpu:
                user_input = torch.from_numpy(np.array(user_input)).cuda()
                item_input = torch.from_numpy(np.array(item_list)).cuda()
                n_idx_input = torch.from_numpy(np.array(n_idxs)).cuda()

                outputs = model(user_input, item_input, n_idx_input)
                loss = loss_fn(outputs, labels.cuda().float())
            else:
                user_input = torch.from_numpy(np.array(user_input))
                item_input = torch.from_numpy(np.array(item_list))
                n_idx_input = torch.from_numpy(np.array(n_idxs))

                outputs = model(user_input, item_input, n_idx_input)
                loss = loss_fn(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print('epoch {}: {}/{} loss = {:.4f}'.format(epoch + 1, i, n_user, loss.item()))
        end_time = time()
        # evaluation
        hr, ndcg = evaluation(model, test_data, user2item, args, n_item)
        eval_time = time()
        writer.add_scalar('epoch loss', running_loss / n_user, global_step=epoch + 1)
        writer.add_scalar('hit ratio@10', hr, global_step=epoch + 1)
        writer.add_scalar('NDCG @ 10', ndcg, global_step=epoch + 1)
        print('epocn {} loss:{:.4f}, train_time = [%.1f s], eval_time = [%.1f s]'.format(epoch + 1, running_loss / n_user,
                                                                                         end_time - start_time, eval_time - end_time))
        logging.info('epocn {} loss:{:.4f}, train_time = [%.1f s], eval_time = [%.1f s]'.format(epoch + 1, running_loss / n_user,
                                                                                                end_time - start_time, eval_time - end_time))
    torch.save(model, '../weights/TSCN_' + args.dataset + '.pth')


def load_data(path):
    print('loading data ...')
    user2item = np.load(path + 'user2item.npy', allow_pickle=True).item()
    n_user = len(user2item)
    # users = set(user2item.keys())
    items = list(np.load(path + 'items.npy', allow_pickle=True))
    n_item = len(items)
    # items = list(range(len(items)))
    # n_user = len(users)

    adj_item = np.load(path + 'adj_item.npy', allow_pickle=True)
    adj_adam = np.load(path + 'adj_adam.npy', allow_pickle=True)
    # user2item = np.load('newdata/user2item.npy', allow_pickle=True).item()
    train_data = list(np.load(path + 'train_data.npy', allow_pickle=True))
    test_data = list(np.load(path + 'test_data.npy', allow_pickle=True))
    # item_emb = np.load(path + 'item_emb.npy', allow_pickle=True)
    return n_item, items, n_user, adj_item, adj_adam, train_data, test_data, user2item
