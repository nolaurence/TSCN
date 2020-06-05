import argparse
import torch
from torch.utils import data as Data
from eval import get_user_record, evaluation
from train import load_data, train
from preprocess import load_retailrocket, load_movielens
# 超参数设置

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='retailrocket', help='set aside')
parser.add_argument('--pooling', type=str, default='adamic', help='which pooling method to use')
parser.add_argument('--n_epochs', type=int, default=10, help=' the number of epochs')
parser.add_argument('--sample_size', type=int, default=3, help='the number of child node of every node')
parser.add_argument('--dim', type=int, default=32, help='dimension of embedding vector, choose in [8, 16, 32]')
parser.add_argument('--k', type=int, default=3, help='the depth of tree')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=0, help='weight of l2 regularization in 1e-6~1')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--use_gpu', type=bool, default=True, help='determine whether use gpu to accelerate training')

args = parser.parse_args()

def test(args):
    model_path = '../weights/TSCN_' + args.dataset + '.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_data, test_data = data[5], data[6]
    model = torch.load(model_path)
    model.to(device)
    train_loader = Data.DataLoader(train_data, args.batch_size, True)
    test_loader = Data.DataLoader(test_data, args.batch_size, True)
    user_record = get_user_record(test_loader)
    # print(1052 in user_record.keys())
    hit_ratio, ndcg_score = evaluation(model, test_loader, user_record, args)
    print('HR:{:.4f} NDCG:{:.4f}'.format(hit_ratio, ndcg_score))


def preprocess(args):
    if args.dataset == 'retailrocket':
        data_dir = '../retailrocket/'
        load_retailrocket(data_dir, args.sample_size)
    elif args.dataset == 'movielens':
        data_dir = '../movielens/'
        load_movielens(data_dir, args.sample_size)
    else:
        raise Exception('Unknown dataset: ' + args.dataset)


# how to run:
# firstly, comment load_data, train, test to run preoprocess
# then, uncomment load_data, train and comment preprocess to train the model
# finally, comment train and uncomment test to run evaluation

# preprocess
preprocess(args)

# train
if args.dataset == 'retailrocket':
    path = '../retailrocket/'
elif args.dataset == 'movielens':
    path = '../movielens/'
else:
    raise Exception('Unknown dataset: ' + args.dataset)
data = load_data(path)
train(args, data=data, gpu=args.use_gpu)

# evaluation
test(args)
