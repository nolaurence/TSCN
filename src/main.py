import argparse
from train import load_data, train
from preprocess import load_retailrocket, load_movielens
# 超参数设置

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='movielens', help='set aside')
parser.add_argument('--pooling', type=str, default='adamic', help='which pooling method to use')
parser.add_argument('--n_epochs', type=int, default=25, help=' the number of epochs')
parser.add_argument('--sample_size', type=int, default=3, help='the number of child node of every node')
parser.add_argument('--dim', type=int, default=32, help='dimension of embedding vector, choose in [8, 16, 32]')
parser.add_argument('--k', type=int, default=3, help='the depth of tree')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=0, help='weight of l2 regularization in 1e-6~1')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--use_gpu', type=bool, default=True, help='determine whether use gpu to accelerate training')

args = parser.parse_args()


def preprocess(args):
    if args.dataset == 'retailrocket':
        data_dir = '../retailrocket/'
        load_retailrocket(args, data_dir, args.sample_size)
    elif args.dataset == 'movielens':
        data_dir = '../movielens/'
        load_movielens(args, data_dir, args.sample_size)
    else:
        raise Exception('Unknown dataset: ' + args.dataset)


# how to run:
# firstly, comment load_data, train, test to run preoprocess
# then, uncomment load_data, train and comment preprocess to train the model
# finally, comment train and uncomment test to run evaluation

# preprocess
# preprocess(args)

# train
if args.dataset == 'retailrocket':
    path = '../retailrocket/'
elif args.dataset == 'movielens':
    path = '../movielens/'
else:
    raise Exception('Unknown dataset: ' + args.dataset)
data = load_data(path)
train(args, data=data, gpu=args.use_gpu)
