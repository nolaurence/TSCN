import argparse
from train import load_data, train
# 超参数设置

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='kaggle', help='set aside')
parser.add_argument('--pooling', type=str, default='adamic', help='which pooling method to use')
parser.add_argument('--n_epochs', type=int, default=10, help=' the number of epochs')
parser.add_argument('--sample_size', type=int, default=3, help='the number of child node of every node')
parser.add_argument('--dim', type=int, default=32, help='number of embedding vector, choose in [8, 16, 32]')
parser.add_argument('--k', type=int, default=3, help='the depth of tree')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-3, help='weight of l2 regularization in 1e-6~1')
parser.add_argument('--lr', type=float, default=1.5e-4, help='learning rate')

args = parser.parse_args()

path = '../data/'
data = load_data(path)
train(args, data=data, show_loss=True)

# evaluation