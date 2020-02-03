import numpy as np
import tensorflow as tf
import argparse
import math
from abc import abstractmethod
from sklearn.metrics import f1_score, roc_auc_score
np.random.seed(1)
# tf.enable_eager_execution()

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='kaggle', help='set aside')
parser.add_argument('--pooling', type=str, default='adamic', help='which pooling method to use')
parser.add_argument('--n_epochs', type=int, default=10, help=' the number of epochs')
parser.add_argument('--sample_size', type=int, default=3, help='the number of child node of every node')
parser.add_argument('--dim', type=int, default=32, help='number of embedding vector, choose in [8, 16, 32]')
parser.add_argument('--k', type=int, default=3, help='the depth of tree')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization in 1e-6~1')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')

args = parser.parse_args()


# TSCN(args, n_users, n_items, adj_item, adj_adam, user2item_dict)
def load_data():
    user2item = np.load('newdata/user2item.npy', allow_pickle=True).item()
    users = set(user2item.keys())
    items = list(np.load('newdata/items.npy', allow_pickle=True))
    n_item = len(items)
    items = list(range(len(items)))
    n_user = len(users)

    adj_item = np.load('newdata/adj_item.npy', allow_pickle=True)
    adj_adam = np.load('newdata/adj_adam.npy', allow_pickle=True)
    # user2item = np.load('newdata/user2item.npy', allow_pickle=True).item()
    train_data = np.load('newdata/traindata.npy', allow_pickle=True)
    test_data = np.load('newdata/testdata.npy', allow_pickle=True)
    return n_user, n_item, items, adj_item, adj_adam, user2item, train_data, test_data


# 池化与卷积步骤的具体函数
LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class AdamicPooling(object):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + str(get_layer_id(layer))
        self.batch_size = batch_size
        self.dim = dim
        self.dropout = dropout
        self.act = act
        self.name = name
        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(shape=[self.dim * 2, self.dim],
                                           initializer=tf.contrib.layers.xavier_initializer(),
                                           name='weights')
            self.bias = tf.get_variable(shape=[self.dim],
                                        initializer=tf.zeros_initializer(),
                                        name='bias')

    def __call__(self, self_vectors, neighbor_vectors, neighbor_adams):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_adams)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_adams):
        # print(self_vectors.shape)
        # print(neighbor_vectors.shape)
        # print(neighbor_adams.shape)
        # self_vectors:根节点的特征向量每个点自己本身的特征向量:[batch_size, -1, dim]
        # neighbor_vectors:邻点的特征向量:[batch_size, -1, n_sample, dim]
        # neighbor_adams:权重值:[batch_size, -1, n_sample]

        # normalize adamic values [batch_size, -1, n_sample]
        adamvalues_normalized = tf.nn.softmax(neighbor_adams, dim=-1)
        # [batch_size, -1, n_sample, 1]
        adamvalues_normalized = tf.expand_dims(adamvalues_normalized, axis=-1)
        # [batch_size, -1, dim]
        neighbors_afterPooling = tf.reduce_mean(adamvalues_normalized * neighbor_vectors, axis=2)

        # [batch_Size, -1, dim*2]
        output = tf.concat([self_vectors, neighbors_afterPooling], axis=-1)
        # [-1, dim*2]
        output = tf.reshape(output, [-1, self.dim * 2])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        # [-1. dim] full connect method
        output = tf.matmul(output, self.weights) + self.bias
        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class AveragePooling(object):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.batch_size = batch_size
        self.dim = dim
        self.dropout = dropout
        self.act = act
        self.name = name
        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(shape=[self.dim * 2, self.dim],
                                           initializer=tf.contrib.layers.xavier_initializer(),
                                           name='weights')
            self.bias = tf.get_variable(shape=[self.dim], 
                                        initializer=tf.contrib.layers.xavier_initializer(), 
                                        name='bias')

    def __call__(self        , self_vectors, neighbor_vectors):
        outputs = self._call(self_vectors, neighbor_vectors)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors):

        # self_vectors:每个点自己本身的特征向量:[batch_size, -1, dim]
        # neighbor_vectors:每个点子节点的特征向量:[batch_size, -1, n_sample, dim]

        # [batch_size, -1, dim]
        neighbors_afterPooling = tf.reduce_mean(neighbor_vectors, axis=-1)
        # [batch_Size, -1, dim*2]
        output = tf.concat([self_vectors, neighbors_afterPooling], axis=-1)
        # [-1, dim*2]
        output = tf.reshape(output, [-1, self.dim * 2])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        # [-1. dim] full connect method
        output = tf.matmul(output, self.weights) + self.bias
        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class MaxPooling(object):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.batch_size = batch_size
        self.dim = dim
        self.dropout = dropout
        self.act = act
        self.name = name
        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(shape=[self.dim * 2, self.dim],
                                           initializer=tf.contrib.layers.xavier_initializer(),
                                           name='weights')
            self.bias = tf.get_variable(shape=[self.dim],
                                        initializer=tf.zeros_initializer(),
                                        name='bias')

    def __call__(self, self_vectors, neighbor_vectors):
        outputs = self._call(self_vectors, neighbor_vectors)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors):
        # self_vectors: [batch_size, -1, dim]
        # neighbor_vectors: [batch_size, -1, n_sample, dim]

        # [batch_size, -1, dim]
        neighbors_afterPooling = tf.reduce_max(neighbor_vectors, axis=-1)
        # [batch_size, -1, dim*2]
        output = tf.concat([self_vectors, neighbors_afterPooling], axis=-1)
        # [-1, dim*2]
        output = tf.reshape(output, [-1, self.dim * 2])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        # [-1. dim] full connect method
        output = tf.matmul(output, self.weights) + self.bias
        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class fc_layer(object):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.softmax, name=None):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.batch_size = batch_size
        self.dim = dim
        self.dropout = dropout
        self.act = act
        self.name = name
        with tf.variable_scope(self.name):
            self.wfc = tf.get_variable(shape=[self.dim * 2, self.dim],
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       name='weights')
            self.bfc = tf.get_variable(shape=[self.dim], 
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       name='bias')
            self.wd = tf.get_variable(shape=[self.dim, 1], initializer=tf.contrib.layers.xavier_initializer(),
                                      name='weights2')
            # self.bd = tf.get_variable(shape=[self.dim])

    def __call__(self, user_embeddings, item_embeddings):
        outputs = self._call(user_embeddings, item_embeddings)
        return outputs

    def _call(self, user_embeddings, item_embeddings):
        # [batch_size, dim*2]
        output1 = tf.concat([user_embeddings, item_embeddings], axis=-1)
        # [-1, dim*2]
        output1 = tf.reshape(output1, [-1, self.dim * 2])
        output1 = tf.nn.dropout(output1, keep_prob=1-self.dropout)
        # [-1, dim] full connect layer
        output1 = tf.matmul(output1, self.wfc) + self.bfc
        # [batch_size, dim]
        output1 = tf.reshape(output1, [self.batch_size, self.dim])
        output1 = tf.nn.relu(output1)

        # output layer
        output2 = tf.reshape(output1, [-1, self.dim])
        output = tf.matmul(output2, self.wd)
        output = tf.reshape(output, [self.batch_size])
        output = self.act(output)
        return output


class TSCN(object):
    def __init__(self, args, n_users, n_items, adj_item, adj_adam, user2item_dict):
        self._parse_args(args, adj_item, adj_adam, user2item_dict)
        self._build_inputs()
        self._build_model(args, n_users, n_items)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_item, adj_adam, user2item_dict):
        self.adj_item = adj_item
        self.adj_adam = adj_adam
        self.user2item_dict = user2item_dict

        self.k = args.k
        self.batch_size = args.batch_size
        self.n_sample = args.sample_size  # the number of sample a vertex's childnode
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        if args.pooling == 'average':
            self.pooling_class = AveragePooling
        elif args.pooling == 'max':
            self.pooling_class = MaxPooling
        elif args.pooling == 'adamic':
            self.pooling_class = AdamicPooling
        else:
            raise Exception('Unknown pooling method: ' + args.pooling)

    def _build_inputs(self):
        # ndices = tf.placeholder(dtype=tf.int32, shape=[None], name='user_indices')
        self.user_indices = tf.placeholder(dtype=tf.int32, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int32, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self, args, n_users, n_items):
        self.item_embedding_matrix = tf.get_variable(
            shape=[n_items, self.dim], initializer=TSCN.get_initializer(), name='item_embedding_matrix')
        self.user_embedding_matrix = tf.get_variable(
            shape=[n_users, self.dim], initializer=TSCN.get_initializer(), name='user_embedding_matrix')
        
        # 解释：为何这里的大小是batch_size x dim:
        # 在训练的过程中，我们输入的user item interaction graph分为三行
        # 第一行为user 第二行为item 第三行为label（positive‘s label = 1,negative's label=0)
        # 当数据处理完成后即开始训练
        # [batch_size, dim]
        self.item_embeddings = tf.nn.embedding_lookup(self.item_embedding_matrix, self.item_indices)
        # 用户特征向量初始化
        self.user_embeddings = tf.nn.embedding_lookup(self.user_embedding_matrix, self.user_indices)

        # self.user_embeddings = self.user_emb_initializer(self.user_indices)

        entities, adamvalues = self.get_childnodes(self.item_indices)
        # pooling step
        self.item_embeddings, self.poolings = self.pool_and_convolution(args, entities, adamvalues)
        # finished the itemgraph convolution step
        # [batch_size, dim]
        # fc layer
        self.fclayer = fc_layer(self.batch_size, self.dim)
        self.output = self.fclayer(self.user_embeddings, self.item_embeddings)

    # 添加了一个n_users, 用户id的

    # 用户向量的初始化函数
    # def user_emb_initializer(self, n_user, ):
    #     print('initializing user embeddings ...')
    #     i = 0
    #     uservec = []
    #     for u in userindex:
    #         itemids = self.user2item_dict[u]
    #         if itemindex[0] in itemids:
    #             itemids.remove(itemindex[0])
    #             length = len(itemids)
    #             itemids = tf.Tensor(itemids)
    #             itemvectors = tf.gather(self.adj_item, itemids)
    #             user_emb = tf.divide(tf.reduce_sum(itemvectors, axis=-1), length)
    #         else:
    #             itemvectors = tf.gather(self.adj_item, itemids)
    #             user_emb = tf.divide(tf.reduce_sum(itemvectors, axis=-1), length)
    #         uservec.append(user_emb)
    #     output = tf.reshape(uservec, [self.batch_size, self.dim])
    #
    #     user_emb_matrix = list()
    #     for user in range(n_user):
    #
    #     return output

    def get_childnodes(self, seeds):
        print('geting childnodes of vertexes ...')
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        adamvalues = []
        for i in range(self.k):
            neighbor_entities = tf.reshape(tf.gather(self.adj_item, entities[i]), [self.batch_size, -1])
            neighbor_adamvalues = tf.reshape(tf.gather(self.adj_adam, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            adamvalues.append(neighbor_adamvalues)
        return entities, adamvalues
    
    # 这个函数需要传入adamvalues的量 12/09 22:19 fixed
    def pool_and_convolution(self, args, entities, adamvalues):
        poolings = [] # store all pooling method
        item_vector = [tf.nn.embedding_lookup(self.item_embedding_matrix, i) for i in entities]
        # adam_vectors = [tf.nn.embedding_lookup(self.adam_embedding_matrix, i) for i in adamvalues]
        
        if args.pooling == 'adamic':
            for i in range(self.k):
                pooling = self.pooling_class(self.batch_size, self.dim)
                poolings.append(pooling)

                item_vector_next_iter = []
                for hop in range(self.k - i):
                    # dimension explanation: batchsize x nodes x numberOfSample x dimension of embedding vector
                    # the number of nodes is uncertain, so it's -1
                    # why: in iteration 1 n_nodes=1, in iteration 2 n_nodes=3 ...
                    shape = [self.batch_size, -1, self.n_sample, self.dim]
                    vector = pooling(self_vectors=item_vector[hop],
                                     neighbor_vectors=tf.reshape(item_vector[hop + 1], shape),
                                     neighbor_adams=tf.reshape(adamvalues[hop], [self.batch_size, -1, self.n_sample]))
                    item_vector_next_iter.append(vector)
                item_vectors = item_vector_next_iter
        else:
            for i in range(self.k):
                pooling = self.pooling_class(self.batch_size, self.dim)
                poolings.append(pooling)

                item_vector_next_iter = []
                for hop in range(self.k - i):
                    shape = [self.batch_size, -1, self.n_sample, self.dim]
                    vector = pooling(self_vectors=item_vector[hop],
                                     neighbor_vectors=tf.reshape(item_vector[hop + 1], shape))
                    item_vector_next_iter.appen(vector)
                item_vectors = item_vector_next_iter
        
        res = tf.reshape(item_vectors[0], [self.batch_size, self.dim])
        return res, poolings

    def _build_train(self):
        # 计算损失函数
        self.base_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.output))
        self.l2_loss = tf.nn.l2_loss(self.user_embedding_matrix) + tf.nn.l2_loss(self.item_embedding_matrix)
        for p in self.poolings:
            self.l2_loss += tf.nn.l2_loss(p.weights)
        self.l2_loss += tf.nn.l2_loss(self.fclayer.wfc) + tf.nn.l2_loss(self.fclayer.wd)
        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        # 优化器
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)
    
    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.output], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1
    
    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.output], feed_dict)

# 计算过程代码
# __init__(self, args, n_users, n_items, adj_item, adj_adam, user2item_dict):
# 2020/01/10 userid indexed itemid indexed

# 以下是TSCN的输入：
# def _build_inputs(self):
#         # ndices = tf.placeholder(dtype=tf.int32, shape=[None], name='user_indices')
#         self.user_indices = tf.placeholder(dtype=tf.int32, shape=[None], name='user_indices')
#         self.item_indices = tf.placeholder(dtype=tf.int32, shape=[None], name='item_indices')
#         self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')
def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict


# is_train的类型为bool
def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

def topn_settings(train_data, test_data, n_item):
    train_record = get_user_record(train_data, True)
    test_record = get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    item_set = set(list(range(n_item)))
    return user_list, train_record, test_record, item_set


def topn_eval(sess, model, user_list, train_record, test_record, item_set, n, batch_size):
    HR_precision_list = list()
    NDCG_precision_list = list()

    Z = 0
    for i in range(n):
        Z += 1 / (math.log2(i + 2))
    Z = 1 / Z

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size, model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size
        
        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size, 
                                                    model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score
        
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        
        hit_num = len(set(item_sorted[:n]) & test_record[user])
        # HR result
        precision1 = hit_num / n
        HR_precision_list.append(precision1)
        
        # NDCG resul\
        sum = 0
        recommended_list = list(item_sorted[:n])
        hit_set = set(item_sorted[:n]) & test_record[user]
        for i in range(len(recommended_list)):
            if recommended_list[i] in hit_set:
                sum += 1 / (math.log2(i + 2))
        precision2 = Z * sum
        NDCG_precision_list.append(precision2)
    HR_precision = np.mean(HR_precision_list)
    NDCG_precision = np.mean(NDCG_precision_list)
    return HR_precision, NDCG_precision


def train(args, data, show_loss):
    # load_data返回值：return n_user, n_item, items, adj_item, adj_adam, user2item, train_data, test_data
    n_user = data[0]
    n_item = data[1]
    items = data[2]
    adj_item = data[3]
    adj_adam = data[4]
    user2item = data[5]
    train_data, test_data = data[6], data[7]
    model = TSCN(args, n_users=n_user, n_items=n_item, adj_item=adj_item, adj_adam=adj_adam, user2item_dict=user2item)
    # 2019/12/21 14:50 :现在按照要求重新处理原始数据
    # topK evaluation settings 论文中貌似没有 忽略
    
    user_list, train_record, test_record, item_set = topn_settings(train_data, test_data, n_item)

    # 训练过程
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for step in range(args.n_epochs):
            # training
            np.random.shuffle(train_data)
            start = 0
            while start + args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print(start, loss)
            # evalution method should be added. (2020/01/10 20:30)
            # HR evaluation step
            # def topn_eval(sess, model, user_list, train_record, test_record, item_set, n, batch_size):
            HR_precision, NDCG_precision = topn_eval(sess, model, user_list, train_record, test_record, item_set, 10, args.batch_size)
            print('epoch {} '.format(step))
            print('HR precision: {:.4f} '.format(HR_precision), end='')
            print('NDCG precision: {:.4f}'.format(NDCG_precision))


data = load_data()
train(args, data=data, show_loss=True)
