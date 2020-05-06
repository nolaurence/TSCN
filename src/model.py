import numpy as np
import tensorflow as tf

import math
from abc import abstractmethod
np.random.seed(1)

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

    def __call__(self, self_vectors, neighbor_vectors):
        outputs = self._call(self_vectors, neighbor_vectors)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors):

        # self_vectors:每个点自己本身的特征向量:[batch_size, -1, dim]
        # neighbor_vectors:每个点子节点的特征向量:[batch_size, -1, n_sample, dim]

        # [batch_size, -1, dim]
        neighbors_afterPooling = tf.reduce_mean(neighbor_vectors, axis=2)
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
        neighbors_afterPooling = tf.reduce_max(neighbor_vectors, axis=2)
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
    def __init__(self, batch_size, dim, dropout=0., name=None):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.batch_size = batch_size
        self.dim = dim
        self.dropout = dropout
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
        # output = self.act(output)
        return output


class TSCN(object):
    def __init__(self, args, n_items, adj_item, adj_adam, user2item_dict):
        self._parse_args(args, adj_item, adj_adam, user2item_dict)
        self._build_inputs()
        self._build_model(args, n_items)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_item, adj_adam, user2item_dict):
        self.adj_item = adj_item
        self.adj_adam = adj_adam
        self.user2item_dict = user2item_dict
        self.userlist = set(user2item_dict.keys())

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
    
    def _build_model(self, args, n_items):
        self.item_embedding_matrix = tf.get_variable(shape=[n_items, self.dim], initializer=TSCN.get_initializer(), name='item_embedding_matrix')
        self.user_embedding_matrix = self.user_emb_initializer()
        
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
        self.output_normalized = tf.nn.softmax(self.output)
    
    def _build_train(self):
        # 计算损失函数
        self.base_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.output))
        self.l2_loss = tf.nn.l2_loss(self.user_embedding_matrix) + tf.nn.l2_loss(self.item_embedding_matrix)
        for p in self.poolings:
            self.l2_loss += tf.nn.l2_loss(p.weights)
        self.l2_loss += tf.nn.l2_loss(self.fclayer.wfc) + tf.nn.l2_loss(self.fclayer.wd)
        self.loss = self.base_loss + self.l2_weight * self.l2_loss
        # self.loss = self.base_loss

        # 优化器
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    
    # 用户嵌入向量的初始化函数
    def user_emb_initializer(self):
        print('user embedding matrix initializing ...')
        for user in self.userlist:
            print('\r', '{}/{} '.format(user, len(self.userlist)), end='')
            item_index = tf.convert_to_tensor(self.user2item_dict[user], dtype=tf.int32)
            # [n_item, dim]
            item_vectors = tf.nn.embedding_lookup(self.item_embedding_matrix, item_index)
            # [dim]
            user_emb = tf.reduce_mean(item_vectors, axis=0)
            # [1, dim]
            user_emb = tf.expand_dims(user_emb, axis=0)
            if user == 0:
                user_emb_matrix = user_emb
            else:
                user_emb_matrix = tf.concat([user_emb_matrix, user_emb], axis=0)

        print('\n')
        print("user embedding matrix's shape: {}".format(user_emb_matrix.shape))
        return user_emb_matrix
    
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
    
    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.output_normalized], feed_dict)
    







# 训练

