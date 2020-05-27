import torch
import math
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

np.random.seed(1)

class AdamicPooling(nn.Module):
    def __init__(self, batch_size, dim):
        super(AdamicPooling, self).__init__()
        self.batch_size = batch_size
        self.dim = dim

        # weight = torch.Tensor(self.dim * 2, self.dim)
        self.weight = Parameter(torch.FloatTensor(self.dim * 2, self.dim))
        self.bias = Parameter(torch.FloatTensor(self.dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, self_vectors, neighbor_vectors, neighbor_adams):

        # self_vectors:[batch_size, -1, dim]
        # neighbor_vectors: [batch_size, -1, n_sample, dim]
        # normalized adamic values [batch_size, -1, n_sample, 1]
        adamvalues_normalized = F.softmax(neighbor_adams, dim=2).unsqueeze(dim=3)
        # [batch size, -1, dim]
        neighbors_after_pooling = torch.mean(adamvalues_normalized * neighbor_vectors, dim=2)

        # [batch size, -1, dim * 2]
        output = torch.cat([self_vectors, neighbors_after_pooling], dim=2)
        # [-1, dim * 2]
        output = output.view(-1, self.dim * 2)
        # [-1, dim]
        output = torch.matmul(output, self.weight) + self.bias
        # [batch size, -1, dim]
        output = output.view(self.batch_size, -1, self.dim)
        output = F.relu(output)
        return output


class AveragePooling(nn.Module):
    def __init__(self, batch_size, dim):
        super(AveragePooling, self).__init__()
        self.batch_size = batch_size
        self.dim = dim

        self.weight = Parameter(torch.FloatTensor(self.dim * 2, self.dim))
        self.bias = Parameter(torch.FloatTensor(self.dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, self_vectors, neighbor_vectors):

        # self vectors: [batch size, -1, dim]
        # neighbor vectors: [batch size, -1, n_sample, dim]
        # [batch size, -1, dim]
        neighbors_after_pooling = torch.mean(neighbor_vectors, dim=2)
        # [batch size, -1, dim * 2]
        output = torch.cat([self_vectors, neighbors_after_pooling], dim=2)
        # [-1, dim * 2]
        output = output.view(-1, self.dim * 2)
        # [-1, dim]
        output = torch.matmul(output, self.weight) + self.bias
        # [batch size, -1, dim]
        output = output.view(self.batch_size, -1, self.dim)
        output = F.relu(output)

        return output


class MaxPooling(nn.Module):
    def __init__(self, batch_size, dim):
        super(MaxPooling, self).__init__()
        self.batch_size = batch_size
        self.dim = dim

        self.weight = Parameter(torch.FloatTensor(self.dim * 2, self.dim))
        self.bias = Parameter(torch.FloatTensor(self.dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, self_vectors, neighbor_vectors):

        # self vectors: [batch size, -1, dim]
        # neighbor vectors: [batch size, -1, n_sample, dim]

        # [batch size, -1, dim]
        neighbors_after_pooling = torch.mean(neighbor_vectors, dim=2)
        # [batch size, -1, dim * 2]
        output = torch.cat([self_vectors, neighbors_after_pooling], dim=2)
        # [-1, dim * 2]
        output = output.view(-1, self.dim * 2)
        # [-1, dim]
        output = torch.matmul(output, self.weight) + self.bias
        # [batch size, -1, dim]
        output = output.view(self.batch_size, -1, self.dim)
        output = F.relu(output)

        return output


class TSCN(nn.Module):
    def __init__(self, args, n_item, n_user, adj_item, adj_adam, user2item):
        super(TSCN, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.adj_item = torch.Tensor(adj_item).to(self.device)  # adjacency matrix
        self.adj_adam = torch.Tensor(adj_adam).to(self.device)
        self.user2item_index = user2item
        self.userlist = set(range(n_user))
        self.n_item = n_item
        self.k = args.k  # the depth of tree for sub-graph
        self.batch_size = args.batch_size
        self.n_sample = args.sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.pool_name = args.pooling

        if args.pooling == 'average':
            self.pooling_class = AveragePooling(self.batch_size, self.dim)
        elif args.pooling == 'max':
            self.pooling_class = MaxPooling(self.batch_size, self.dim)
        elif args.pooling == 'adamic':
            self.pooling_class = AdamicPooling(self.batch_size, self.dim)
        else:
            raise Exception('Unknown pooling method: '+ args.pooling)

        # initialization of embedding matrix
        self.item_embedding_matrix = torch.FloatTensor(self.n_item, self.dim)
        nn.init.xavier_uniform_(self.item_embedding_matrix)
        self.user_embedding_matrix = self.user_emb_initializer()

        self.item_embedding_matrix = nn.Embedding.from_pretrained(self.item_embedding_matrix)
        self.user_embedding_matrix = nn.Embedding.from_pretrained(self.user_embedding_matrix)

        # hidden layer in paper
        self.fc1 = nn.Linear(in_features=self.dim * 2, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=2)

    def forward(self, inputs):
        self.user_indices = inputs[:, 0].long()
        self.item_indices = inputs[:, 1].long()

        # [batch size, dim]
        # find the embedding vector
        # self.item_embeddings = self.item_embedding_matrix(self.item_indices)
        self.user_embeddings = self.user_embedding_matrix(self.user_indices)

        # construct tree sub graph for every item
        entities, adamvalues = self.get_childnodes(self.item_indices)
        # pooling and convolution
        self.item_embeddings = self.pooling_and_convolution(entities, adamvalues)

        output = torch.cat([self.user_embeddings, self.item_embeddings], dim=-1)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)
        # output = F.log_softmax(output, dim=-1)
        return output

    def user_emb_initializer(self):
        # initialization function of user embedding matrix

        print('user embedding matrix initializing ...')
        for user in self.userlist:
            print('\r', '{}/{} '.format(user, len(self.userlist)), end='')
            item_index = torch.LongTensor(list(self.user2item_index[user]))

            item_vectors = torch.index_select(self.item_embedding_matrix, dim=0, index=item_index)
            user_embedding = torch.mean(item_vectors, dim=0)
            user_embedding = user_embedding.unsqueeze(dim=0)
            if user == 0:
                user_embedding_matrix = user_embedding
            else:
                user_embedding_matrix = torch.cat([user_embedding_matrix, user_embedding], dim=0)
        print("\nuser embedding matrix's shape: {}".format(user_embedding_matrix.shape))
        print('Done')
        return user_embedding_matrix

    def get_childnodes(self, vertexes):
        # print('getting childnodes of vertexes ...')
        vertexes = torch.unsqueeze(vertexes, dim=1)
        entities = [vertexes]
        adamvalues = []
        for i in range(self.k):
            indices = entities[i].long().view(-1)
            neighbor_entities = torch.index_select(
                self.adj_item, index=indices, dim=0).view(self.batch_size, -1)
            neighbor_adamvalues = torch.index_select(
                self.adj_adam, index=indices, dim=0).view(self.batch_size, -1)
            entities.append(neighbor_entities)
            adamvalues.append(neighbor_adamvalues)
        return entities, adamvalues

    def pooling_and_convolution(self, entities, adamvalues):
        item_vectors = [self.item_embedding_matrix(i.long().to(self.device)) for i in entities]

        if self.pool_name == 'adamic':
            for i in range(self.k):
                # pooling = self.pooling_class(self.batch_size, self.dim)
                # pooling.to(self.device)

                item_vector_next_iter = []
                for hop in range(self.k - i):
                    vector = self.pooling_class(self_vectors=item_vectors[hop],
                                                neighbor_vectors=item_vectors[hop + 1].view(
                                                    self.batch_size, -1, self.n_sample, self.dim),
                                                neighbor_adams=adamvalues[hop].view(self.batch_size, -1, self.n_sample))
                    item_vector_next_iter.append(vector)
                item_vectors = item_vector_next_iter
        else:
            for i in range(self.k):
                # pooling = self.pooling_class(self.batch_size, self.dim)

                item_vector_next_iter = []
                for hop in range(self.k - i):
                    vector = self.pooling_class(self_vectors=item_vectors[hop],
                                                neighbor_vectors = item_vectors[hop + 1].view(
                                                    self.batch_size, -1, self.n_sample, self.dim))
                    item_vector_next_iter.append(vector)
                item_vectors = item_vector_next_iter
        result = item_vectors[0].view(self.batch_size, self.dim)
        return result
