import torch
import math
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

np.random.seed(1)

class AdamicPooling(nn.Module):
    def __init__(self, dim):
        super(AdamicPooling, self).__init__()
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

    def forward(self, self_vectors, neighbor_vectors, neighbor_adams, n_u):

        # self_vectors:[batch_size, -1, dim]
        # neighbor_vectors: [batch_size, -1, n_sample, dim]
        # normalized adamic values [batch_size, -1, n_sample, 1]
        adamvalues_normalized = F.softmax(neighbor_adams, dim=2).unsqueeze(dim=3)
        # adam_sum = torch.sum(neighbor_adams, dim=2).unsqueeze(dim=-1)
        # adamvalues_normalized = (neighbor_adams / adam_sum).unsqueeze(dim=3)

        # [batch size, -1, dim]
        neighbors_after_pooling = torch.sum(adamvalues_normalized * neighbor_vectors, dim=2)

        # [batch size, -1, dim * 2]
        output = torch.cat([self_vectors, neighbors_after_pooling], dim=2)
        # [-1, dim * 2]
        output = output.view(-1, self.dim * 2)
        # [-1, dim]
        output = torch.matmul(output, self.weight) + self.bias
        # [batch size, -1, dim]
        output = output.view(n_u, -1, self.dim)
        output = F.relu(output)
        return output


class AveragePooling(nn.Module):
    def __init__(self, dim):
        super(AveragePooling, self).__init__()
        self.dim = dim

        self.weight = Parameter(torch.FloatTensor(self.dim * 2, self.dim))
        self.bias = Parameter(torch.FloatTensor(self.dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, self_vectors, neighbor_vectors, n_u):

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
        output = output.view(n_u, -1, self.dim)
        output = F.relu(output)

        return output


class MaxPooling(nn.Module):
    def __init__(self, dim):
        super(MaxPooling, self).__init__()
        self.dim = dim

        self.weight = Parameter(torch.FloatTensor(self.dim * 2, self.dim))
        self.bias = Parameter(torch.FloatTensor(self.dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, self_vectors, neighbor_vectors, n_u):

        # self vectors: [batch size, -1, dim]
        # neighbor vectors: [batch size, -1, n_sample, dim]

        # [batch size, -1, dim]
        neighbors_after_pooling = torch.max(neighbor_vectors, dim=2)
        # [batch size, -1, dim * 2]
        output = torch.cat([self_vectors, neighbors_after_pooling], dim=2)
        # [-1, dim * 2]
        output = output.view(-1, self.dim * 2)
        # [-1, dim]
        output = torch.matmul(output, self.weight) + self.bias
        # [batch size, -1, dim]
        output = output.view(n_u, -1, self.dim)
        output = F.relu(output)

        return output


class TSCN(nn.Module):
    def __init__(self, args, n_item, n_user, adj_item, adj_adam):
        super(TSCN, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.adj_item = torch.Tensor(adj_item).to(self.device)  # adjacency matrix
        self.adj_adam = torch.Tensor(adj_adam).to(self.device)
        # self.user2item_index = user2item
        self.userlist = set(range(n_user))
        self.n_user = n_user
        self.n_item = n_item
        self.k = args.k  # the depth of tree for sub-graph
        self.n_sample = args.sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.pool_name = args.pooling

        if args.pooling == 'average':
            self.pooling_class = AveragePooling(self.dim)
        elif args.pooling == 'max':
            self.pooling_class = MaxPooling(self.dim)
        elif args.pooling == 'adamic':
            self.pooling_class = AdamicPooling(self.dim)
        else:
            raise Exception('Unknown pooling method: ' + args.pooling)

        # initialization of embedding matrix
        # nn.init.xavier_uniform_(self.item_embedding_matrix)
        # self.user_embedding_matrix = self.user_emb_initializer()

        self.item_embedding_matrix = torch.FloatTensor(self.n_item, self.dim)
        nn.init.xavier_uniform_(self.item_embedding_matrix)
        self.zero = torch.zeros(1, self.dim)
        self.item_embedding_matrix = torch.cat((self.item_embedding_matrix, self.zero), dim=0)
        self.item_embedding_matrix = nn.Embedding.from_pretrained(self.item_embedding_matrix)

        # hidden layer in paper
        self.fc1 = nn.Linear(in_features=self.dim * 2, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)

    def forward(self, user_inputs, item_inputs, n_idxs):
        self.n_u = user_inputs.shape[0]
        self.user_input = user_inputs.long()
        self.item_indices = item_inputs.long()
        self.n_idx = n_idxs
        # [n_u, 1, dim]
        self.mask = self.sequence_mask(self.n_idx, maxlen=self.user_input.shape[1], dtype=torch.float32).float().unsqueeze(dim=-1)
        # [n_u, n_item, dim]
        self.user_embeddings = self.item_embedding_matrix(self.user_input)
        # [n_u, dim]
        self.user_embeddings = torch.sum(self.user_embeddings * self.mask, dim=1)
        self.user_embeddings = self.user_embeddings / torch.max(self.n_idx)

        # construct tree sub graph for every item
        # [[vertex], [[childnode *3], [childnode * 3]]
        entities, adamvalues = self.get_childnodes(self.item_indices, self.n_u)
        # pooling and convolution
        self.item_embeddings = self.pooling_and_convolution(entities, adamvalues, self.n_u)

        output = torch.cat([self.user_embeddings, self.item_embeddings], dim=-1)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.softmax(output.view(-1), dim=-1)
        # output = F.log_softmax(output, dim=-1)
        return output

    def sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1).to(self.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix

        mask.type(dtype)
        return mask

    def get_childnodes(self, vertexes, n_u):
        # print('getting childnodes of vertexes ...')
        vertexes = torch.unsqueeze(vertexes, dim=1)
        entities = [vertexes]
        adamvalues = []
        for i in range(self.k):
            indices = entities[i].long().view(-1)
            neighbor_entities = torch.index_select(
                self.adj_item, index=indices, dim=0).view(n_u, -1)
            neighbor_adamvalues = torch.index_select(
                self.adj_adam, index=indices, dim=0).view(n_u, -1)
            entities.append(neighbor_entities)
            adamvalues.append(neighbor_adamvalues)
        return entities, adamvalues

    def pooling_and_convolution(self, entities, adamvalues, n_u):
        item_vectors = [self.item_embedding_matrix(i.long().to(self.device)) for i in entities]

        if self.pool_name == 'adamic':
            for i in range(self.k):
                # pooling = self.pooling_class(self.batch_size, self.dim)
                # pooling.to(self.device)

                item_vector_next_iter = []
                for hop in range(self.k - i):
                    vector = self.pooling_class(self_vectors=item_vectors[hop],
                                                neighbor_vectors=item_vectors[hop + 1].view(
                                                    n_u, -1, self.n_sample, self.dim),
                                                neighbor_adams=adamvalues[hop].view(n_u, -1, self.n_sample),
                                                n_u=n_u)
                    item_vector_next_iter.append(vector)
                item_vectors = item_vector_next_iter
        else:
            for i in range(self.k):
                # pooling = self.pooling_class(self.batch_size, self.dim)

                item_vector_next_iter = []
                for hop in range(self.k - i):
                    vector = self.pooling_class(self_vectors=item_vectors[hop],
                                                neighbor_vectors = item_vectors[hop + 1].view(
                                                    n_u, -1, self.n_sample, self.dim), n_u=n_u)
                    item_vector_next_iter.append(vector)
                item_vectors = item_vector_next_iter
        result = item_vectors[0].view(n_u, self.dim)
        return result
