import numpy as np


def build_newgraph(user2item, item2index):
    adj_graph = []
    for user in set(user2item.keys()):
        for item in user2item[user]:
            adj_graph.append([user, item2index[item]])
    return adj_graph


user2item = np.load('data/user2item.npy', allow_pickle=True).item()
item2index = np.load('data/item2index.npy', allow_pickle=True).item()
graph = np.array(build_newgraph(user2item, item2index))
print(graph)
np.save('data/usergraph.npy', graph)
