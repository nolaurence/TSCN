import numpy as np
import pandas as pd


def load_events():
    print('loading raw data from events.csv ...')
    behaviordata = pd.read_csv('ecommerce-dataset/events.csv')

    # 先收集用户id 与物品id
    users = list(set(behaviordata['visitorid'].values))
    items = list(set(behaviordata['itemid'].values))
    print(behaviordata.shape)
    print(behaviordata.head())
    print(len(users))  # 由此得知用户id是由0~1407979
    print(len(items))  # 235061个
    n_users, n_items = len(users), len(items)

    user2item = dict.fromkeys(users)
    print('constructing index of user to items')
    # 构建每个用户交互过的商品的数据集n
    print('Cycle progress: ')
    for idx in range(behaviordata.shape[0]):
        if user2item[behaviordata['visitorid'][idx]] is None:
            user2item[behaviordata['visitorid'][idx]] = []
            user2item[behaviordata['visitorid'][idx]].append(behaviordata['itemid'][idx])
        else:
            user2item[behaviordata['visitorid'][idx]].append(behaviordata['itemid'][idx])
        if idx % 10000 == 0:
            print('\r', '{:.2f} %'.format(idx / behaviordata.shape[0] * 100), end='')
    print('\n')

    item2user = dict.fromkeys(items)
    print('constructing index of item to user')
    # item对于userid的字典m
    print('Cycle progress: ')
    for i in range(behaviordata.shape[0]):
        if item2user[behaviordata['itemid'][i]] == None:
            item2user[behaviordata['itemid'][i]] = []
            item2user[behaviordata['itemid'][i]].append(behaviordata['visitorid'][i])
        else:
            item2user[behaviordata['itemid'][i]].append(behaviordata['visitorid'][i])
        if i % 10000 == 0:
            print('\r', '{:.2f} %'.format(i / behaviordata.shape[0] * 100), end='')
    return users, items, n_users, n_items, user2item, item2user




# In[29]:


def build_itemgraph(items, user2item, item2user):
    temporary_graph = []
    a = []
    print('contructing itemgraph without adamic value...')
    # 建立商品图的邻接表 temporary_graph
    # 2019/11/23 新的想法 在这里实现创建itemid与item_index的索引
    # 解释：items直接可以作为itemid与item_index的索引：
    # 返回itemid：items[index], 范围index：items.index[itemid]
    for i in range(len(items)):
        if i % 1000 == 0:
            print('\r', '{:.2f} %'.format(i / len(items) * 100), end='')
        for user in item2user[items[i]]:
            a += user2item[user]
        a = list(set(a))
        for j in a:
            temporary_graph.append([items[i], j])
        a.clear()
    print('\n')
    temporary_graph = np.array(temporary_graph)
    # for i in range(temporary_graph.shape[0]):
    #     if i % 10000 == 0:
    #         print('\r', '{:.2f} % '.format(i / temporary_graph.shape[0] * 100), end='')
    #     idx = items.index(temporary_graph[i][1])
    #     temporary_graph[i][1] = idx
    # adamValues = []
    itemgraph = []
    print('conputing adamic value between items ...')
    print('Cycle progress:')
    for i in range(temporary_graph.shape[0]):
        if i % 10000 == 0:
            print('\r', '{:.2f} %'.format(i / temporary_graph.shape[0] * 100), end='')
        # user1 = item2user[temporary_graph[i][0]]
        user1 = item2user[temporary_graph[i][0]]
        # user2 = item2user[temporary_graph[i][1]]
        user2 = item2user[temporary_graph[i][1]]
        a = [x for x in user1 if x in user2]
        adamic = 0
        for user in a:
            adamic += 1 / (np.log(len(user2item[user])))
        if temporary_graph[i][0] != temporary_graph[i][1]:
            itemgraph.append([temporary_graph[i][0], temporary_graph[i][1], adamic])
    np.save('/data/itemgraph.npy', itemgraph)

    return itemgraph

def construct_graph(itemgraph, n_sample):
    print('constructing itemgraph with adamic value ...')
    dictionary = {}
    newgraph = {}
    for i in range(itemgraph.shape[0]):
        if i % 10000 == 0:
            print('\r', '{:.2f} %'.format(i / itemgraph.shape[0] * 100), end='')
        first = int(itemgraph[i][0])
        second = int(itemgraph[i][1])
        adamweight = itemgraph[i][2]
        # 此处建立无向图
        if first not in dictionary:
            dictionary[first] = []
        dictionary[first].append((second, adamweight))

        # if second not in dictionary:
        #     dictionary[second] = []
        # dictionary[second].append((first, adamweight))
        lingshi = dictionary[first]
        dictionary[first] = list(set(lingshi))
        # lingshi = dictionary[second]
        # dictionary[second] = list(set(lingshi))
    # 子图规则化
    j = 0
    print('\n')
    print('operating on subgraph normalization ...')
    dtype = [('id', 'int'), ('adamvalue', 'float')]
    for stuff in dictionary.keys():
        j += 1
        if j % 1000 == 0:
            print('\r', '{:.2f} %'.format(j / len(dictionary) * 100), end='')
        # first = item
        # second = graph[item][0]
        if len(dictionary[stuff]) > n_sample:
            x = np.array(dictionary[stuff], dtype=dtype)
            newgraph[stuff] = list(np.sort(x, order='adamvalue')[-n_sample:])
        else:
            newgraph[stuff] = dictionary[stuff]

    return newgraph

def itemid_to_index(graph, item2index):
    print('indexing itemgraph ...')
    indexed_graph = {}
    for stuff in graph.keys():
        index = item2index[stuff]
        alist = graph[stuff]
        blist = []
        
        for i in range(len(alist)):
            # alist[i][0] = item2index[alist[i][0]]
            blist.append((item2index[alist[i][0]], alist[i][1]))
        indexed_graph[index] = blist
        del blist
    print('\n')
    return indexed_graph

# 此函数待修改
def construct_adj(graph, items, item_number, n_sample):
    print('constructing adjacency matrix ...')
    # 矩阵的每一行都对应这子图规则化选出的三个adamic weight最大的n_sample个子物品（vertex的子节点）
    # item2index = create_index(items)
    adj_item = np.zeros([item_number, n_sample], dtype=np.int32)
    adj_adam = np.zeros([item_number, n_sample], dtype=np.float32)
    for i in range(item_number):
        if i % 1000 == 0:
            print('\r', '{:.2f} %'.format(i / n_items * 100), end='')
        if i in graph.keys():
            neighbors = graph[i]
            # adj_item[i] = np.array([neighbors[j][0] for j in range(len(neighbors))])
            itemlist = [neighbors[j][0] for j in range(len(neighbors))]
            adamlist = [neighbors[j][1] for j in range(len(neighbors))]
            if len(itemlist) < n_sample:
                m = n_sample - len(itemlist)
                for x in range(m):
                    itemlist.append(0)
                    adamlist.append(0)
            adj_item[i] = np.array(itemlist)
            adj_adam[i] = np.array(adamlist)
            del itemlist
            del adamlist
            # adj_adam[i] = np.array([neighbors[j][1] for j in range(len(neighbors))])
    print('\n')
    return adj_item, adj_adam
    # 待开发
    # 2019/11/16 21:52需要在此函数设计一个item 2 adjacency id的index
# n_item, index, item2index = create_index(graph=graph, itemgraph=itemgraph)
def create_index(items):
    print('creating index from serial to itemid ...')
    item2index = {}
    n_item = 0
    for item in items:
        if n_item % 10000 == 0:
            print('\r', '{:.2f} %'.format(n_item / len(items) * 100), end='')
        item2index[item] = n_item
        n_item += 1
    return item2index


# 执行部分
n_sample = 3
users, items, n_users, n_items, user2item, item2user = load_events()
np.save('data/item_index.npy', items)
np.save('data/item2user.npy', item2user)
np.save('data/user2item.npy', user2item)

itemgraph = build_itemgraph(items=items, user2item=user2item, item2user=item2user)
itemgraph = np.array(itemgraph)

item2index = create_index(items=items)
np.save('data/item2index.npy', item2index)
graph = construct_graph(itemgraph=itemgraph, n_sample=n_sample)

np.save('data/graph.npy', graph)
indexed_graph = itemid_to_index(graph=graph, item2index=item2index)
print(len(indexed_graph))
np.save('data/indexgraph.npy', indexed_graph)
adj_item, adj_adam = construct_adj(graph=indexed_graph, items=items, item_number=n_items, n_sample=n_sample)

np.save('data/adj_item.npy', adj_item)
np.save('data/adj_adam.npy', adj_adam)
print(adj_adam.shape)
