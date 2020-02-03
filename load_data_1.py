# 2019/12/21 15:00 :重新按照论文要求处理数据
import numpy as np
import pandas as pd
np.random.seed(1)


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
    # n_users, n_items = len(users), len(items)

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

    # item2user = dict.fromkeys(items)
    # print('constructing index of item to user')
    # # item对于userid的字典m
    # print('Cycle progress: ')
    # for i in range(behaviordata.shape[0]):
    #     if item2user[behaviordata['itemid'][i]] == None:
    #         item2user[behaviordata['itemid'][i]] = []
    #         item2user[behaviordata['itemid'][i]].append(behaviordata['visitorid'][i])
    #     else:
    #         item2user[behaviordata['itemid'][i]].append(behaviordata['visitorid'][i])
    #     if i % 10000 == 0:
    #         print('\r', '{:.2f} %'.format(i / behaviordata.shape[0] * 100), end='')
    return user2item


def build_newdict(user2item, n_minitems):
    # 过滤掉每个列表中重复的item， 过滤掉列表元素数量小于10的键值对
    print('filtering ...')
    newdict = dict()
    itemlist = list()
    n_user = len(user2item)
    i = 0
    for user in user2item.keys():
        print('\r', '{:.2f} %'.format(i / n_user * 100), end='')
        items = set(user2item[user])
        if len(items) >= n_minitems:
            newdict[user] = list(items)
            for item in items:
                itemlist.append(item)
        i += 1
    itemlist = list(set(itemlist))
    userlist = list(newdict.keys())
    print('\n')
    return newdict, userlist, itemlist


def construct_uiigraph(user2item):
    print('constructing user-item interaction graph without negative samples ...')
    n_user = len(user2item)
    uigraph = list()  # user-item interaction graph
    i = 0 
    for user in user2item.keys():
        if i % 50 == 0:
            print('\r', '{:.2f} %'.format(i / n_user * 100), end='')
        i += 1
        items = list(set(user2item[user]))
        for item in items:
            uigraph.append([user, item, 1])
    uigraph = np.array(uigraph)
    print('\n')
    return uigraph


def construct_itemgraph(uigraph, user2item, items):
    print('constructing index of item to user')
    item2user = dict.fromkeys(items)
    for i in range(uigraph.shape[0]):
        if item2user[uigraph[i][1]] == None:
            item2user[uigraph[i][1]] = list()
        item2user[uigraph[i][1]].append(uigraph[i][0])
        if i % 1000 == 0:
            print('\r', '{:.2f} %'.format(i / uigraph.shape[0] * 100), end='')
    print('\n')

    print('constructing item graph without adamic value...')
    a = list()
    itemgraph_noadam = list()
    for i in range(len(items)):
        if i % 1000 == 0:
            print('\r', '{:.2f} %'.format(i / len(items) * 100), end='')
        for user in item2user[items[i]]:
            a += user2item[user]
        a = list(set(a))
        for j in a:
            itemgraph_noadam.append([items[i], j])
        a.clear()
    itemgraph_noadam = np.array(itemgraph_noadam)
    print('\n')

    print('calculating adamic value between items ...')
    itemgraph = list()
    for i in range(itemgraph_noadam.shape[0]):
        if i % 1000 == 0:
            print('\r', '{:.2f} %'.format(i / itemgraph_noadam.shape[0] * 100), end='')
        user1 = item2user[itemgraph_noadam[i][0]]
        user2 = item2user[itemgraph_noadam[i][1]]
        common_user = [x for x in user1 if x in user2]
        adamic = 0
        for user in common_user:
            adamic += 1 / (np.log(len(user2item[user])))
        if itemgraph_noadam[i][0] != itemgraph_noadam[i][1]:
            itemgraph.append([itemgraph_noadam[i][0], itemgraph_noadam[i][1], adamic])
    itemgraph = np.array(itemgraph)
    print('\n')
    return itemgraph


def build_itemgraph(graph, items, n_sample):
    print('creating index of item id ...')
    item2index = dict()
    n_item = 0
    items.sort()
    print(items)
    for item in items:
        item2index[item] = n_item
        n_item += 1
    n_item += 1 #将序号转化为数量
    np.save('newdata/item2index.npy', item2index)
    print('indexing ...')
    new_itemgraph = []
    for i in range(graph.shape[0]):
        if i % 1000 == 0:
            print('\r', '{:.2f} %'.format(i / graph.shape[0] * 100), end='')
        new_itemgraph.append([item2index[graph[i][0]], item2index[graph[i][1]], graph[i][2]])
    print('\n')
    print('constructing itemgraph ...')
    temporary = dict()
    newgraph = dict()
    for i in range(new_itemgraph.shape[0]):
        if i % 10000 == 0:
            print('\r', '{:.2f} %'.format(i / new_itemgraph.shape[0] * 100), end='')
        first = int(new_itemgraph[i][0])
        second = int(new_itemgraph[i][1])
        adamweight = new_itemgraph[i][2]
        # 此处建立无向图
        if first not in temporary:
            temporary[first] = []
        temporary[first].append((second, adamweight))

        lingshi = temporary[first]
        temporary[first] = list(set(lingshi))
    print('\n operating on sub-graph regularization ...')
    i = 0
    dtype = [('id', 'int'), ('adamvalue', 'float')]
    for s in temporary.keys():
        i += 1
        if i % 1000 == 0:
            print('\r', '{:.2f} %'.format(i / len(temporary) * 100), end='')
        if len(temporary[s]) > n_sample:
            x = np.array(temporary[s], dtype=dtype)
            newgraph[s] = list(np.sort(x, order='adamvalue')[-n_sample:])
        else:
            newgraph[s] = temporary[s]
    return newgraph


def construct_adj(graph, items, n_sample):
    n_item = len(items)
    print('\n constructing adjacency matrix ...')
    adj_item = np.zeros([n_item, n_sample], dtype=np.int32)
    adj_adam = np.zeros([n_item, n_sample], dtype=np.float32)
    for i in range(n_item):
        if i % 1000 == 0:
            print('\r', '{:.2f} %'.format(i / n_item * 100), end='')
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
    print('\n')
    return adj_item, adj_adam


def index_userid(user2item):
    print('indexing the user id ...')
    users = list(set(user2item.keys()))
    users.sort()
    userid2index = dict()
    newuser2item = dict()
    for idx in range(len(users)):
        if idx % 1000 == 0:
            print('\r', '{:.2f} %'.format(idx / len(users) * 100), end='')
        userid2index[users[idx]] = idx
    print('\n')
    i = 0
    for user in user2item.keys():
        if i % 1000 == 0:
            print('\r', '{:.2f} %'.format(i / len(users) * 100), end='')
        i += 1
        newuser2item[userid2index[user]] = user2item[user]
    print('\n')
    return newuser2item


def generate_negative_sample(user2item, items, sample_size):
    print('constructing user-item interaction graph ...')
    i = 0
    uigraph = list()
    for user in user2item.keys():
        if i % 50 == 0:
            print('\r', '{:.2f} %'.format(i / len(user2item) * 100), end='')
        i += 1
        itemlist = list(set(user2item[user]))
        for item in itemlist:
            uigraph.append([user, item, 1])
        # itemlist = set(user2item[user])
        choose = np.random.choice(list(set(items) - set(itemlist)), size=sample_size, replace=False)
        for item in choose:
            uigraph.append([user, item, 0])
    uigraph = np.array(uigraph)
    return uigraph


def dataset_split(uigraph, test_ratio):
    print('spliting dataset ...')
    # test_ratio = 0.3
    n_interactions = uigraph.shape[0]

    test_indices = np.random.choice(list(range(n_interactions)), size=int(n_interactions * test_ratio), replace=False)
    train_indices = list(set(range(n_interactions)) - set(test_indices))

    train_data = uigraph[train_indices]
    test_data = uigraph[test_indices]

    return train_data, test_data


# item graph应该在没有negative sample的数据上建立


user2item = load_events()
user2item, users, items = build_newdict(user2item, n_minitems=10)
np.save('newdata/user2item.npy', user2item)
np.save('newdata/items.npy', items)
uigraph_no_negative = construct_uiigraph(user2item)
np.save('newdata/uigraph_no_negative.npy', uigraph_no_negative)
itemgraph = construct_itemgraph(uigraph=uigraph_no_negative, user2item=user2item, items=items)
itemgraph = build_itemgraph(graph=itemgraph, items=items, n_sample=3)
np.save('newdata/itemgraph.npy', itemgraph)
adj_item, adj_adam = construct_adj(graph=itemgraph, items=items, n_sample=3)
np.save('newdata/adj_item.npy', adj_item)
np.save('newdata/adj_adam.npy', adj_adam)

# user2item = np.load('newdata/user2item.npy', allow_pickle=True).item()
# items = np.load('newdata/items.npy', allow_pickle=True)

# user2item = index_userid(user2item)
# np.save('newdata/newuser2item.npy', user2item)
#
# uigraph = generate_negative_sample(user2item=user2item, items=items, sample_size=4)
# np.save('newdata/uigraph.npy', uigraph)
# train_data, test_data = dataset_split(uigraph=uigraph, test_ratio=0.3)
# np.save('newdata/traindata.npy', train_data)
# np.save('newdata/testdata.npy', test_data)
#
# # index操作应该在split之后，construct adjacency matrix之前
#
# # 注意：与论文不同：原文：9128 users，59,406 items，178,372 interactions
# # 实际：261,417 interactions, 68,433 items
# print(uigraph_no_negative.shape)