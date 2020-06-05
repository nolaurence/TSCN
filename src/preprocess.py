import pandas as pd
import numpy as np
np.random.seed(1)

def load_events(path):
    print('loading retailrocket dataset ...')
    df = pd.read_csv(path + 'events.csv')
    del df['transactionid']
    # df = df[True ^ df['Sensorvalue'].isin([1, 2, 5])]
    df = df[True ^ df['event'].isin(['addtocart', 'transaction'])]
    del df['event']
    # df = df.reset_index(drop=True)
    df.sort_values('timestamp', inplace=True)

    print(df.shape)
    df = df.drop_duplicates(['visitorid', 'itemid'], keep='last')
    df = df.reset_index(drop=True)
    print(df.shape)
    user2item = dict()
    # dtype = [('itemid', int), ('timestamp', int)]
    for idx in range(df.shape[0]):
        if idx % 10000 == 0:
            print('\r', '{:.2f} %'.format(idx / df.shape[0] * 100), end='')
        user = df['visitorid'][idx]
        item = df['itemid'][idx]
        timestamp = df['timestamp'][idx]
        if user not in user2item.keys():
            user2item[user] = list()
        user2item[user].append([item, timestamp])
    print()
    return user2item


def load_ml(path):
    print('loading movielens-1m dataset ...')
    df = pd.read_csv(path + 'ratings.dat', sep='::', engine='python')
    del df['score']
    print(df.shape)
    df.sort_values('timestamp', inplace=True)
    df = df.drop_duplicates(['userid', 'movieid'], keep='last')
    print(df.shape)
    df = df.reset_index(drop=True)

    data_dict = dict()
    for idx in range(df.shape[0]):
        if idx % 10000 == 0:
            print('\r', '{:.2f} %'.format(idx / df.shape[0] * 100), end='')
        user = df['userid'][idx]
        item = df['movieid'][idx]
        timestamp = df['timestamp'][idx]
        if user not in data_dict.keys():
            data_dict[user] = list()
        data_dict[user].append([item, timestamp])
    print()

    return data_dict


def dataset_filter(dictionary):
    print('filtering ...')
    threshold = 10  # the user whose itemset's number less than threshold will be ignored
    output = dict()
    itemset = set()
    i = 0
    for user in dictionary.keys():
        print('\r', '{}/{}'.format(i, len(dictionary)), end='')
        i += 1
        history = np.array(dictionary[user])
        items = set(history[:, 0])
        if len(items) >= threshold:
            output[user] = history
            for item in items:
                itemset.add(int(item))

    print()
    return output, itemset


def data_split(input):
    print('data split ...')
    # dtype = [('itemid', int), ('timestamp', int)]
    train = dict()
    test = dict()
    i = 0
    for user in input.keys():
        print('\r', '{}/{}'.format(i, len(input)), end='')
        i += 1
        record = input[user]
        # record = record[record[:, 1].argsort()]
        # print(record)
        trainset = set(record[0:-1, 0])
        testset = record[-1, 0]
        train[user] = trainset
        test[user] = int(testset)
    print()
    return train, test


def id_index(dataset, itemset):
    userset = list(dataset.keys())
    userset.sort()
    itemset = list(itemset)
    itemset.sort()
    n_item = len(itemset)
    n_user = len(userset)
    n_interaction = 0
    for user in dataset.keys():
        n_interaction += len(dataset[user])
    print('interactions: {}'.format(n_interaction))
    print('user: {}\nitem: {}'.format(n_user, n_item))
    user_index = dict()
    item_index = dict()
    for i in range(n_user):
        user_index[userset[i]] = i
    for i in range(n_item):
        item_index[itemset[i]] = i
    return user_index, item_index, n_item


def generate_negatives(itemset, trainset, testset, user_index, item_index, dataset):
    print('generating negative smaples ...')
    trainrecord = list()
    testrecord = list()
    x = 0
    for user in trainset.keys():
        print('\r', '{}/{}'.format(x, len(trainset)), end='')
        x += 1
        record = trainset[user]
        pos = dataset[user][:, 0]
        for item in record:
            trainrecord.append([user_index[user], item_index[item], 1])

        # for each event, we random sample 4 negative samples
        chosen = np.random.choice(list(itemset - set(pos)), size=len(record) * 4, replace=False)
        for item in chosen:
            trainrecord.append([user_index[user], item_index[item], 0])
    print()
    x = 0
    for user in testset.keys():
        print('\r', '{}/{}'.format(x, len(testset)), end='')
        x += 1
        record1 = testset[user]
        pos = dataset[user][:, 0]
        testrecord.append([user_index[user], item_index[record1], 1])
        # for each user, we random sample 99 negative samples
        sample = np.random.choice(list(itemset - set(pos)), size=99, replace=False)
        for item in sample:
            testrecord.append([user_index[user], item_index[item], 0])
    print()
    trainrecord = np.array(trainrecord)
    testrecord = np.array(testrecord)

    return trainrecord, testrecord


# return n_item, items, adj_item, adj_adam, user2item, train_data, test_data
def build_itemgraph(train_record, test_record, path, n_sample):
    user2item = dict()
    item2user = dict()
    # build index
    print('building index ...')
    for i in range(train_record.shape[0]):
        if i % 1000 == 0:
            print('\r', '{}/{}'.format(i, train_record.shape[0]), end='')
        user = train_record[i, 0]
        item = train_record[i, 1]
        label = train_record[i, 2]
        if label == 1:
            if user not in user2item:
                user2item[user] = set()
            user2item[user].add(item)
            if item not in item2user:
                item2user[item] = set()
            item2user[item].add(user)
    print()


    for i in range(test_record.shape[0]):
        if i % 1000 == 0:
            print('\r', '{}/{}'.format(i, test_record.shape[0]), end='')
        user = test_record[i, 0]
        item = test_record[i, 1]
        label = test_record[i, 2]
        if label == 1:
            if user not in user2item:
                user2item[user] = set()
            user2item[user].add(item)
            if item not in item2user:
                item2user[item] = set()
            item2user[item].add(user)
    print()
    np.save(path + 'user2item.npy', user2item)
    # np.save(path + 'user2item.npy', user2item)
    # np.save(path + 'item2user.npy', item2user)
    # construct item graph index
    print('build index of item graph ...')
    temp = list()
    graph = list()
    idx = 0
    for item in item2user.keys():
        print('\r', '{:.2f} %'.format(idx / len(item2user) * 100), end='')
        idx += 1
        for user in item2user[item]:
            temp += user2item[user]
        temp = list(set(temp))
        for it in temp:
            graph.append([item, it])
        temp.clear()
    graph = np.array(graph)
    # np.save(path + 'graph.npy', graph)
    print()
    # calculate adamic value
    print('calculating adamic value ...')
    graph_with_value = list()
    for i in range(graph.shape[0]):
        if i % 1000 == 0:
            print('\r', '{:.2f} %'.format(i / graph.shape[0] * 100), end='')
        usergroup1 = item2user[graph[i][0]]
        usergroup2 = item2user[graph[i][1]]
        common_user = usergroup1 & usergroup2
        adamic_value = 0
        for user in common_user:
            adamic_value += 1 / (np.log(len(user2item[user])))
        if graph[i][0] != graph[i][1]:
            graph_with_value.append([graph[i][0], graph[i][1], adamic_value])
    graph_with_value = np.array(graph_with_value)
    print()
    # np.save(path + 'graph_with_value.npy', graph_with_value)
    # construct item graph
    temp_graph = dict()
    itemgraph = dict()
    print('constructing item graph ...')
    for i in range(graph_with_value.shape[0]):
        if i % 10000 == 0:
            print('\r', '{:.2f} %'.format(i / graph_with_value.shape[0] * 100), end='')
        item1 = int(graph_with_value[i][0])
        item2 = int(graph_with_value[i][1])
        adam = graph_with_value[i][2]
        if item1 not in temp_graph:
            temp_graph[item1] = []
        temp_graph[item1].append((item2, adam))

    print('\noperating on subgraph regularization ...')
    dtype = [('id', 'int'), ('adamvalue', 'float')]
    i = 0
    for item in temp_graph.keys():
        i += 1
        if i % 1000 == 0:
            print('\r', '{:.2f} %'.format(i / len(temp_graph) * 100), end='')
        if len(temp_graph[item]) > n_sample:
            x = np.array(temp_graph[item], dtype=dtype)
            itemgraph[item] = list(np.sort(x, order='adamvalue')[-n_sample:])
        else:
            itemgraph[item] = temp_graph[item]
    print()
    return itemgraph


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


def load_retailrocket(path, n_sample):
    user2item = load_events(path)
    dataset, itemset = dataset_filter(user2item)
    trainset, testset = data_split(dataset)
    user_index, item_index, n_item = id_index(dataset, itemset)
    items = list(range(n_item))
    np.save(path + 'items.npy', items)
    trainrecord, testrecord = generate_negatives(itemset, trainset, testset, user_index, item_index, dataset)
    print(trainrecord.shape)
    np.save(path + 'train_data.npy', trainrecord)
    np.save(path + 'test_data.npy', testrecord)

    itemgraph = build_itemgraph(trainrecord, testrecord, path, n_sample)
    # np.save(path + 'itemgraph.npy', itemgraph)
    adj_item, adj_adam = construct_adj(itemgraph, itemset, n_sample)
    np.save(path + 'adj_item.npy', adj_item)
    np.save(path + 'adj_adam.npy', adj_adam)


def load_movielens(path, n_sample):
    data = load_ml(path)
    dataset, itemset = dataset_filter(data)
    # np.save(path + 'dataset.npy', dataset)
    trainset, testset = data_split(dataset)
    user_index, item_index, n_item = id_index(dataset, itemset)
    items = list(range(n_item))
    np.save(path + 'items.npy', items)
    trainrecord, testrecord = generate_negatives(itemset, trainset, testset, user_index, item_index, dataset)
    print(trainrecord.shape)
    np.save(path + 'train_data.npy', trainrecord)
    np.save(path + 'test_data.npy', testrecord)

    itemgraph = build_itemgraph(trainrecord, testrecord, path, n_sample)
    # np.save(path + 'itemgraph.npy', itemgraph)
    adj_item, adj_adam = construct_adj(itemgraph, itemset, n_sample)
    np.save(path + 'adj_item.npy', adj_item)
    np.save(path + 'adj_adam.npy', adj_adam)

