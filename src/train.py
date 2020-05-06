from model import TSCN
import numpy as np
import tensorflow as tf
import math
np.random.seed(1)


def train(args, data, show_loss):
    # load_data返回值：return 0n_item, 1items, 2adj_item, 3adj_adam, 4user2item, 5train_data, 6test_data
    # n_user = data[0]
    n_item = data[0]
    items = data[1]
    adj_item = data[2]
    adj_adam = data[3]
    user2item = data[4]
    train_data, test_data = data[5], data[6]
    model = TSCN(args, n_items=n_item, adj_item=adj_item, adj_adam=adj_adam, user2item_dict=user2item)
    # 2019/12/21 14:50 :现在按照要求重新处理原始数据
    # topK evaluation settings 论文中貌似没有 忽略

    user_list, train_record, test_record = topn_settings(train_data, test_data)

    # 训练过程
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(args.n_epochs):
            # training
            print('epoch {} '.format(step))
            np.random.shuffle(train_data)
            start = 0
            while start + args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print('epoch {}: {}/{} loss={:.5f}'.format(step + 1, start, train_data.shape[0], loss))
                    # print(start, loss)
            # evalution method should be added. (2020/01/10 20:30)
            # HR evaluation step
            # def topn_eval(sess, model, user_list, train_record, test_record, item_set, n, batch_size):

        HR_precision, NDCG_precision = topn_eval(sess, model, user_list, train_record,
                                                 test_record, set(items), 10, args.batch_size)
        print('HR precision: {:.4f} '.format(HR_precision), end='')
        print('NDCG precision: {:.4f}'.format(NDCG_precision))


def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict


def get_user_record(data, is_train):
    user_dict = dict()
    for idx in range(data.shape[0]):
        user = data[idx][0]
        item = data[idx][1]
        label = data[idx][2]
        if is_train or label == 1:
            if user not in user_dict:
                user_dict[user] = set()
            user_dict[user].add(item)
    return user_dict


def topn_settings(train_data, test_data):
    train_record = get_user_record(train_data, True)
    test_record = get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    return user_list, train_record, test_record


def topn_eval(sess, model, user_list, train_record, test_record, item_set, n, batch_size):
    # HR_precision_list = list()
    # NDCG_precision_list = list()

    Z = 0
    for s in range(n):
        Z += 1 / (math.log2(s + 2))
    Z = 1 / Z

    print('operating on top n evaluating ...')
    print('n_user: {}'.format(len(user_list)))
    idx = 0
    HR = 0
    NDCG = 0
    for user in user_list:
        idx += 1
        print('\r', '{}/{}'.format(idx, len(user_list)), end='')
        # print('\r', i, end='')
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:] + [
                                                        test_item_list[-1]] * (batch_size - len(
                                                        test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        # print(item_sorted[:n])
        # print(test_record[user])

        hit_num = len(set(item_sorted[:n]) & test_record[user])
        # HR result
        precision1 = hit_num / n
        HR += precision1

        # NDCG result
        sum = 0
        recommended_list = list(item_sorted[:n])
        hit_set = set(item_sorted[:n]) & test_record[user]
        for j in range(len(recommended_list)):
            if recommended_list[j] in hit_set:
                sum += j / (math.log2(j + 2))
        precision2 = Z * sum
        NDCG += precision2
        print(' {:.2f} {:.2f}'.format(precision1, precision2))
    print('\n')
    HR_precision = HR / len(user_list)
    NDCG_precision = NDCG / len(user_list)
    return HR_precision, NDCG_precision
