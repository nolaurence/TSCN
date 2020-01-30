import numpy as np
import math


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
            items, scores = model.get_score(sess, {model.user_indices: [user] * batch_size,
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

        hit_num = len(set(item_sorted[:n]) & test_record[user])
        # HR result
        precision1 = hit_num / n
        HR_precision_list.append(precision1)

        # NDCG result
        sum = 0
        recommended_list = list(item_sorted[:n])
        hit_set = set(item_sorted[:n]) & test_record[user]
        for i in len(recommended_list):
            if recommended_list[i] in hit_set:
                sum += 1 / (log2(i + 2))
        precision2 = Z * sum
        NDCG_precision_list.append(precision2)
    HR_precision = np.mean(HR_precision_list)
    NDCG_precision = np.mean(NDCG_precision_list)
    return HR_precision, NDCG_precision