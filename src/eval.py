import torch
import numpy as np
import math
import torch.nn.functional as F

def get_user_record(data_loader):
    print("getting user's record ...")
    record = dict()
    for _, data in enumerate(data_loader):
        inputs, labels = data
        users = inputs.numpy()[:, 0]
        items = inputs.numpy()[:, 1]
        for i in range(inputs.shape[0]):
            if labels[i] == 1:
                if int(users[i]) not in record.keys():
                    record[int(users[i])] = set()
                record[int(users[i])].add(int(items[i]))
    return record


def evaluation(model, test_loader, user_record, args):
    # get predictions
    print('working on evaluation ...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_record = dict()
    with torch.no_grad():
        for batch, data in enumerate(test_loader):
            inputs = data[0]
            if inputs.shape[0] < args.batch_size:
                continue
            print('\r', 'batch:', batch, end='')
            if torch.cuda.is_available():
                outputs = F.softmax(model(inputs.cuda()), dim=-1)
                outputs = outputs.cpu()
            else:
                outputs = model(inputs)
            recommendation_scores = outputs.numpy()[:, 1]  # 预测为1的概率
            # users = inputs[:, 0]
            # items = inputs[:, 1]
            for i in range(inputs.shape[0]):
                user, item = inputs.numpy()[i, 0], inputs.numpy()[i, 1]
                if user not in test_record.keys():
                    test_record[int(user)] = []
                test_record[int(user)].append([int(item), float(recommendation_scores[i])])

    np.save('test_record.npy', test_record)  # debug fragment

    # eval
    print()
    HR = []
    NDCG = []
    x = 0

    IDCG = 1 / (math.log2(2))
    # 1
    # NDCG = DCG / IDCG

    hit_num_sum = 0
    for user in test_record.keys():
        x += 1
        print('\r', '{}/{}'.format(x, len(test_record)), end='')
        # dtype = [('itemid', 'int'), ('score', 'float')]
        ground_truth = user_record[int(user)]
        test_data = np.array(test_record[user])
        item_score_map = dict()
        item_ids = test_data[:, 0].astype(np.int)
        item_reco_scores = test_data[:, 1]
        for i in range(test_data.shape[0]):
            item_score_map[item_ids[i]] = item_reco_scores[i]
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        del item_score_map

        recommend_list = list(item_sorted[:10])

        # hit ratio computation
        hit_number = len(set(ground_truth) & set(recommend_list))
        # hit_ratio = hit_number / 10
        hit_num_sum += hit_number

        # NDCG computation
        sum = 0
        hit_set = set(item_sorted[:10]) & ground_truth
        for j in range(len(recommend_list)):
            if recommend_list[j] in hit_set:
                sum += 1 / (math.log2(j + 2))
        ndcg_score = sum / IDCG
        NDCG.append(ndcg_score)

    HR_result = hit_num_sum / len(user_record)
    NDCG_result = np.array(NDCG).mean()
    print()

    return HR_result, NDCG_result
