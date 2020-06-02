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
            length = inputs.shape[0]
            if inputs.shape[0] < args.batch_size:
                # padding the last batch
                inputs = torch.cat((inputs, torch.zeros((args.batch_size - inputs.shape[0], 2), dtype=torch.int32)), 0)
            print('\r', 'batch:', batch, end='')
            if torch.cuda.is_available():
                outputs = F.softmax(model(inputs.cuda()), dim=-1)
                outputs = outputs.cpu()
            else:
                outputs = model(inputs)
            recommendation_scores = outputs.numpy()[:, 1]  # 预测为1的概率
            # users = inputs[:, 0]
            # items = inputs[:, 1]
            for i in range(length):
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

    for user in test_record.keys():
        x += 1
        print('\r', '{}/{}'.format(x, len(test_record)), end='')
        # dtype = [('itemid', 'int'), ('score', 'float')]
        ground_truth = list(user_record[int(user)])[0]
        test_data = np.array(test_record[user])
        item_score_map = dict()
        item_ids = test_data[:, 0].astype(np.int)
        item_reco_scores = test_data[:, 1]

        for i in range(test_data.shape[0]):
            item_score_map[int(item_ids[i])] = float(item_reco_scores[i])

        pos_predict = item_score_map[ground_truth]
        del item_score_map[ground_truth]
        neg_predict = list(item_score_map.values())
        position = (np.array(neg_predict) >= pos_predict).sum()

        hr = position < 10
        ndcg = math.log(2) / math.log(position + 2) if hr else 0
        if hr:
            hit = 1
        else:
            hit = 0
        del item_score_map

        HR.append(hit)
        NDCG.append(ndcg)

    HR_result = np.array(HR).mean()
    NDCG_result = np.array(NDCG).mean()
    print()

    return HR_result, NDCG_result
