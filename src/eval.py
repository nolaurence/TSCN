import torch
import numpy as np
import math
import torch.nn.functional as F
from preprocess import prepare_batch_input
np.random.seed(1)


def evaluation(model, test_data, user_record, args, n_item):
    # get predictions
    test_record = get_test_record(test_data)
    hit_ratio, ndcg = [], []
    print('working on evaluation ...')
    with torch.no_grad():
        for i in test_data:
            data = np.array(test_data[i])
            np.random.shuffle(data)
            user_list = data[:, 0]
            item_list = data[:, 1]
            labels = torch.from_numpy(data[:, 2])

            user = user_list[0]

            user_input, n_idxs = prepare_batch_input(user_list, item_list, user_record, n_item)
            if torch.cuda.is_available() and args.use_gpu:
                user_input = torch.from_numpy(np.array(user_input)).cuda()
                item_input = torch.from_numpy(np.array(item_list)).cuda()
                n_idx_input = torch.from_numpy(np.array(n_idxs)).cuda()

                outputs = model(outputs = model(user_input, item_input, n_idx_input))
                outputs = outputs.cpu()
            else:
                user_input = torch.from_numpy(np.array(user_input))
                item_input = torch.from_numpy(np.array(item_list))
                n_idx_input = torch.from_numpy(np.array(n_idxs))

                outputs = model(outputs=model(user_input, item_input, n_idx_input))

            recommendation_scores = outputs.numpy()  # 预测为1的概率
            index = np.argsort(-recommendation_scores)
            sorted_result = item_list[index]

            ground_truth = test_record[user]
            position = list(sorted_result).index(ground_truth)

            hr = position < 10
            hit_ratio.append(hr)
            ndcg.append(math.log(2) / math.log(position + 2) if hr else 0)

    hr_result = np.array(hit_ratio).mean()
    ndcg_result = np.array(ndcg).mean()
    return hr_result, ndcg_result


def get_test_record(test_data):
    test_record = {}
    for i in range(len(test_data)):
        data = test_data[i]
        for j in range(data.shape[0]):
            user = data[j, 0]
            item = data[j, 1]
            label = data[j, 2]
            if label == 1:
                test_record[user] = item
    return test_record


# def test(args, data):
#     model_path = '../weights/TSCN_' + args.dataset + '.pth'
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     n_item, train_data, test_data = data[0], data[5], data[6]
#     model = torch.load(model_path)
#     model.to(device)
#     user_record = data[7]
#     # print(1052 in user_record.keys())
#     hit_ratio, ndcg_score = evaluation(model, test_data, user_record, args, n_item)
#     print('HR:{:.4f} NDCG:{:.4f}'.format(hit_ratio, ndcg_score))
