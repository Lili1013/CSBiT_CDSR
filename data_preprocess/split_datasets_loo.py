
import json
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import csv

def split_datasets(path,sample,n_samples,max_seq_length):
    with open(path,'r') as f:
        user_seq = json.load(f)

    print(len(user_seq))
    if sample:
        sampled_keys = random.sample(list(user_seq.keys()),n_samples)
        user_seq = {key:user_seq[key] for key in sampled_keys}
    train_inter, valid_inter, test_inter = [], [], []
    for key,value in user_seq.items():
        item_IDs = value['item_IDs'][-max_seq_length:][:-2]
        if len(item_IDs) <= 3:
            continue
        item_ids = value['item_ids'][-max_seq_length:][:-2]
        item_titles = value['item_titles'][-max_seq_length:][:-2]
        domain_ids = value['domain_ids'][-max_seq_length:][:-2]
        ratings = value['ratings'][-max_seq_length:][:-2]
        for i in range(2, len(item_IDs)+1):
            # one_data["user"] = uid
            history_item_IDs = item_IDs[:i]
            history_item_ids = item_ids[:i]
            history_item_titles = item_titles[:i]
            history_domain_ids = domain_ids[:i]
            history_ratings = ratings[:i]

            train_inter.append([key,history_item_IDs[:-1],history_item_IDs[-1],
                                history_item_ids[:-1],history_item_ids[-1],
                                history_item_titles[:-1],history_item_titles[-1],
                                history_domain_ids[:-1],history_domain_ids[-1],
                                history_ratings[:-1],history_ratings[-1]
                                ])
        valid_inter.append([key, value['item_IDs'][:-2], value['item_IDs'][-2],
                           value['item_ids'][:-2], value['item_ids'][-2],
                           value['item_titles'][:-2], value['item_titles'][-2],
                           value['domain_ids'][:-2], value['domain_ids'][-2],
                           value['ratings'][:-2], value['ratings'][-2]])
        test_inter.append([key,value['item_IDs'][:-1], value['item_IDs'][-1],
                            value['item_ids'][:-1], value['item_ids'][-1],
                            value['item_titles'][:-1], value['item_titles'][-1],
                            value['domain_ids'][:-1], value['domain_ids'][-1],
                            value['ratings'][:-1],value['ratings'][-1]])
    return train_inter,valid_inter,test_inter

def split_datasets_1(path,sample,n_samples,max_seq_length):
    with open(path,'r') as f:
        user_seq = json.load(f)

    print(len(user_seq))
    # sample_user_seq = {k: v for k, v in user_seq.items() if k not in long_user_seq}
    # n_samples = n_samples - len(long_user_seq)
    # if sample:
    #     sampled_keys = random.sample(list(sample_user_seq.keys()), n_samples)
    #     total_keys = long_user_seq + sampled_keys
    #     user_seq = {key: user_seq[key] for key in total_keys}
    train_inter, valid_inter, test_inter = [], [], []
    for key,value in user_seq.items():
        item_IDs = value['item_IDs'][-max_seq_length:][:-2]
        if len(item_IDs) <= 3:
            continue
        item_ids = value['item_ids'][-max_seq_length:][:-2]
        item_titles = value['item_titles'][-max_seq_length:][:-2]
        domain_ids = value['domain_ids'][-max_seq_length:][:-2]
        ratings = value['ratings'][-max_seq_length:][:-2]
        # for i in range(2, len(item_IDs)+1):
            # one_data["user"] = uid
        history_item_IDs = item_IDs
        history_item_ids = item_ids
        history_item_titles = item_titles
        history_domain_ids = domain_ids
        history_ratings = ratings

        train_inter.append([key,history_item_IDs[:-1],history_item_IDs[-1],
                            history_item_ids[:-1],history_item_ids[-1],
                            history_item_titles[:-1],history_item_titles[-1],
                            history_domain_ids[:-1],history_domain_ids[-1],
                            history_ratings[:-1],history_ratings[-1]
                            ])
        valid_inter.append([key, value['item_IDs'][-max_seq_length:][:-2], value['item_IDs'][-max_seq_length:][-2],
                           value['item_ids'][-max_seq_length:][:-2], value['item_ids'][-max_seq_length:][-2],
                           value['item_titles'][-max_seq_length:][:-2], value['item_titles'][-max_seq_length:][-2],
                           value['domain_ids'][-max_seq_length:][:-2], value['domain_ids'][-max_seq_length:][-2],
                           value['ratings'][-max_seq_length:][:-2], value['ratings'][-max_seq_length:][-2]])
        test_inter.append([key,value['item_IDs'][-max_seq_length:][:-1], value['item_IDs'][-max_seq_length:][-1],
                            value['item_ids'][-max_seq_length:][:-1], value['item_ids'][-max_seq_length:][-1],
                            value['item_titles'][-max_seq_length:][:-1], value['item_titles'][-max_seq_length:][-1],
                            value['domain_ids'][-max_seq_length:][:-1], value['domain_ids'][-max_seq_length:][-1],
                            value['ratings'][-max_seq_length:][:-1],value['ratings'][-max_seq_length:][-1]])
    return train_inter,valid_inter,test_inter




if __name__ == '__main__':
    n_samples = 10000
    flag = 1
    task = 'general'
    dataset = 'beauty'
    if task == 'specific':
        train_to_path = f'../datasets/{task}/{dataset}/train.csv'
        valid_to_path = f'../datasets/{task}/{dataset}/valid.csv'
        test_to_path = f'../datasets/{task}/{dataset}/test.csv'
        seq_path = f'../datasets/{task}/{dataset}/user_seq.txt'
    else:
        train_to_path = f'../datasets/{task}/train.csv'
        valid_to_path = f'../datasets/{task}/valid.csv'
        test_to_path = f'../datasets/{task}/test.csv'
        seq_path = f'../datasets/{task}/user_seq.txt'


    with open(seq_path, 'r') as f:
        user_seq = json.load(f)

    # train_inter,valid_inter,test_inter = split_datasets(path='phone_cloth_sport/user_seq.txt',sample=True,n_samples=n_samples,max_seq_length=30
    if flag == 1:

        train_inter, valid_inter, test_inter = split_datasets_1(
            path=seq_path,
            sample=False,
            n_samples=n_samples, max_seq_length=10)
        with open(train_to_path, 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['user_id', 'history_item_ID', 'item_ID', 'history_item_id', 'item_id', 'history_item_title', 'item_title', 'history_domain_id', 'domain_id','history_rating', 'rating', 'timestamp'])
            csvwriter.writerows(train_inter)
        with open(valid_to_path, 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['user_id', 'history_item_ID', 'item_ID', 'history_item_id', 'item_id', 'history_item_title', 'item_title', 'history_domain_id', 'domain_id','history_rating', 'rating', 'timestamp'])
            csvwriter.writerows(valid_inter)
        with open(test_to_path, 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['user_id', 'history_item_ID', 'item_ID', 'history_item_id', 'item_id', 'history_item_title', 'item_title', 'history_domain_id', 'domain_id','history_rating', 'rating', 'timestamp'])
            csvwriter.writerows(test_inter)
    else:
        train_inter, valid_inter, test_inter = split_datasets(
            path='/data/lwang9/CDR_data_process/data_process_LLM_CDR/datasets/phone_cloth_sport/user_seq.txt',
            sample=True,
            n_samples=n_samples, max_seq_length=20)
        with open(f'phone_cloth_sport/loo/train_loo_{n_samples}.csv', 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['user_id', 'history_item_ID', 'item_ID', 'history_item_id', 'item_id', 'history_item_title', 'item_title', 'history_domain_id', 'domain_id','history_rating', 'rating', 'timestamp'])
            csvwriter.writerows(train_inter)
        with open(f'phone_cloth_sport/loo/valid_loo_{n_samples}.csv', 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['user_id', 'history_item_ID', 'item_ID', 'history_item_id', 'item_id', 'history_item_title', 'item_title', 'history_domain_id', 'domain_id','history_rating', 'rating', 'timestamp'])
            csvwriter.writerows(valid_inter)
        with open(f'phone_cloth_sport/loo/test_loo_{n_samples}.csv', 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['user_id', 'history_item_ID', 'item_ID', 'history_item_id', 'item_id', 'history_item_title', 'item_title', 'history_domain_id', 'domain_id','history_rating', 'rating', 'timestamp'])
            csvwriter.writerows(test_inter)


