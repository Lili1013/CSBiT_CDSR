#--------------------------For sampling rep items, select popular items in each cluster----------


import pandas as pd
import torch
import pickle
from sklearn.cluster import DBSCAN
import json
from loguru import logger


# with open('datasets/phone_cloth_sport/item_embedding.pt','rb') as f:
#     item_embeddings = pickle.load(f)

def cluster_dbscan(item_embeddings,item_IDs):
    item_embeddings = item_embeddings.cpu().numpy()
    dbscan = DBSCAN(eps=0.3,min_samples=3,metric='cosine')
    labels = dbscan.fit_predict(item_embeddings)
    unique_clusters = set(labels)
    cluster_dict = {cluster:[] for cluster in unique_clusters}
    for item_index,cluster_label in enumerate(labels):
        cluster_dict[cluster_label].append(item_IDs[item_index])
    return cluster_dict

def item_sampling(user_dict,df):
    item_inter_cnt_df = df.groupby('itemID').size().reset_index(name='interaction_count')
    item_inter_cnt_dict = item_inter_cnt_df.set_index('itemID')['interaction_count'].to_dict()
    user_dict_new = {}
    for user, seq in user_dict.items():
        total_items = sum(len(items) for items in seq.values())
        new_seq = []
        for cluster_id, cluster_items_value in seq.items():
            if cluster_id == -1:
                new_seq.extend(cluster_items_value)
            else:
                cluster_items = cluster_items_value
                # last_two_items = cluster_items_value[-2:]
                sample_ratio = len(cluster_items)/total_items
                sort_cluster_items = sorted(cluster_items,key=lambda item:item_inter_cnt_dict.get(item,0),reverse=True)
                sample_items = sort_cluster_items[0:round(len(cluster_items)*sample_ratio)]
                new_seq.extend(sample_items)
                # new_seq.extend(last_two_items)
        last_two_items = list(df[df['user_id']==user].sort_values('time')['itemID'])[-2:]
        new_seq.extend(last_two_items)
        user_dict_new[user] = new_seq
    return user_dict_new


task = 'general_P_C_S_H'
overlap_ratio = 0.4
dataset = 'toys'
logger.info('read item embeddings')
if task == 'specific':
    item_embeddings = torch.load(f'/data/lwang9/CDR_data_process/data_process_LLM_CDR/datasets/{task}/{dataset}/item_embedding_ds_8B.pt')
else:
    item_embeddings = torch.load(
        f'/data/lwang9/CDR_data_process/data_process_LLM_CDR/datasets/{task}/item_embedding_ds_8B.pt')

logger.info('read all inter data')
if task == 'specific':
    df = pd.read_csv(f'/data/lwang9/CDR_data_process/data_process_LLM_CDR/datasets/{task}/{dataset}/inter_data.csv')
else:
    df = pd.read_csv(f'/data/lwang9/CDR_data_process/data_process_LLM_CDR/datasets/{task}/inter_data.csv')

interaction_count = df.groupby('user_id').size().reset_index(name='interaction_count')
user_dict = {}
logger.info('read item titles')
if task == 'specific':
    with open(f'/data/lwang9/CDR_data_process/data_process_LLM_CDR/datasets/{task}/{dataset}/ID_title.txt', 'r') as f:
        lines = f.readlines()
else:
    with open(f'/data/lwang9/CDR_data_process/data_process_LLM_CDR/datasets/{task}/ID_title.txt', 'r') as f:
        lines = f.readlines()

ID_title = {}
for line in lines:
    record = line.rstrip('\n').split('\t')
    ID_title[int(record[0])] = record[1]

num = 0
logger.info('start clustering')
for index,line in df.groupby('user_id'):
    # if index != 'A00428403I6YA6YYRHJD8':
    #     continue
    sorted_line = line.sort_values('time')
    group_item_IDs = list(sorted_line['itemID'])[0:-2]
    group_item_embs = item_embeddings[group_item_IDs]
    cluster_dict = cluster_dbscan(group_item_embs,group_item_IDs)
    # for key,value in cluster_dict.items():
    #     value.extend(list(sorted_line['itemID'])[-2:])
    #     cluster_dict[key] = value
    user_dict[index] = cluster_dict
    # if num==100:
    #     break
    # num+=1
    # group_item_ids = list(line['item_id'])
    # domain_ids = list(line['domain_id'])
    # titles = []
    # for each_item_id in group_item_ids:
    #     title = id_title[each_item_id]
    #     titles.append(title)
logger.info('start item sampling')

user_dict_new = item_sampling(user_dict,df)
user_dict_all = {}
logger.info('construct new user seq')
cnt = 0
for user, seq in user_dict_new.items():
    # if user != 'A04673051X8PCSPHE4E79':
    #     continue [36651, 89986, 89988, 43555, 86234, 49632]
    if cnt % 10000==0:
        logger.info(cnt)
    if user not in user_dict_all:
        user_dict_all[user] = {
            'item_ids': [],
            'item_IDs':[],
            'item_titles':[],
            'ratings': [],
            'timestamps': [],
            'domain_ids': []
        }
    item_IDs = seq
    new_df = df[(df['user_id'] == user) & (df['itemID'].isin(item_IDs))].sort_values(by=['time','itemID'])
    item_ids = list(new_df['item_id'])
    domain_ids = list(new_df['domain_id'])
    ratings = list(new_df['rating'])
    timestamps = list(new_df['time'])
    item_titles = [ID_title[x] for x in item_IDs]
    sorted_item_IDs = list(new_df['itemID'])
    # all = list(zip(item_ids,item_IDs, ratings, timestamps, domain_ids,item_titles))
    # res = sorted(all, key=lambda x: int(x[-3]))
    # item_ids,item_IDs, ratings, timestamps, domain_ids,item_titles = zip(*res)
    # item_ids,item_IDs, ratings, timestamps, domain_ids,item_titles = list(item_ids),list(item_IDs), list(ratings), list(timestamps), list(domain_ids),list(item_titles)
    user_dict_all[user]['item_ids'] = item_ids
    user_dict_all[user]['item_IDs'] = sorted_item_IDs
    user_dict_all[user]['item_titles'] = item_titles
    user_dict_all[user]['ratings'] = ratings
    user_dict_all[user]['timestamps'] = timestamps
    user_dict_all[user]['domain_ids'] = domain_ids

    cnt+=1
logger.info('write user seq')
if task == 'specific':
    with open(f'{task}/{dataset}/user_seq.txt','w') as f:
        json.dump(user_dict_all,f,indent=4)
else:
    with open(f'{task}/user_seq.txt','w') as f:
        json.dump(user_dict_all,f,indent=4)



