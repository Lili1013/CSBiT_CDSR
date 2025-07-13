import pandas as pd
import gzip
from loguru import logger
import tqdm
import json



if __name__ == '__main__':
    task = 'general'
    dataset_1 = 'phone'
    dataset_2 = 'cloth'
    dataset_3 = 'sport'
    dataset_4 = 'health'
    dataset_5 = 'home'
    df_1 = pd.read_csv(f'../datasets/{task}/{dataset_1}/{dataset_1}_inter.csv')
    df_2 = pd.read_csv(f'../datasets/{task}/{dataset_2}/{dataset_2}_inter.csv')
    df_3 = pd.read_csv(f'../datasets/{task}/{dataset_3}/{dataset_3}_inter.csv')
    df_4 = pd.read_csv(f'../datasets/{task}/{dataset_4}/{dataset_4}_inter.csv')
    df_5 = pd.read_csv(f'../datasets/{task}/{dataset_5}/{dataset_5}_inter.csv')
    users_1 = set(df_1['user_id'].unique())
    users_2 = set(df_2['user_id'].unique())
    users_3 = set(df_3['user_id'].unique())
    users_4 = set(df_4['user_id'].unique())
    users_5 = set(df_5['user_id'].unique())
    intersection = users_1&users_2&users_3&users_4&users_5
    print(len(intersection))
    df = pd.concat([df_1,df_2,df_3,df_4,df_5],axis=0)


    with open(f'../datasets/{task}/{dataset_1}/item_title.txt','r') as f:
        id_title_1 = json.load(f)
    with open(f'../datasets/{task}/{dataset_2}/item_title.txt','r') as f:
        id_title_2 = json.load(f)
    with open(f'../datasets/{task}/{dataset_3}/item_title.txt','r') as f:
        id_title_3 = json.load(f)
    with open(f'../datasets/{task}/{dataset_4}/item_title.txt','r') as f:
        id_title_4 = json.load(f)
    with open(f'../datasets/{task}/{dataset_5}/item_title.txt','r') as f:
        id_title_5 = json.load(f)
    id_title = {**id_title_1,**id_title_2,**id_title_3,**id_title_4,**id_title_5}
    filter_ids = list(id_title.keys())
    filter_df = df[df['item_id'].isin(filter_ids)]  #('user_id', 'A1RVD8YEZJF89') ('interaction_count', 23)
    interaction_count = filter_df.groupby('user_id').size().reset_index(name='interaction_count')
    users_with_sufficient_interactions = interaction_count[interaction_count['interaction_count']>=5]['user_id']
    filter_df = filter_df[filter_df['user_id'].isin(users_with_sufficient_interactions)]
    filter_df = filter_df.sort_values(by='item_id')
    #
    #item id map
    items = list(filter_df['item_id'].unique())
    item2id = dict()
    count = 0
    for item in items:
        item2id[item] = count
        count += 1
    filter_df['itemID'] = filter_df['item_id'].map(item2id)
    filter_df.to_csv(f'../datasets/{task}/inter_data.csv', index=False)
    logger.info('start process meta data')

    logger.info('start write ID2name.txt')
    with open(f'../datasets/{task}/ID_title.txt',"w") as f:
        for item,item_id in item2id.items():
            if item not in id_title:
                continue
            item_title = id_title[item].replace('\t','').replace('\n','')
            f.write(f"{item_id}\t{item_title}\n")





