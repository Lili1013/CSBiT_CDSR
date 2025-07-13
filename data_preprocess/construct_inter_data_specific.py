import pandas as pd
import gzip
from loguru import logger
import tqdm
import json



if __name__ == '__main__':
    df_cloth = pd.read_csv('../datasets/general/cloth/cloth_inter.csv')
    df_health = pd.read_csv('../datasets/general/health/health_inter.csv')
    df_home = pd.read_csv('../datasets/general/home/home_inter.csv')
    df_phone = pd.read_csv('../datasets/general/phone/phone_inter.csv')
    df_sport = pd.read_csv('../datasets/general/sport/sport_inter.csv')
    cloth_users = list(df_cloth['user_id'].unique())
    health_users = list(df_health['user_id'].unique())
    home_users = list(df_home['user_id'].unique())
    phone_users = list(df_phone['user_id'].unique())
    sport_users = list(df_sport['user_id'].unique())
    cloth_items = list(df_cloth['item_id'].unique())
    health_items = list(df_health['item_id'].unique())
    home_items = list(df_home['item_id'].unique())
    phone_items = list(df_phone['item_id'].unique())
    sport_items = list(df_sport['item_id'].unique())
    general_users = set(cloth_users+health_users+home_users+phone_users+sport_users)
    general_items = set(cloth_items+health_items+home_items+phone_items+sport_items)

    task = 'specific'
    dataset = 'beauty'

    df = pd.read_csv(f'../datasets/{task}/{dataset}/{dataset}_inter.csv')
    common_users = set(df['user_id'].unique()).intersection(general_users)
    common_items = set(df['item_id'].unique()).intersection(general_items)

    with open(f'../datasets/{task}/{dataset}/item_title.txt','r') as f:
        id_title = json.load(f)

    filter_df_common = df[(~df['user_id'].isin(common_users))&(~df['item_id'].isin(common_items))]
    filter_ids = list(id_title.keys())
    filter_df = filter_df_common[filter_df_common['item_id'].isin(filter_ids)]  #('user_id', 'A1RVD8YEZJF89') ('interaction_count', 23)
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
    filter_df.to_csv(f'../datasets/{task}/{dataset}/inter_data.csv', index=False)
    logger.info('start process meta data')

    logger.info('start write ID2name.txt')
    with open(f'../datasets/{task}/{dataset}/ID_title.txt',"w") as f:
        for item,item_id in item2id.items():
            if item not in id_title:
                continue
            item_title = id_title[item].replace('\t','').replace('\n','')
            f.write(f"{item_id}\t{item_title}\n")





