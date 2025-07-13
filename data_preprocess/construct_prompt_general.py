import pandas as pd
import tqdm
import json
import random
from loguru import logger



def item_pred_one_cand_task(input_path, output_path,item_list,df,id_title,user_seq,k):
    '''
        pair-wise ranking
        :param input_path:
        :param output_path:
        :param item_list:
        :param df:
        :param id_title:
        :return:
        '''
    data = pd.read_csv(input_path)
    # if train_sample:
    #     data = data.sample(n=train_n, random_state=42).reset_index(drop=True)
    # if test_sample:
    #     data = data.sample(n=test_n, random_state=42).reset_index(drop=True)
    #     data.to_csv(output_path[:-5] + ".csv", index=False)
    json_list = []
    for index, row in data.iterrows():
        # Convert strings to Python lists
        row['history_item_title'] = eval(row['history_item_title'])
        row['history_rating'] = eval(row['history_rating'])
        row['history_item_ID'] = eval(row['history_item_ID'])
        row['history_domain_id'] = eval(row['history_domain_id'])

        # Get the interaction history length
        L = len(row['history_item_title'])

        # Ensure there is enough history to mask
        if L < 2:
            continue  # Skip rows with insufficient history

        # Construct the history with the items
        history = ""
        for i in range(L):
            history += f"item ID: {row['history_item_ID'][i]}, Title: {row['history_item_title'][i]}, Domain ID: {row['history_domain_id'][i]}; "
        pos_neg_items = []
        pos_neg_items.append({
            "item_id": row['item_ID'],
            "title": row['item_title'],
            "domain_id": row['domain_id']
        })
        user_inter_item_IDs = user_seq[row['user_id']]['item_IDs']
        neg_item_indexes = random.sample(list(set(item_list) - set(user_inter_item_IDs)), k)
        # item_id = df[df['itemID']==neg_item_index]['item_id'].iloc[0]
        for neg_item_index in neg_item_indexes:
            pos_neg_items.append({
                "item_id": neg_item_index,
                "title": id_title[neg_item_index],
                "domain_id": df[df['itemID'] == neg_item_index]['domain_id'].iloc[0],
            })

        # Prepare the prompt
        # history_prompt = f"The user has interacted with the following items: {masked_history.strip()}"
        # task_instruction = "Given the item list that the user has interacted with, predict the masked items in chronological order."
        # input_prompt = f"The user has interacted with the following items: {history.strip()}. Predict which item the user would buy based on the following two items. {pair_item_prompt}."
        # input_prompt = f"The user has interacted with the following items: {history.strip()}. Based on the user's past interactions, recommend a new item the user is most likely to buy next from the following candidate items. Candidate items: {pair_item_prompt}.",
        # # Prepare the output
        # input_prompt = (f"The user has interacted with the following items: {history.strip()}. "
        #                 f"Based on the user's past interactions, select the top five items the user is most likely to buy next from the following candidate items. Then rank these selected items."
        #                 f"Candidate items: {pair_item_prompt}."),
        input_prompt_list = []
        output_prompt_list = []
        for i in range(len(pos_neg_items)):
            pair_item_prompt = f"".join([
                f"item ID: {pos_neg_items[i]['item_id']}, Title: {pos_neg_items[i]['title']}, Domain ID: {pos_neg_items[i]['domain_id']}"
            ])
            input_prompt = f"The user has interacted with the following items: {history.strip()}. Candidate item: {pair_item_prompt}. "
            # input_prompt_list.append(input_prompt)
            if i == 0:
                output_prompt = '1'
            else:
                output_prompt = '0'
            json_list.append({
                'user': row['user_id'],
                # "Instruction":task_instruction,
                "Input": input_prompt,
                "Output": output_prompt
            })

        # Prepare the output

        # output_prompt = f"; ".join([
        #     f"item ID: {pos_neg_items[0]['item_id']}, Title: {pos_neg_items[0]['title']}, Domain ID: {pos_neg_items[0]['domain_id']}"
        # ])

    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)

split_way = 'loo'

task = 'general'
dataset = 'beauty'
logger.info(task)

root_path = f'../datasets/{task}'


logger.info('item prediction task candidate items')
df = pd.read_csv(f'{root_path}/inter_data.csv')
item_list = list(df['itemID'].unique())

with open(f'{root_path}/ID_title.txt', 'r') as f:
    lines = f.readlines()
id_title = {}
for line in lines:
    record = line.rstrip('\n').split('\t')
    id_title[int(record[0])] = record[1]


with open(f'../datasets_no_filter/{task}/user_seq.txt','r') as f:
    user_seq = json.load(f)


logger.info('save train')
flag = 1
if flag == 1:
    item_pred_one_cand_task(
        f'{root_path}/train.csv',
                        f'{root_path}/train_prompt.json',
                        item_list=item_list, df=df, id_title=id_title, user_seq=user_seq, k=1)

    logger.info('save valid')
    item_pred_one_cand_task(f'{root_path}/valid.csv',
                            f'{root_path}/valid_prompt.json',
                            item_list=item_list, df=df, id_title=id_title, user_seq=user_seq, k=1)

    logger.info('save test')
    item_pred_one_cand_task(f'{root_path}/test.csv',
                            f'{root_path}/test_prompt.json',
                            item_list=item_list, df=df, id_title=id_title, user_seq=user_seq, k=19)

