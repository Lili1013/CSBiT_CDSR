import gzip
from loguru import logger
import json

#process meta data
def process_meta_data(meta_path):
    g_meta = gzip.open(meta_path, 'r')
    i = 0
    metadata = []
    for line in g_meta:
        d = eval(line, {"true": True, "false": False, "null": None})
        if i % 10000 == 0:
            print(i)
        i += 1
        # if i >= 20000:
        #     break
        # metadata.append([d['asin'], d['title']])
        metadata.append(d)
    logger.info('meta data complete')
    return metadata

meta_path = f'/data/lwang9/datasets/amazon/meta_features'
to_general_path = f'../datasets/general'
to_specific_path = f'../datasets/specific'
# metadata = process_meta_data(
#     meta_path=f'{meta_path}/meta_Cell_Phones_and_Accessories.json.gz')
# metadata = process_meta_data(
#     meta_path=f'{meta_path}/meta_Clothing_Shoes_and_Jewelry.json.gz')
# metadata = process_meta_data(
#     meta_path=f'{meta_path}/meta_Sports_and_Outdoors.json.gz')
# metadata = process_meta_data(
#     meta_path=f'{meta_path}/meta_Health_and_Personal_Care.json.gz')
# metadata = process_meta_data(
#     meta_path=f'{meta_path}/meta_Home_and_Kitchen.json.gz')
# metadata = process_meta_data(
#     meta_path=f'{meta_path}/meta_Beauty.json.gz')
# metadata = process_meta_data(
#     meta_path=f'{meta_path}/meta_Toys_and_Games.json.gz')
metadata = process_meta_data(
    meta_path=f'{meta_path}/meta_Grocery_and_Gourmet_Food.json.gz')


# metadata = meta_1+meta_2+meta_3
id_title = {}
cnt = 0
for meta in metadata:
    try:
        if meta['title']:  # remove the item without title
            id_title[meta['asin']] = meta['title'].replace('\t','').replace('\n','')
    except:
        continue
with open(f'{to_specific_path}/grocery/item_title.txt','w') as f:
    json.dump(id_title,f,indent=4)