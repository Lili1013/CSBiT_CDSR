import pandas as pd
from loguru import logger
import gzip
import array

def preprocess(json_file_path,to_path,domain_id):
    ''''
    extract all users and items from the original files
    '''
    # json_gz_file_path = 'C:\work\Amazon_datasets\\Electronics_5.json'

    fin = gzip.open(json_file_path, 'r')
    review_list = []
    i = 0# 存储筛选出来的字段，如果数据量过大可以尝试用dict而不是list
    for line in fin:
        # 顺序读取json文件的每一行
        try:
            if i % 10000 == 0:
                logger.info(i)
            d = eval(line, {"true":True,"false":False,"null":None})
            review_list.append([d['reviewerID'],d['asin'],d['overall'],d['unixReviewTime'],domain_id])
        except:
            continue
        i += 1
    df = pd.DataFrame(review_list, columns =['user_id', 'item_id','rating','time','domain_id']) # 转换为dataframe
    df.to_csv(to_path,index=False)

if __name__ == '__main__':
    ##transform review data into inter data
    general_root_path = f'../datasets/general'
    specifc_root_path = f'../datasets/specific'
    logger.info('process cloth')
    preprocess(json_file_path='/data/lwang9/datasets/amazon/review_texts/reviews_Clothing_Shoes_and_Jewelry_5.json.gz',
               to_path=f'{general_root_path}/cloth/cloth_inter.csv',domain_id=0)
    logger.info('process phone')
    preprocess(json_file_path='/data/lwang9/datasets/amazon/review_texts/reviews_Cell_Phones_and_Accessories_5.json.gz',
               to_path=f'{general_root_path}/phone/phone_inter.csv',domain_id=1)
    logger.info('process sport')
    preprocess(json_file_path='/data/lwang9/datasets/amazon/review_texts/reviews_Sports_and_Outdoors_5.json.gz',
               to_path=f'{general_root_path}/sport/sport_inter.csv',domain_id=2)
    logger.info('process health')
    preprocess(json_file_path='/data/lwang9/datasets/amazon/review_texts/reviews_Health_and_Personal_Care_5.json.gz',
               to_path=f'{general_root_path}/health/health_inter.csv', domain_id=3)
    logger.info('process home')
    preprocess(json_file_path='/data/lwang9/datasets/amazon/review_texts/reviews_Home_and_Kitchen_5.json.gz',
               to_path=f'{general_root_path}/home/home_inter.csv', domain_id=4)

    logger.info('process beauty')
    preprocess(json_file_path='/data/lwang9/datasets/amazon/review_texts/reviews_Beauty_5.json.gz',
               to_path=f'{specifc_root_path}/beauty/beauty_inter.csv', domain_id=5)
    logger.info('process toys')
    preprocess(json_file_path='/data/lwang9/datasets/amazon/review_texts/reviews_Toys_and_Games_5.json.gz',
               to_path=f'{specifc_root_path}/toys/toys_inter.csv', domain_id=6)
    logger.info('process grocery')
    preprocess(json_file_path='/data/lwang9/datasets/amazon/review_texts/reviews_Grocery_and_Gourmet_Food_5.json.gz',
               to_path=f'{specifc_root_path}/grocery/grocery_inter.csv', domain_id=6)


