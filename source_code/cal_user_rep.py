


import torch
import json
import os

import warnings
warnings.filterwarnings("ignore")

import sys
import torch
import transformers
import json
import os
from peft import PeftModel
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM,AutoModelForCausalLM,AutoTokenizer
from tqdm import tqdm
from loguru import logger
import argparse
from peft import (
    prepare_model_for_kbit_training,
)
import re
from llm_rep import Llm_Rep


# Set environment variables for threading
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    test_data_path: list = [],
    train_data_path:list = [],
    result_json_data: str = "temp.json",
    batch_size: int = 8,
    save_user_rep_path: str="",
    n_neg_sample: int = 10,
    fine_tune: str=""
):
    assert base_model, "Please specify a --base_model, e.g. --base_model='DeepSeek-R1-Distill-Llama-8B'"
    params = {
        "device":device,
        "base_model":base_model,
        "load_8bit": True,
        "lora_weights":lora_weights,
        "fine_tune":fine_tune
    }

    llm_rep = Llm_Rep(**params)

    for each_train_data_path in train_data_path:
        outputs = []
        output_probs = []
        logger.info(each_train_data_path)
        with open(each_train_data_path, 'r') as f:
            train_data = json.load(f)
        # inputs = [_['Input'] for _ in train_data]

        # Extract the "Domain-specific interaction history"

        inputs = [train_data[i]['Input'][train_data[i]['Input'].find("Domain-specific interaction history: "):train_data[i]['Input'].find(". Candidate item: ")] for i in range(0,len(train_data),n_neg_sample)]
        input_users = [train_data[i]['user'] for i in range(0,len(train_data),n_neg_sample)]
        # cand_item_inputs = [re.search(r"Candidate item:.*?Title: (.*?)(?:, Domain ID)", x).group(1) for x in inputs]

        def batch(list_input, list_input_users,batch_size=batch_size):
            chunk_size = (len(list_input) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield list_input[batch_size * i: batch_size * (i + 1)],list_input_users[batch_size * i: batch_size * (i + 1)]

        # for i, batch in tqdm(enumerate(zip(batch(instructions), batch(inputs)))):

        user_reps = {}
        item_reps = {}
        for i, (input_batch, input_user_batch) in tqdm(enumerate(batch(inputs,input_users))):
            logger.info(i)
            # if i < 1574:
            #     continue

            user_rep = llm_rep.get_user_rep(input_batch)
            user_rep = user_rep.cpu().tolist()# List of input texts
            for i in range(len(user_rep)):
                if input_user_batch[i] not in user_reps:
                    user_reps[input_user_batch[i]] = user_rep[i]
                else:
                    continue


        logger.info('save embeddings')
        save_embeddings(user_reps, save_user_rep_path)

def save_embeddings(data, save_path):
    """Saves embeddings to a file."""
    with open(save_path, 'w') as f:
        json.dump(data, f,indent=4)

if __name__ == "__main__":
    # Define parameters explicitly

    split_way = 'loo'
    train_n_samples = 10000
    test_n_samples = 10000
    valid_n_samples = 10000
    load_8bit = True
    task = 'specific'
    dataset = 'beauty'
    overlap_ratio = 0.2
    base_model = "../DeepSeek-R1-Distill-Llama-8B/"  # Replace with your base model path
    # lora_weights_general = f"../save_paras_P_C_S_H/general_P_C_S_H/best_model"
    lora_weights_specific = f"../save_paras/specific/{dataset}/best_model_{dataset}"# Directory containing adapter files
    train_data_path = [f"../datasets/{task}/{dataset}/train_prompt.json"]  # Replace with your test data path
    test_data_path = [f"../datasets/{task}/{dataset}/test_prompt.json"]
    valid_data_path = [f"../datasets/{task}/{dataset}/valid_prompt.json"]
    # result_json_data = f"datasets/phone_cloth_sport/{split_way}/temp-{split_way}-1-1024.json"  # Replace with your desired output path
    batch_size = 2  # Replace with your desired batch size
    logger.info('start')

    #for the test, the n_neg_sample is 10; for train, n_neg_sample is 1
    main(
        load_8bit=load_8bit,
        base_model=base_model,
        lora_weights=lora_weights_specific,
        train_data_path=test_data_path,
        result_json_data=[],
        batch_size=batch_size,
        save_user_rep_path = f"../datasets/specific/{dataset}/test_user_reps_specific.json",
        n_neg_sample = 20,
        fine_tune = True
    )

    # with open('datasets/specific/beauty/train_10000_user_reps_general.json','r') as f:
    #     data = json.load(f)
    # print('gg')

