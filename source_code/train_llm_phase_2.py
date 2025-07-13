
import torch
import transformers
import json
import os
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from loguru import logger
from torch.nn.functional import softmax
import re
import sys
from typing import List
from typing import List
import random

import numpy as np
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback,GenerationConfig,AdamW
from safetensors.torch import load_file
import safetensors
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from loguru import logger
import json
from para_parser_specific import parse


from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer,AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from swanlab.integration.huggingface import SwanLabCallback
import torch.nn.functional as F
from bitsandbytes.optim import Adam8bit

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


# **Collate Function for DataLoader**
def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]
    user_ids = [item["user"] for item in batch]
    output_labels = [torch.tensor(int(item["Output"])) for item in batch]
    input_text = [item['Input'] for item in batch]


    max_len = max(len(seq) for seq in input_ids)

    # Pad sequences
    input_ids = [F.pad(seq,(max_len-len(seq),0),value=0) for seq in input_ids]
    attention_mask = [F.pad(seq, (max_len - len(seq), 0), value=0) for seq in attention_mask]
    labels = [F.pad(seq, (max_len - len(seq), 0), value=-100) for seq in labels]
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    labels = torch.stack(labels)
    # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    # attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    # labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # Ignore padding for loss

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "user_ids": user_ids, "output_labels":output_labels, "input_text":input_text}


def main(
        base_model,
        train_data_path,
        val_data_path,
        output_dir,
        refine_tune_model: str,
        # test_user_reps_general: dict,
        train_user_reps_general,
        valid_user_reps_general,
        sample,
        val_sample,
        args
        # seed: int = 0,
        # batch_size: int = 4,
        # micro_batch_size: int = 2,
        # num_epochs: int = 1,
        # cutoff_len: int = 512,
        # lora_r: int = 8,
        # lora_alpha: int = 16,
        # lora_dropout: float = 0.05,
        # lora_target_modules: List[str] = ["q_proj", "v_proj"],
        # train_on_inputs: bool = True,
        # group_by_length: bool = False,
        # wandb_project: str = "",
        # wandb_run_name: str = "",
        # wandb_watch: str = "",
        # wandb_log_model: str = "",
        # resume_from_checkpoint: str = None,

):
    assert base_model, "Please specify a --base_model, e.g. --base_model='DeepSeek-R1-Distill-Llama-8B'"

    logger.info('start')
    logger.info(
        f"Training {args.base_model_path} model with params:\n"
        f"base_model: {base_model}\n"
        f"Task: {args.task}\n"
        f"dataset: {args.dataset}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"val_sample: {val_sample}\n"
        f"seed: {args.seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {args.batch_size}\n"
        f"micro_batch_size: {args.micro_batch_size}\n"
        f"num_epochs: {args.epochs}\n"
        f"learning_rate: {args.lr}\n"
        f"cutoff_len: {args.cutoff_len}\n"
        f"lora_r: {args.lora_r}\n"
        f"lora_alpha: {args.lora_alpha}\n"
        f"lora_dropout: {args.lora_dropout}\n"
        f"lora_target_modules: {args.lora_target_modules}\n"
        f"train_on_inputs: {args.train_on_inputs}\n"
        f"group_by_length: {args.group_by_length}\n"
        # f"wandb_project: {wandb_project}\n"
        # f"wandb_run_name: {wandb_run_name}\n"
        # f"wandb_watch: {wandb_watch}\n"
        # f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {args.resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='DeepSeek-R1-Distill-Llama-8B'"
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    # gradient_accumulation_steps = 4

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=False,
    #     bnb_4bit_compute_dtype=torch.float16
    # )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # quantization_config=bnb_config,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # Configure tokenizer and model
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 128000
    model.config.eos_token_id = 128001
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['[UserEmb]']})
    model.resize_token_embeddings(len(tokenizer))

    # Get `[UserEmb]` token ID
    user_token_id = tokenizer.convert_tokens_to_ids("[UserEmb]")

    def tokenize(prompt, add_eos_token=True):
        model_inputs = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            # padding="max_length",
            padding=False,
            return_tensors=None,
        )
        if (
                model_inputs["input_ids"][-1] != tokenizer.eos_token_id
                and len(model_inputs["input_ids"]) < args.cutoff_len
                and add_eos_token
        ):
            model_inputs["input_ids"].append(tokenizer.eos_token_id)
            model_inputs["attention_mask"].append(1)
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        return model_inputs

    def generate_and_tokenize_prompt(data_point):

        input_prompt = generate_prompt(data_point)
        # output_prompt = data_point['Output']

        tokenized_full_prompt = tokenize(input_prompt)
        if not args.train_on_inputs:
            user_prompt = generate_prompt_1(data_point)
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]
        return tokenized_full_prompt

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    train_data_list = []
    val_data_list = []
    logger.info('process train and validation data')
    for path in train_data_path:
        if path.endswith(".json"):
            train_data_list.append(load_dataset("json", data_files=path))
        else:
            train_data_list.append(load_dataset(path))

    for path in val_data_path:
        if path.endswith(".json"):
            val_data_list.append(load_dataset("json", data_files=path))
        else:
            val_data_list.append(load_dataset(path))

    for i in range(len(train_data_list)):
        train_data_list[i]["train"] = train_data_list[i]["train"].shuffle(seed=args.seed).select(
            range(sample)) if sample > -1 else train_data_list[i]["train"].shuffle(seed=args.seed)
        # train_data_list[i]["train"] = train_data_list[i]["train"].shuffle(seed=seed)
        # tokenized_fuaall_prompt =  generate_and_tokenize_prompt(train_data_list[i]["train"][0])
        train_data_list[i] = train_data_list[i].map(lambda x: generate_and_tokenize_prompt(x))
    for i in range(len(val_data_list)):
        val_data_list[i]["train"] = val_data_list[i]["train"].select(
            range(val_sample)) if val_sample > -1 else val_data_list[i]["train"]
        val_data_list[i] = val_data_list[i].map(lambda x: generate_and_tokenize_prompt(x))
    train_data = concatenate_datasets([_["train"] for _ in train_data_list])
    val_data = concatenate_datasets([_["train"] for _ in val_data_list])
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                pin_memory=True)

    if args.resume_from_checkpoint:
        adapter_model_path = os.path.join(args.resume_from_checkpoint, "adapter_model.safetensors")
        if os.path.exists(adapter_model_path):
            logger.info(f"Loading adapter weights from {adapter_model_path}")
            adapters_weights = safetensors.torch.load_file(adapter_model_path)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            logger.warning(f"Adapter weights not found at {adapter_model_path}")

    logger.info(f'output trainable parameteres: {model.print_trainable_parameters()}')


    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # ==============================
    # 4. Evaluation: HR@K & NDCG@K
    # ==============================

    def evaluate_accuracy(predictions, labels):
        num_users = labels.shape[0]
        acc = 0
        ndcg_total = 0

        # Rank items based on model prediction scores
        sorted_indices = predictions.argsort(dim=-1, descending=True)  # Sort in descending order

        for i in range(num_users):
            true_index = labels[i].item()
            if true_index == 1:
                true_index = 16
            if true_index == 0:
                true_index = 15
            pred_item = sorted_indices[i][0].item()  # Extract top-K items for this user

            # HR@K: Check if the correct item appears in top-K
            if true_index == pred_item:
                acc += 1

        return acc / num_users

    # ==============================
    # 5. Training with HR@K and NDCG@K Evaluation
    # ==============================

    def evaluate_1(tokenizer):
        all_predictions = []
        all_labels = []
        for batch in val_dataloader:
            user_ids = batch["user_ids"]
            output_labels = batch['output_labels']
            user_embs = torch.stack(
                [torch.tensor(valid_user_reps_general[uid], dtype=torch.float16).to(model.device) for uid in user_ids])
            input_embeds = model.get_input_embeddings()(batch["input_ids"][:, 0:-2].to(model.device))

            for inx in range(len(batch["input_ids"])):
                idx_tensor = (batch["input_ids"][inx] == user_token_id).nonzero().view(-1)
                input_embeds[inx][idx_tensor] = user_embs[inx].unsqueeze(0)
            generation_config = GenerationConfig(
                # temperature=1.0,
                top_p=0.9,
                top_k=40,
                num_beams=1,
                num_return_sequences=1,
            )
            with torch.no_grad():
                # logger.info(
                #     f"Model weights at start of epoch {epoch + 1}: {model.state_dict()['base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'][0, :10]}")
                generation_output = model.generate(
                    # **inputs,
                    inputs_embeds=input_embeds,
                    attention_mask=batch['attention_mask'][:, 0:-2].to(model.device),
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,

                )
                # tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
            probs = F.softmax(generation_output.scores[0], dim=-1)
            all_predictions.append(probs.cpu())
            all_labels.append(torch.tensor(output_labels))
            # break

        predictions = torch.cat(all_predictions, dim=0)  # (num_users, num_candidates)
        labels = torch.cat(all_labels, dim=0)  # (num_users,)

        # hr10, ndcg10 = evaluate_metrics(predictions, labels, k=5)
        accuracy = evaluate_accuracy(predictions, labels)
        # print(f"HR@10: {hr10:.4f}, NDCG@10: {ndcg10:.4f}")
        return accuracy

    # visualize parameters
    # swanlab_callback = SwanLabCallback(wandb_project='BiGRec-DeepSeek-R1-Distill-LLama-8B')
    # Training with Early Stopping
    best_hr10 = 0.0  # Best HR@10 performance
    patience = 5  # Early stopping patience
    patience_counter = 0
    best_accuracy = 0.0
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    for epoch in range(args.epochs):
        logger.info(f"Starting Epoch {epoch + 1}...")

        model.train()
        total_loss = 0.0
        # progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        # lora_q_A_weight_init = model.state_dict()[
        #     'base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'].data.clone()
        # lora_q_B_weight_init = model.state_dict()[
        #     'base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight'].data.clone()
        # lora_v_A_weight_init = model.state_dict()[
        #     'base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight'].data.clone()
        # lora_v_B_weight_init = model.state_dict()[
        #     'base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight'].data.clone()
        for i, batch in enumerate(train_dataloader, 0):
            optimizer.zero_grad()

            # **Replace [UserEmb] token embeddings**
            user_ids = batch["user_ids"]
            user_embs = torch.stack(
                [torch.tensor(train_user_reps_general[uid], dtype=torch.float16).to(model.device) for uid in user_ids])
            input_embeds = model.get_input_embeddings()(batch["input_ids"].to(model.device))

            for inx in range(len(batch["input_ids"])):
                idx_tensor = (batch["input_ids"][inx] == user_token_id).nonzero().view(-1)
                input_embeds[inx][idx_tensor] = user_embs[inx].unsqueeze(0)

            # model.set_input_embeddings(torch.nn.Embedding.from_pretrained(input_embeds, freeze=False))

            # **Move batch to device**
            # inputs = {k: v.to(device) for k, v in batch.items() if k != "user_ids"}
            # outputs = model(**inputs)
            # with autocast():
            outputs = model(
                inputs_embeds=input_embeds,
                attention_mask=batch["attention_mask"].to(model.device),
                return_dict=True,
                labels=batch["labels"].to(model.device),
            )
            loss = outputs.loss
            # scaler.scale(loss).backward()  # Scale the loss for mixed precision
            # scaler.step(optimizer)  # Step the optimizer
            # scaler.update()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # torch.cuda.empty_cache()
            # logger.info(loss.item())
            total_loss += loss.item()
            # break
        # lora_q_A_weight = model.state_dict()[
        #     'base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'].data.clone()
        # lora_q_B_weight = model.state_dict()[
        #     'base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight'].data.clone()
        # lora_v_A_weight = model.state_dict()[
        #     'base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight'].data.clone()
        # lora_v_B_weight = model.state_dict()[
        #     'base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight'].data.clone()
        # if torch.equal(lora_q_A_weight, lora_q_A_weight_init):
        #     logger.info('lora_q_A_weight dont update!')
        # if torch.equal(lora_q_B_weight, lora_q_B_weight_init):
        #     logger.info('lora_q_B_weight dont update!')
        # if torch.equal(lora_v_A_weight, lora_v_A_weight_init):
        #     logger.info('lora_v_A_weight dont update!')
        # if torch.equal(lora_v_B_weight, lora_v_B_weight_init):
        #     logger.info('lora_v_B_weight dont update!')
        # else:
        #     logger.info('all update')
        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}")

        logger.info(f"Evaluating after Epoch {epoch + 1}...")
        accuracy = evaluate_1(tokenizer)
        # logger.info(f"HR@5: {hr10:.4f}, NDCG@5: {ndcg10:.4f}")
        logger.info(f"accuracy: {accuracy:.4f}")

        # Save best model based on HR@10
        # global best_hr10, patience_counter
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            logger.info(f"New best accuracy found! Saving model...")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"{output_dir} has been created!")
            else:
                logger.info(f"{output_dir} hase been existed!")

            model.save_pretrained(output_dir)
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience count: {patience_counter}")

        # Stop training if performance does not improve
        if patience_counter >= patience:
            print(f"Early stopping triggered. Best accuracy: {best_accuracy:.4f}")
            break


def generate_prompt(data_point):
    if data_point["Input"]:
        return f"""###Instruction: GIven the user's general preferences and domain-specific interaction history. Predict whether the user will like the candidate item. Output 1 if the user likes the item, or 0 if they dislike it.
### Input: {data_point["Input"]}
### Output: {data_point["Output"]}"""
    else:
        return f"""###Instruction: Predict whether the user will buy the candidate item based on user preferences. - Output the answer 1 or 0.
### Output:
{data_point["output"]}"""

def generate_prompt_1(data_point):
    if data_point["Input"]:
        return f"""###Instruction: GIven the user's general preferences and domain-specific interaction history. Predict whether the user will like the candidate item. Output 1 if the user likes the item, or 0 if they dislike it.
### Input: {data_point["Input"]}
### Output: """
    else:
        return f"""###Instruction: Predict whether the user will buy the candidate item based on user preferences. - Output the answer 1 or 0.
### Output:
{data_point["output"]}"""

def generate_prompt_eval(input=None):
    if input:
        return f"""Instruction: GIven the user's general preferences and domain-specific interaction history. Predict whether the user will like the candidate item. Output 1 if the user likes the item, or 0 if they dislike it.
### Input: {input}
### Output: """
    else:
        return f"""- Select the most relevant item based on user preferences. - Consider domain alignment and similarity to past interactions. - Output the select item in the format: Item ID: <Predicted Item ID>, Title: <Predicted Item Title>, Domain ID: <Predicted Domain ID>.

### Response:"""


# Entry point
if __name__ == "__main__":
    args = parse()
    # Define parameters explicitly
    split_way = 'loo'
    train_n_samples = 0
    test_n_samples = 10000
    valid_n_samples = 10000
    load_8bit = True
    # overlap_ratio = 0.2
    # base_model = "../Llama-3.1-8B/"  # Replace with your base model path
    # base_model = "../DeepSeek-R1-Distill-Llama-8B/"
    # test_data_path = [f"../datasets/specific/beauty/test_{valid_n_samples}_item_pred_rank_cand_phase_2.json"]  # Replace with your test data path
    # result_json_data = f"../validate_rep_items/datasets/phone_cloth_sport/result_{test_n_samples}_item_cand_ds_new.json"  # Replace with your desired output path
    # batch_size = 10  # Replace with your desired batch size
    logger.info('start')
    source_path = f'../datasets'
    save_path = f'../save_paras'
    task = args.task
    dataset = args.dataset

    # Call the main function with parameters
    # with open(f'{source_path}/{task}/{dataset}/test_user_reps_general.json', 'r') as f:
    #     test_user_reps_general = json.load(f)
    with open(f'{source_path}/{task}/{dataset}/train_user_reps.json', 'r') as f:
        train_user_reps_general = json.load(f)
    with open(f'{source_path}/{task}/{dataset}/valid_user_reps.json', 'r') as f:
        valid_user_reps_general = json.load(f)
    logger.info(f'train_n_sample:{train_n_samples}')

    if train_n_samples > 0:
        with open(f"{source_path}/{task}/{dataset}/train_prompt.json",'r') as f:
            train_data = json.load(f)
        if not os.path.exists(f'train_prompt_{train_n_samples}.json'):
            all_users = [x['user'] for x in train_data]
            all_users = list(set(all_users))
            sample_user_num = int(train_n_samples/2)
            sample_users = random.sample(all_users,sample_user_num)
            train_data_sample = [x for x in train_data if x['user'] in sample_users]
            logger.info(f'the number of train sample:{len(train_data_sample)}')
            with open(f'{source_path}/{task}/{dataset}/train_prompt_{train_n_samples}.json','w') as f:
                json.dump(train_data_sample,f,indent=4)

        main(
            base_model=args.base_model_path,
            # base_model='Llama-3.1-8B/',
            train_data_path=[f"{source_path}/{task}/{dataset}/train_prompt_{train_n_samples}.json"],
            val_data_path=[f"{source_path}/{task}/{dataset}/valid_prompt.json"],
            output_dir=f'{save_path}/{task}/{dataset}/best_model_{dataset}_{train_n_samples}',
            refine_tune_model=args.base_model_path,
            # test_user_reps_general=test_user_reps_general,
            train_user_reps_general=train_user_reps_general,
            valid_user_reps_general = valid_user_reps_general,
            sample=-1,
            val_sample=1000,
            args=args
        )
    else:
        main(
            base_model=args.base_model_path,
            # base_model='Llama-3.1-8B/',
            train_data_path=[f"{source_path}/{task}/{dataset}/train_prompt.json"],
            val_data_path=[f"{source_path}/{task}/{dataset}/valid_prompt.json"],
            output_dir=f'{save_path}/{task}/{dataset}/best_model_{dataset}',
            refine_tune_model=args.base_model_path,
            # test_user_reps_general=test_user_reps_general,
            train_user_reps_general=train_user_reps_general,
            valid_user_reps_general=valid_user_reps_general,
            sample=-1,
            val_sample=1000,
            args=args
        )

