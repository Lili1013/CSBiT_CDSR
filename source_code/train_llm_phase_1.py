import json
import os

os.environ['LD_LIBRARY_PATH'] = ''
import sys
from typing import List
import random

import numpy as np
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback,GenerationConfig
from safetensors.torch import load_file
import safetensors
from torch.utils.data import DataLoader

from loguru import logger
from para_parser_general import parse


from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer,AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from swanlab.integration.huggingface import SwanLabCallback
import torch.nn.functional as F

def train(
        # model/data params
        base_model,train_data_path,val_data_path,output_dir,refine_tune_model,sample,val_sample,args
        # batch_size: int = 128,
        # micro_batch_size: int = 2,
        # num_epochs: int = 1,
        # learning_rate: float = 3e-4,
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
    logger.info('start')
    logger.info(
        f"Training {args.base_model_path} model with params:\n"
        f"Task: {args.task}\n"
        f"dataset: {args.dataset}\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"val sample: {val_sample}\n"
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
        f"resume_from_checkpoint: {args.resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='DeepSeek-R1-Distill-Llama-8B'"
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

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
        refine_tune_model,
        # quantization_config=bnb_config,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model,padding_side='left')

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

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
        # labels = tokenizer(label_prompt,truncation=True,
        #     max_length=cutoff_len,
        #     padding=False,
        #     return_tensors=None,)
        #
        # # result["labels"] = result["input_ids"].copy()
        # model_inputs["labels"] = labels["input_ids"]
        # model_inputs_1 = tokenizer(
        #     prompt,
        #     truncation=True,
        #     max_length=cutoff_len,
        #     padding="max_length",
        #     return_tensors=None,
        # )
        # labels_1 = tokenizer(label_prompt, truncation=True,
        #                    max_length=10,
        #                    padding="max_length",
        #                    return_tensors=None, )
        # print('gg')
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

    if args.resume_from_checkpoint:
        adapter_model_path = os.path.join(args.resume_from_checkpoint, "adapter_model.safetensors")
        if os.path.exists(adapter_model_path):
            logger.info(f"Loading adapter weights from {adapter_model_path}")
            adapters_weights = safetensors.torch.load_file(adapter_model_path)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            logger.warning(f"Adapter weights not found at {adapter_model_path}")


    logger.info('output trainable parameteres')
    model.print_trainable_parameters()

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # ==============================
    # 4. Evaluation: HR@K & NDCG@K
    # ==============================

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=20,
        num_train_epochs=1,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=8,
        optim="adamw_torch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=args.group_by_length,
        report_to=None,
    )


    def evaluate_accuracy(predictions, labels):
        num_users = labels.shape[0]
        acc = 0
        ndcg_total = 0

        # Rank items based on model prediction scores
        sorted_indices = predictions.argsort(dim=-1, descending=True)  # Sort in descending order

        for i in range(num_users):
            true_index = labels[i].item()
            pred_item = sorted_indices[i][0].item()  # Extract top-K items for this user

            # HR@K: Check if the correct item appears in top-K
            if true_index == pred_item:
                acc += 1



        return acc / num_users

    # ==============================
    # 5. Training with HR@K and NDCG@K Evaluation
    # ==============================


    def evaluate_1():
        input_list = []
        label_list = []
        all_predictions = []
        all_labels = []
        for each in val_data:
            input_list.append(each["Input"])
            label_list.append(each["Output"])
        def batch(input_list, label_list,batch_size=10):
            chunk_size = (len(input_list) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield input_list[batch_size * i: batch_size * (i + 1)],label_list[batch_size * i: batch_size * (i + 1)]

        for i, (batch_input,batch_label) in enumerate(batch(input_list,label_list)):
            prompt = [generate_prompt_eval(input) for input in batch_input]
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(training_args.device)
            batch_label = tokenizer(batch_label, return_tensors="pt", padding=True, truncation=True).to(training_args.device)['input_ids'][:,-1]
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
                    **inputs,
                    # inputs_embeds=input_embeddings,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id = tokenizer.pad_token_id

                )
            # tokenizer.batch_decode(outputs.logits.argsort(dim=-1, descending=True))
            s = generation_output.sequences
            # output = tokenizer.batch_decode(s, skip_special_tokens=True)
            probs = F.softmax(generation_output.scores[0], dim=-1)
            all_predictions.append(probs.cpu())
            all_labels.append(batch_label.cpu())
            # predictions = torch.argmax(probs, dim=-1)

        predictions = torch.cat(all_predictions, dim=0)  # (num_users, num_candidates)
        labels = torch.cat(all_labels, dim=0)  # (num_users,)

        # hr10, ndcg10 = evaluate_metrics(predictions, labels, k=5)
        accuracy = evaluate_accuracy(predictions, labels)
        # print(f"HR@10: {hr10:.4f}, NDCG@10: {ndcg10:.4f}")
        return accuracy


    #visualize parameters
    # swanlab_callback = SwanLabCallback(wandb_project='BiGRec-DeepSeek-R1-Distill-LLama-8B')
    # Training with Early Stopping
    best_hr10 = 0.0  # Best HR@10 performance
    patience = 5  # Early stopping patience
    patience_counter = 0
    best_accuracy = 0.0

    for epoch in range(args.epochs):
        logger.info(f"Starting Epoch {epoch + 1}...")

        # print(
        #     f"Model weights at start of epoch {epoch + 1}: {model.state_dict()['base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight'][0, :10]}")
        # print(
        #     f"Model weights at start of epoch {epoch + 1}: {model.state_dict()['base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'][0, :10]}")

        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=training_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            # compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )
        trainer.train()

        logger.info(f"Evaluating after Epoch {epoch + 1}...")
        accuracy = evaluate_1()
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
            model.save_pretrained(training_args.output_dir)
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience count: {patience_counter}")

        # Stop training if performance does not improve
        if patience_counter >= patience:
            print(f"Early stopping triggered. Best accuracy: {best_accuracy:.4f}")
            break

def generate_prompt(data_point):
    if data_point["Input"]:
        return f"""###Instruction: Predict whether the user will buy the candidate item based on user preferences. Output 1 if the user likes the item, or 0 if they dislike it.
### Input: {data_point["Input"]}
### Output: {data_point["Output"]}"""
    else:
        return f"""###Instruction: Predict whether the user will buy the candidate item based on user preferences. - Output the answer 1 or 0.

### Response:
{data_point["output"]}"""

def generate_prompt_1(data_point):
    if data_point["Input"]:
        return f"""###Instruction: Predict whether the user will buy the candidate item based on user preferences. Output 1 if the user likes the item, or 0 if they dislike it.
### Input: {data_point["Input"]}
### Output: """
    else:
        return f"""###Instruction: Predict whether the user will buy the candidate item based on user preferences. - Output the answer 1 or 0.

### Response:
{data_point["output"]}"""

def generate_prompt_eval(input=None):
    if input:
        return f"""Instruction: Predict whether the user will buy the candidate item based on user preferences. Output 1 if the user likes the item, or 0 if they dislike it.
### Input: {input}
### Output: """
    else:
        return f"""- Select the most relevant item based on user preferences. - Consider domain alignment and similarity to past interactions. - Output the select item in the format: Item ID: <Predicted Item ID>, Title: <Predicted Item Title>, Domain ID: <Predicted Domain ID>.

### Response:"""



def fine_tune_item_pred_rank_cand_general(sample,train_n_samples,valid_n_samples,args):
    source_path = f'../datasets'
    save_path = f'../save_paras'
    task = args.task
    if sample:
        with open(f"{source_path}/{task}/train_prompt.json",'r') as f:
            train_data = json.load(f)
        all_users = [x['user'] for x in train_data]
        all_users = list(set(all_users))
        sample_user_num = int(train_n_samples/2)
        sample_users = random.sample(all_users,sample_user_num)
        train_data_sample = [x for x in train_data if x['user'] in sample_users]
        with open(f'{source_path}/{task}/train_prompt_{train_n_samples}.json','w') as f:
            json.dump(train_data_sample,f,indent=4)

        train(
            base_model=args.base_model_path,
            # base_model='Llama-3.1-8B/',
            train_data_path=[f"{source_path}/{task}/train_prompt_{train_n_samples}.json"],
            val_data_path=[f"{source_path}/{task}/valid_prompt.json"],
            output_dir =f'{save_path}/{task}/best_model_train_{train_n_samples}',
            refine_tune_model = args.base_model_path,
            sample = -1,
            val_sample = valid_n_samples,
            args = args
            # refine_tune_model='Llama-3.1-8B/',
        )
    else:

        train(
            # base_model='../DeepSeek-R1-Distill-Llama-8B/',
            base_model=args.base_model_path,
            # base_model='Llama-3.1-8B/',
            train_data_path=[f"{source_path}/{task}/train_prompt.json"],
            val_data_path=[f"{source_path}/{task}/valid_prompt.json"],
            output_dir=f'{save_path}/{task}/best_model',
            # refine_tune_model='../DeepSeek-R1-Distill-Llama-8B/',
            refine_tune_model=args.base_model_path,
            sample=-1,
            val_sample=valid_n_samples,
            args =args
        )

if __name__ == "__main__":
    args = parse()
    split_way = 'loo'
    train_n_samples = 10000
    valid_n_samples = 1000
    fine_tune_item_pred_rank_cand_general(False, train_n_samples, valid_n_samples,args)
