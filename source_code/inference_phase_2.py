import sys
import torch
import transformers
import json
import os
from peft import PeftModel
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM,AutoModelForCausalLM,AutoTokenizer
from tqdm import tqdm
from loguru import logger
import random
import pickle
from torch.nn.functional import softmax
import torch.nn.functional as F

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
    lora_weights: str = "",  # Directory containing adapter files
    test_data_path: list=[],
    result_json_data: str = "",
    batch_size: int = 8,
    sample: str=False,
    n_samples: int=5000,
    max_new_tokens: int=1024,
    long_user_seq:list = [],
    test_user_reps_general: dict={},
):
    assert base_model, "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            # attn_implementation="flash_attention_2"
        )
        tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[UserEmb]']})
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={'': 0}
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # Configure tokenizer and model
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 128000
    model.config.eos_token_id = 128001
    # tokenizer.add_special_tokens(
    #     {'additional_special_tokens': ['[UserEmb]']})
    # model.resize_token_embeddings(len(tokenizer))
    user_token_id = tokenizer.convert_tokens_to_ids("[UserEmb]")

    # if not load_8bit:
    #     model.half()  # Fix bugs for some users
    #
    # model.eval()
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    # Evaluation function
    def evaluate(inputs=None, output = None,batch_input_users = None, test_user_reps_general = None, **kwargs):
        # prompt = [generate_prompt(inputs)]
        # aa= model.state_dict()['_orig_mod.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight']
        prompt = [generate_prompt(input) for input in inputs]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        user_ids = batch_input_users
        user_embs = torch.stack(
            [torch.tensor(test_user_reps_general[uid], dtype=torch.float16).to(model.device) for uid in user_ids])
        input_embeds = model.get_input_embeddings()(inputs["input_ids"].to(model.device))

        for inx in range(len(inputs["input_ids"])):
            idx_tensor = (inputs["input_ids"][inx] == user_token_id).nonzero().view(-1)
            input_embeds[inx][idx_tensor] = user_embs[inx].unsqueeze(0)

        batch_label = tokenizer(output, return_tensors="pt", padding=True, truncation=True).to(device)['input_ids'][:, -1]
        generation_config = GenerationConfig(
            # temperature=0,
            top_p=0.9,
            top_k=40,
            num_beams=1,
            num_return_sequences=1,
        )
        with torch.no_grad():
            generation_output = model.generate(
                # **inputs,
                inputs_embeds=input_embeds,
                attention_mask=inputs['attention_mask'].to(model.device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        probs = F.softmax(generation_output.scores[0], dim=-1)
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        logits = generation_output.scores  # Get logits of the last token
        yes_token = tokenizer.encode('1')[1]
        token_probs = [softmax(logit, dim=-1) for logit in logits]
        generated_probs = [token_probs[0][i, yes_token].item() for i in
                           range(len(s))]  # Apply softmax to get the probabilities
        # real_outputs = [_.split('Output: ')[-1] for _ in output]
        # real_outputs = [output_1[i * num_beams: (i + 1) * num_beams] for i in range(len(output_1) // num_beams)]
        return probs,batch_label,output,generated_probs

    # Process test data

    logger.info('start evaluate')
    # result_test_data = []
    for each_test_data_path in test_data_path:
        outputs = []
        output_probs = []
        logger.info(each_test_data_path)
        all_predictions = []
        all_labels = []
        with open(each_test_data_path, 'r') as f:
            test_data = json.load(f)
            if sample:
                sample_to_test_keys = [x['user'] for x in test_data if x['user'] not in long_user_seq]
                n_samples = n_samples - len(long_user_seq)
                sampled_keys = random.sample(sample_to_test_keys,n_samples)
                total_keys = long_user_seq + sampled_keys
                sampled_test_data = [x for x in test_data if x['user'] in total_keys]
            # instructions = [_['instruction'] for _ in test_data]
            else:
                sampled_test_data = test_data
            inputs = [_['Input'] for _ in sampled_test_data]
            output_list = [_['Output'] for _ in sampled_test_data]
            input_users = [_['user'] for _ in sampled_test_data]

            def batch(list, outputs,input_user_list,batch_size=batch_size):
                chunk_size = (len(list) - 1) // batch_size + 1
                for i in range(chunk_size):
                    yield list[batch_size * i: batch_size * (i + 1)],outputs[batch_size * i: batch_size * (i + 1)],input_user_list[batch_size * i: batch_size * (i + 1)]

            # for i, batch in tqdm(enumerate(zip(batch(instructions), batch(inputs)))):
            for i, (batch_input,batch_output,batch_input_users) in tqdm(enumerate(batch(inputs,output_list,input_users))):
                if i % 100 == 0:
                    logger.info(i)
                inputs = batch_input
                probs, batch_label,output,output_prob = evaluate(inputs,batch_output,batch_input_users,test_user_reps_general)
                # all_predictions.append(probs.cpu())
                # all_labels.append(batch_label.cpu())
                outputs = outputs + output
                output_probs = output_probs + output_prob
            for i, test in tqdm(enumerate(sampled_test_data)):
                sampled_test_data[i]['predict'] = outputs[i]
                sampled_test_data[i]['yes_prob'] = output_probs[i]
            logger.info('save results')
            # path = each_test_data_path.split('.')[0] + '_temp.json'
            with open(result_json_data, 'w') as f:
                json.dump(sampled_test_data, f, indent=4)

        # predictions = torch.cat(all_predictions, dim=0)  # (num_users, num_candidates)
        # labels = torch.cat(all_labels, dim=0)
        # num_users = labels.shape[0]
        # acc = 0
        # ndcg_total = 0
        #
        # # Rank items based on model prediction scores
        # sorted_indices = predictions.argsort(dim=-1, descending=True)  # Sort in descending order
        #
        # for i in range(num_users):
        #     true_index = labels[i].item()
        #     pred_item = sorted_indices[i][0].item()  # Extract top-K items for this user
        #
        #     # HR@K: Check if the correct item appears in top-K
        #     if true_index == pred_item:
        #         acc += 1
        # logger.info(f"accuracy:{acc / num_users}")
        # result_test_data.append(test_data)

    # # Save results
    # logger.info('save results')
    # result_json_data_path = []
    # for each in test_data_path:
    #     path = each.split('.')[0]+'_temp.json'
    #     result_json_data_path.append(path)
    # for i in range(len(result_test_data)):
    #     with open(result_json_data_path[i], 'w') as f:
    #         json.dump(result_test_data[i], f, indent=4)


def generate_prompt(input=None):
    if input:
        return f"""Instruction: GIven the user's general preferences and domain-specific interaction history. Predict whether the user will like the candidate item. Output 1 if the user likes the item, or 0 if they dislike it.
### Input: {input}
### Output: """
    else:
        return f"""- GIven the user's general preferences and domain-specific interaction history. Predict whether the user will like the candidate item. Output 1 if the user likes the item, or 0 if they dislike it.
### Output: """



# Entry point
if __name__ == "__main__":
    # Define parameters explicitly
    split_way = 'loo'
    train_n_samples = 256
    test_n_samples = 10000
    valid_n_samples = 10000
    load_8bit = True
    task = 'specific'
    dataset = 'toys'
    weight_path = f'../save_paras_P_C_S_H'
    datasets_path = f'../datasets'
    base_model = "../DeepSeek-R1-Distill-Llama-8B/"  # Replace with your base model path
    # base_model = "Llama-3.1-8B/"
    lora_weights = f"./{weight_path}/{task}/{dataset}/best_model_{dataset}_PCSH"  # Directory containing adapter files
    test_data_path = [f"{datasets_path}/{task}/{dataset}/test_prompt.json"]  # Replace with your test data path
    result_json_data = f"{datasets_path}/{task}/{dataset}/predict_results_PCSH.json"  # Replace with your desired output path

    # lora_weights = f"./{weight_path}/{task}/item_pred_cand_{train_n_samples}"  # Directory containing adapter files
    # test_data_path = [
    #     f"{datasets_path}/{task}/valid_{test_n_samples}_item_pred_rank_cand.json"]  # Replace with your test data path
    # result_json_data = f"{datasets_path}/{task}/predict_{train_n_samples}_result_item_pred_cand.json"  # Replace with your desired output path
    batch_size = 20  # Replace with your desired batch size
    logger.info('start')
    # Call the main function with parameters
    with open('../datasets_no_filter/phone_cloth_sport/long_user_seq.pkl', 'rb') as f:
        long_user_seq = pickle.load(f)
    with open(f'../datasets/specific/{dataset}/test_user_reps_PCSH.json', 'r') as f:
        test_user_reps_general = json.load(f)
    main(
        load_8bit=load_8bit,
        base_model=base_model,
        lora_weights=lora_weights,
        test_data_path=test_data_path,
        result_json_data=result_json_data,
        batch_size=batch_size,
        sample=False,
        n_samples=5000,
        max_new_tokens=1,
        long_user_seq = long_user_seq,
        test_user_reps_general = test_user_reps_general
    )