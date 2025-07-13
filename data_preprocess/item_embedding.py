from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from loguru import logger
from tqdm import tqdm

logger.info('start transform')

# Load model and tokenizer
base_model = "/data/lwang9/LLM_CDR/DeepSeek-R1-Distill-Llama-8B/"
# base_model = "/data/lwang9/LLM_CDR/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Configure tokenizer and model
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 128000
model.config.eos_token_id = 128001
model.eval()
task = 'general'
dataset = 'toys'
# Load item names
if 'general' in task:
    with open(f'../datasets/{task}/ID_title.txt', 'r') as f:
        lines = f.readlines()
else:
    with open(f'../datasets/{task}/{dataset}/ID_title.txt', 'r') as f:
        lines = f.readlines()

text = [_.split('\t')[1].strip(" ").strip('\"').rstrip('\n') for _ in lines]  # Preprocess item names


# Batch processing function
tokenizer.padding_side = "left"
from tqdm import tqdm
def batch(list, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]
item_embedding = []
for i, batch_input in tqdm(enumerate(batch(text, 4))):
    logger.info(i)
    input = tokenizer(batch_input, return_tensors="pt", padding=True)
    input_ids = input.input_ids
    attention_mask = input.attention_mask
    outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    item_embedding.append(hidden_states[-1][:, -1, :].detach().cpu())
    # break
item_embedding = torch.cat(item_embedding, dim=0)
logger.info('save results')
if 'general' in task:
    torch.save(item_embedding, f'../datasets/{task}/item_embedding_ds_8B.pt')
else:
    torch.save(item_embedding, f'../datasets/{task}/{dataset}/item_embedding_ds_8B.pt')
