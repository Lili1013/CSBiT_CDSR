import torch
import torch.nn as nn
from peft import PeftModel
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM,AutoModelForCausalLM,AutoTokenizer
from tqdm import tqdm
from loguru import logger
from peft import (
    prepare_model_for_kbit_training,
)
import sys



class Llm_Rep(nn.Module):
    def __init__(self,**args):
        super().__init__()
        self.device = args['device']
        self.base_model = args['base_model']
        self.load_8bit = args['load_8bit']
        self.lora_weights = args['lora_weights']
        self.fine_tune = args['fine_tune']
        if self.fine_tune:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    load_in_8bit=self.load_8bit,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    # attn_implementation="flash_attention_2"
                )
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.lora_weights,
                    torch_dtype=torch.float16,
                    device_map={'': 0}
                )
            elif self.device == "mps":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    device_map={"": self.device},
                    torch_dtype=torch.float16,
                )
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.lora_weights,
                    device_map={"": self.device},
                    torch_dtype=torch.float16,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model, device_map={"": self.device}, low_cpu_mem_usage=True
                )
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.lora_weights,
                    device_map={"": self.device},
                )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    load_in_8bit=self.load_8bit,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            elif self.device == "mps":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    device_map={"": self.device},
                    torch_dtype=torch.float16,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model, device_map={"": self.device}, low_cpu_mem_usage=True
                )

        # Configure tokenizer and model
        self.tokenizer.padding_side = "left"
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 128000
        self.model.config.eos_token_id = 128001
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[UserEmb]', '[ItemEmb]']})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = prepare_model_for_kbit_training(self.model)

        if not self.load_8bit:
            self.model.half()  # Fix bugs for some users
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        # for _, param in self.model.named_parameters():
        #     if 'token' in _:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False


    def get_user_rep(self,inputs):
        # self.model.eval()
        prompt = [self.generate_prompt_user(input) for input in inputs]
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        inputs_embeds = self.model.get_input_embeddings()(inputs['input_ids'])
        # with torch.cuda.amp.autocast():
        with torch.no_grad():
            outputs = self.model.forward(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True
            )

        user_out_token_id = self.tokenizer.convert_tokens_to_ids('[UserEmb]') #128256
        user_out_indices = (inputs['input_ids'] == user_out_token_id).nonzero(as_tuple=True)[1]  # Get the token indices for `[UserOut]`16,22

        # Extract the embeddings from the last hidden state for these indices
        # user_out_embeddings = outputs.hidden_states[-1][0, user_out_indices]
        # logger.info(user_out_indices)
        # logger.info(len(inputs))
        user_out_embeddings = torch.stack([outputs.hidden_states[-1][i, user_out_indices[i]] for i in range(len(prompt))], dim=0)
        return user_out_embeddings
    
    def get_item_rep(self,inputs):
        # self.model.eval()
        prompt = [self.generate_prompt_item(input) for input in inputs]
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        inputs_embeds = self.model.get_input_embeddings()(inputs['input_ids'])
        # with torch.cuda.amp.autocast():
        with torch.no_grad():
            outputs = self.model.forward(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True
            )

        item_out_token_id = self.tokenizer.convert_tokens_to_ids('[ItemEmb]')
        item_out_indices = (inputs['input_ids'] == item_out_token_id).nonzero(as_tuple=True)[1]  # Get the token indices for `[UserOut]`

        # Extract the embeddings from the last hidden state for these indices
        item_out_embeddings = torch.stack([outputs.hidden_states[-1][i, item_out_indices[i]] for i in range(len(prompt))], dim=0)
        return item_out_embeddings

    def generate_prompt_user(self,input=None):
        if input:
            return f"""###Instruction: Based on the user interaction sequences provided, generate user representation:[UserEmb].
    ### Input:
    {input}

    ### Response:
    """
        else:
            return f"""###Instruction: Based on the user interaction sequences provided, generate user representation:[UserEmb].
    ### Response:
    """
    
    def generate_prompt_item(self,input=None):
        if input:
            return f"""###Instruction: Based on the item title provided, generate item representation:[ItemEmb].
    ### Input:
    {input}

    ### Response:
    """
        else:
            return f"""###Instruction: Based on the item title provided, generate item representation:[ItemEmb].
    ### Response:
    """
    

