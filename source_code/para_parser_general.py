import argparse

def parse():
    parser = argparse.ArgumentParser(description='FedPCL_MDR')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--micro_batch_size', type=int, default=2, metavar='N', help='input batch size for each device')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR', help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=512, metavar='cutoff_len', help='the maximum length of LLM input')
    parser.add_argument('--lora_r', type=int, default=8, metavar='lora_r', help='the rank of LORA')
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=int, default=0.05)
    parser.add_argument('--lora_target_modules', type=list, default=["q_proj", "v_proj"])
    parser.add_argument('--train_on_inputs', type=bool, default=True)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--base_model_path', type=str, default='../DeepSeek-R1-Distill-Llama-8B/')
    parser.add_argument('--task', type=str, default='general_P_C_S_H')
    parser.add_argument('--dataset', type=str, default='beauty')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--group_by_length', type=bool, default=False)


    args = parser.parse_args()
    return args