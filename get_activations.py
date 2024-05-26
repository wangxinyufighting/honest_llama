import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
from utils import get_llama_activations_bau, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q, tokenized_triva_qa, get_token_index, tokenized_gsm8k, tokenized_triva_qa_opt, tokenized_triva_qa_v2, tokenized_hotpot_qa, tokenized_hotpot_qa_no_label
import llama
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

HF_NAMES = {
    'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'honest_llama_7B': 'validation/results_dump/llama_7B_seed_42_top_48_heads_alpha_15',
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': '../model/vicuna-7b', 
    'vicuna_13B': '/root/autodl-tmp/model/vicuna-13b', 
    'llama2_chat_7B': '/root/autodl-tmp/model/llama2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
}

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    args = parser.parse_args()

    MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir

    tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
    device = "cuda"
    model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16)
    model.to(device)

    # model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True)
    
    n = 'v5'
    i = 2
    if args.dataset_name == "tqa_mc2": 
        dataset = load_dataset("truthful_qa", "multiple_choice")['validation']
        formatter = tokenized_tqa
    elif args.dataset_name == "tqa_gen": 
        dataset = load_dataset("truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen
    elif args.dataset_name == 'tqa_gen_end_q': 
        dataset = load_dataset("truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen_end_q
    elif args.dataset_name == 'triva_qa_opt':
        dataset = load_dataset(path='json', data_files=f'/root/autodl-fs/prompt_opt/{i}.json')['train']
        formatter = tokenized_triva_qa_opt 
    elif args.dataset_name in ['hpd_train_standard', 'hpd_train_hard', 'hpd_test_standard', 'hpd_test_hard']:
        dataset = load_dataset(path='json', data_files=f'/root/autodl-fs/dataset/trivia_qa/{args.dataset_name}.json')['train']
        formatter = tokenized_triva_qa_v2 
    elif 'test_standard_with_predict_seed' in args.dataset_name: 
        dataset = load_dataset(path='json', data_files=f'/root/autodl-fs/dataset/trivia_qa/v3/random_seed_test/{args.dataset_name}.json')['train']
        formatter = tokenized_triva_qa_v2 
    elif 'trivia_qa_v1' in args.dataset_name:
        dataset = load_dataset(path='json', data_files=f'/root/autodl-fs/dataset/trivia_qa/v1/{args.dataset_name}.json')['train']
        formatter = tokenized_triva_qa_v2
    elif 'trivia_qa_v4' in args.dataset_name:
        dataset = load_dataset(path='json', data_files=f'/root/autodl-fs/dataset/trivia_qa/v4/{args.dataset_name}.json')['train']
        formatter = tokenized_triva_qa_v2
    elif 'trivia_qa_v3' in args.dataset_name:
        dataset = load_dataset(path='json', data_files=f'/root/autodl-fs/dataset/trivia_qa/v3/{args.dataset_name}.json')['train']
        formatter = tokenized_triva_qa_v2
    elif '5+1' in args.dataset_name:
        dataset = load_dataset(path='json', data_files=f'/root/autodl-fs/dataset/trivia_qa/v5/{args.dataset_name}.json')['train']
        formatter = tokenized_triva_qa_v2
    elif 'gsm8k' in args.dataset_name:
        name = args.dataset_name.replace('gsm8k_','')
        dataset = load_dataset(path='json', data_files=f'/root/autodl-tmp/dataset/gsm8k/v3/label/{name}.json')['train']
        formatter = tokenized_gsm8k
    elif 'hotpot_qa' in args.dataset_name:
        name = args.dataset_name.replace('hotpot_qa_','')
        # dataset = load_dataset(path='json', data_files=f'/root/autodl-tmp/dataset/hotpot_qa/v3_3+1/label/{name}.json')['train']
        dataset = load_dataset(path='json', data_files=f'/root/autodl-tmp/dataset/hotpot_qa/v3_1_sample/{name}.json')['train']
        formatter = tokenized_hotpot_qa 

        # dataset = load_dataset(path='json', data_files=f'/root/autodl-tmp/dataset/hotpot_qa/v2/output/{name}.json')['train']
        # formatter = tokenized_hotpot_qa_no_label 
    elif args.dataset_name == 'triva_qa_v1':
        dataset = load_dataset(path='json', data_files='/root/autodl-fs/honest_llama/data/'+n+'.json')['train']
        formatter = tokenized_triva_qa

    else: 
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    if args.dataset_name == "tqa_gen" or args.dataset_name == "tqa_gen_end_q": 
        prompts, labels, categories = formatter(dataset, tokenizer)
        with open(f'features/{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
    else: 
        prompts, labels, qids = formatter(dataset, tokenizer)
        # prompts, qids = formatter(dataset, tokenizer)

    all_layer_wise_activations = []
    all_head_wise_activations = []

    print("Getting activations")
    index = 0
    import time
    all_time = 0

    import os
    if not os.path.exists(f'./features/{args.dataset_name}'):
        os.mkdir(f'./features/{args.dataset_name}')

    # with open(f'features/{args.dataset_name}/{args.model_name}_{args.dataset_name}_layer_wise.txt', 'w') as f, \
    #         open(f'features/{args.dataset_name}/{args.model_name}_{args.dataset_name}_head_wise.txt', 'w') as w

    for prompt in tqdm(prompts):
        index += 1
        start = time.time()
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(model, prompt, device)
        end = time.time()
        all_time += (end-start)
        if index in [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]:
            print(f'count:{index}, time:{all_time:.3f}')
        
        all_layer_wise_activations.append(layer_wise_activations[:,-1,:])
        all_head_wise_activations.append(head_wise_activations[:,-1,:])

        
    print("Saving labels")
    np.save(f'features/{args.dataset_name}/{args.model_name}_{args.dataset_name}_labels.npy', labels)

    print("Saving qids")
    np.save(f'features/{args.dataset_name}/{args.model_name}_{args.dataset_name}_qids.npy', qids)

    print("Saving head wise activations")
    np.save(f'features/{args.dataset_name}/{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations)

    print("Saving layer wise activations")
    np.save(f'features/{args.dataset_name}/{args.model_name}_{args.dataset_name}_layer_wise.npy', all_layer_wise_activations)

if __name__ == '__main__':
    main()