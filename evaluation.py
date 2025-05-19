import re
import sys
import io, os
import torch
import torch.nn.functional as F
import numpy as np
import logging
import tqdm
import fcntl
import time
import argparse
from prettytable import PrettyTable
import transformers
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pdb
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def lock_and_write_file(file_path, content):
    with open(file_path, 'a') as file:
        while True:
            try:
                # Acquire an exclusive lock (non-blocking)
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Perform your write operations here
                file.write(content + '\n')
                file.flush()

            except IOError as e:
                print("File is locked by another process. Can't write.")
                time.sleep(1)
            finally:
                # Release the lock
                fcntl.flock(file, fcntl.LOCK_UN)
                break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str,
                        help="Transformers' model name or path")
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument('--tensor_parallel', action='store_true')
    parser.add_argument('--prompt_method', type=str, default='prompteol', help="What prompt method to use (prompteol/metaeol).")
    parser.add_argument('--ratio', type=float, default=0.3)
    parser.add_argument('--contra', action='store_true', help="Whether to use contrastive layers")
    parser.add_argument('--selection', type=int)



    args = parser.parse_args()

    if args.tensor_parallel:
        import tensor_parallel as tp
        n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     low_cpu_mem_usage = True, torch_dtype=torch.float16)
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     device_map='auto',
                                                     output_hidden_states=True,
                                                     trust_remote_code=True,
                                                     max_memory={0: "16GB"},
                                                     offload_folder="./offload",
                                                    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = 0  # Set the padding token. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        if args.mode == 'dev':
            args.tasks = ['STSBenchmark-dev']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 8}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 8,
                                         'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size':2}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    if args.prompt_method == "metaeol":
        task_prompts = ["In this task, you're presented with a text excerpt. Your task is to categorize the excerpt into a broad category such as 'Education', 'Technology', 'Health', 'Business', 'Environment', 'Politics', or 'Culture'. These categories help in organizing content for better accessibility and targeting. For this task, this sentence : \"*sent 0*\" should be classified under one general category in one word:\"",
                        "In this task, you're given a statement and you need to determine whether it's presenting an 'Opinion' or a 'Fact'. This distinction is vital for information verification, educational purposes, and content analysis. For this task, this sentence : \"*sent 0*\" discriminates between opinion and fact in one word:\"",
                        "In this task, you're given a review from an online platform. Your task is to generate a rating for the product based on the review on a scale of 1-5, where 1 means 'extremely negative' and 5 means 'extremely positive'. For this task, this sentence : \"*sent 0*\" reflects the sentiment in one word:\"",
                        "In this task, you're reading a personal diary entry. Your task is to identify the predominant emotion expressed, such as joy, sadness, anger, fear, or love. For this task, this sentence : \"*sent 0*\" conveys the emotion in one word:\"",
                        "In this task, you're presented with two sentences. Your task is to assess whether the sentences convey the same meaning. Use 'identical', 'similar', 'different', or 'unrelated' to describe the relationship. To enhance the performance of this task, this sentence : \"*sent 0*\" means in one word:\"",
                        "In this task, you're given a sentence and a phrase. Your task is to determine if the phrase can be a contextual synonym within the given sentence. Options include 'yes', 'no', or 'partially'. To enhance the performance of this task, this sentence : \"*sent 0*\" means in one word:\"",
                        "In this task, you're examining a news article. Your task is to extract the most critical fact from the article. For this task, this sentence : \"*sent 0*\" encapsulates the key fact in one word:\"",
                        "In this task, you're reviewing a scientific abstract. Your task is to identify the main entities (e.g., proteins, diseases) and their relations (e.g., causes, treats). For this task, this sentence : \"*sent 0*\" highlights the primary entity or relation in one word:\"",
                        ]
    elif args.prompt_method == "prompteol":
        task_prompts = ["This sentence : \"*sent 0*\" means in one word:\""]
    elif args.prompt_method == "cot":
        task_prompts = ['After thinking step by step , this sentence : \"*sent 0*\" means in one word:"']
    elif args.prompt_method == "knowledge":
        task_prompts = ['The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind , this sentence : \"*sent 0*\" means in one word:"']
            
    def process_hidden_states(hidden_states, logits): 
        # length of the hidden_states is the number of layers + 1 (for the input embeddings)       
        with torch.no_grad():     
            # hidden_states[-1]  # [batch_size, seq_length, hidden_dim]
            last_token_state = hidden_states[-1][:, -1, :]  # [batch_size, hidden_dim]
            last_token_norm = F.normalize(last_token_state, dim=-1)  # [batch_size, hidden_dim]
                            
            if not args.contra:                 
                output = last_token_norm                 
                if output.dtype == torch.bfloat16:                     
                    output = output.float()                 
                return output.cpu()                              
            
            num_layers = len(hidden_states)             
            bucket_size = 8                       
            start_idx = args.selection * bucket_size             
            # print(f"Selection: {args.selection}, Start: {start_idx}")             
            end_idx = min((args.selection + 1) * bucket_size, num_layers)                          
            
            last_layer = hidden_states[-1]  # [batch_size, seq_length, hidden_dim]
            batch_size, seq_length, hidden_dim = last_layer.shape              
            
            # Length of the logits is the number of layers
            # shape of logits[i] is [batch_size, seq_length, vocab_size]
            candidate_layers_logits = torch.stack([logits[i][:, -1, :] for i in range(start_idx, end_idx)])  # [num_candidate_layers, batch_size, vocab_size]
            
            # Get last layer logits for all batches
            last_layer_logits = logits[-1][:, -1, :]  # [batch_size, vocab_size]
            
            # Expand last layer logits to compare with all candidate layers
            last_layer_logits_expanded = last_layer_logits.unsqueeze(0).expand(end_idx - start_idx, -1, -1)  
            # [num_candidate_layers, batch_size, vocab_size]
            
            last_log_probs = F.log_softmax(last_layer_logits_expanded, dim=-1)
            last_probs = F.softmax(last_layer_logits_expanded, dim=-1)
            candidate_log_probs = F.log_softmax(candidate_layers_logits, dim=-1)
            candidate_probs = F.softmax(candidate_layers_logits, dim=-1)
            
            # Calculate KL divergences with batchmean reduction
            kl_last_to_candidate = F.kl_div(
                candidate_log_probs,     # input should be log probabilities
                last_probs,              # target should be probabilities
                reduction='none'
            ).sum(-1)  # sum over vocabulary dimension
            
            kl_candidate_to_last = F.kl_div(
                last_log_probs,          # input should be log probabilities
                candidate_probs,         # target should be probabilities
                reduction='none'
            ).sum(-1)  # sum over vocabulary dimension
            
            # Normalize by vocabulary size to get values between 0 and 1
            vocab_size = last_layer_logits.size(-1)
            jsd_all = 0.5 * (kl_last_to_candidate + kl_candidate_to_last)  # [num_candidate_layers, batch_size]
            # For each batch item, find the layer with maximum JSD
            max_jsd, max_jsd_layer_indices = jsd_all.max(dim=0)
           
            # Convert indices to actual layer numbers
            max_jsd_layers = max_jsd_layer_indices + start_idx  # [batch_size]
            # print(f"Max JSD layers: {max_jsd_layers}")
            
            # Get hidden states from the layers with maximum JSD for each batch item
            max_jsd_states = torch.zeros_like(last_token_norm)  # [batch_size, hidden_dim]
            # for b in range(batch_size):
            #     selected_layer = max_jsd_layers[b]
            #     max_jsd_states[b] = F.normalize(hidden_states[selected_layer][b, -1, :], dim=-1)
            
            batch_indices = torch.arange(batch_size, device=hidden_states[0].device)  # [batch_size]

            # 将hidden_states堆叠为单个张量
            # stacked_hidden: Tensor(num_layers, batch_size, sequence_length, hidden_dim)
            stacked_hidden = torch.stack(hidden_states)

            # 使用高级索引一次性获取所有选定层的状态
            # 索引张量: (batch_size,) 用于层和批次维度
            # max_jsd_states: Tensor(batch_size, hidden_dim)
            max_jsd_states = F.normalize(
                stacked_hidden[max_jsd_layers, batch_indices, -1, :],  
                dim=-1
            )
            output = last_token_norm + args.ratio * max_jsd_states  # [batch_size, hidden_dim]
            output = F.normalize(output, dim=-1)  # [batch_size, hidden_dim]
            
            if output.dtype == torch.bfloat16:                 
                output = output.float()              
            
            return output.cpu()

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        if max_length == 500:
            sentences = [tokenizer.decode(tokenizer.encode(s, add_special_tokens=False)[:max_length]) for s in sentences]
            max_length = 512

        new_sentences = []
        for i, s in enumerate(sentences):
            if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
            s = s.replace('"', '\'')
            if len(s) > 0 and '?' == s[-1]: s = s[:-1] + '.'
        
            for prompt in task_prompts:
                new_sentences.append(prompt.replace('*sent 0*', s).strip())

        sentences = new_sentences
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
            max_length=max_length,
            truncation=max_length is not None
        )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device) if batch[k] is not None else None

        # Get raw embeddings and logits
        with torch.no_grad():
            # Forward pass to get all hidden states
            outputs = model(output_hidden_states=True, return_dict=True, **batch)
            hidden_states = outputs.hidden_states  # List[Tensor(batch_size, seq_length, hidden_dim)]
            
            # Stack all hidden states into a single tensor for batch processing
            # hidden_states: List[Tensor(batch_size, seq_length, hidden_dim)] -> 
            # all_hidden: Tensor(num_layers, batch_size, seq_length, hidden_dim)
            all_hidden = torch.stack(hidden_states)
            
            # Reshape for batch matrix multiplication
            # Combine num_layers and batch_size dimensions
            # (num_layers, batch_size, seq_length, hidden_dim) -> 
            # (num_layers * batch_size, seq_length, hidden_dim)
            num_layers, batch_size, seq_length, hidden_dim = all_hidden.shape
            reshaped_hidden = all_hidden.reshape(-1, seq_length, hidden_dim)
            
            # Single forward pass through lm_head for all layers at once
            # (num_layers * batch_size, seq_length, hidden_dim) -> 
            # (num_layers * batch_size, seq_length, vocab_size)
            all_logits = model.lm_head(reshaped_hidden)
            
            # Reshape back to separate layers
            # (num_layers * batch_size, seq_length, vocab_size) -> 
            # (num_layers, batch_size, seq_length, vocab_size)
            vocab_size = all_logits.size(-1)
            logits = all_logits.reshape(num_layers, batch_size, seq_length, vocab_size)
            
            # Convert to list of tensors if needed for compatibility
            # Each tensor will be (batch_size, seq_length, vocab_size)
            logits = [logits[i] for i in range(num_layers)]
                
            embeddings = process_hidden_states(hidden_states, logits)
        torch.cuda.empty_cache()
        return embeddings

    results = {}

    for task in tqdm(args.tasks):
        print(task)
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result
        torch.cuda.empty_cache()

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark-dev']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)
        model_name = args.model_name_or_path.split('/')[-1]
        file_path = f'./{model_name}/{args.prompt_method}'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'a') as f:
            f.write("STSBenchmark-dev\n")
            f.write(args.prompt_method + '\n')
            if args.contra:
                f.write(f' last_token_norm + {args.ratio} * max_jsd_states, Contra layer: {args.selection}/4\n')
            else:
                f.write('Contra: False\n')
            f.write(model_name + ' ' + ' '.join([str(s) for s in scores]) + '\n')
            f.write('\n')

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 
                     'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
        #
        # write results and template to file
        if args.task_set != 'transfer':
            model_name = args.model_name_or_path.split('/')[-1]
            file_path = f'./{model_name}/{args.prompt_method}_fasttest'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'a') as f:
                f.write("STS\n")
                f.write(args.prompt_method + '\n')
                if args.contra:
                    f.write(f'last_token_norm + {args.ratio} * max_jsd_states, Contra layer: {args.selection}\n')
                else:
                    f.write('Contra: False\n')
                f.write(model_name + ' ' + ' '.join([str(s) for s in scores]) + '\n')
                f.write('\n')
        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
        model_name = args.model_name_or_path.split('/')[-1]
        file_path = f'./{model_name}/{args.prompt_method}'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
     
            
if __name__ == "__main__":
    main()
