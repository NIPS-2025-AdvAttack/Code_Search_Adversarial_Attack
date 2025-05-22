import pickle
from code_search_adv_attack_batch_codet5p import calculate_similarity
from typing import Dict, List, Union, Optional
import voyageai
import numpy
from tqdm import tqdm
import numpy as np
from transformers import BatchEncoding, AutoTokenizer, MPNetTokenizerFast, AutoModel, T5EncoderModel
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)
import torch
import torch.utils.checkpoint
from torch import nn
from typing import Optional
import torch.nn.functional as F
import pdb
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import sys
from typing import Dict, List, Union, Optional
from sentence_transformers import SentenceTransformer
import argparse
import json

class TextDataset(Dataset):
    def __init__(self, text_list):
        self.text_list = text_list

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        return self.text_list[idx]


def emb_with_codet5_plus(text_dict, max_tokens=2048, batch_size=64, mode='query', loading_path=None):
    device = torch.device('cuda:0')
    print("CodeT5+ use the same encoder for both query and corpus")
    if loading_path is None:
        print('='*50)
        print("Loading the model from Hugging Face")
        print('='*50)
        model_name = "Salesforce/codet5p-110m-embedding"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    else:
        raise ValueError("Loading path is not None, please provide a valid path to load the model.")
        
    model.eval()

    dataset = TextDataset(list(text_dict.values()))
    dataloader = DataLoader(dataset, batch_size=batch_size)

    embeds = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        inputs = tokenizer(batch, padding='max_length', truncation=True, max_length=max_tokens, return_tensors="pt").to(device)
        outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask).cpu().detach()
        embeds.append(outputs)

    text_embeddings = torch.cat(embeds, dim=0)
    results = {}
    for i, k in enumerate(text_dict.keys()):
        results[k] = text_embeddings[i].numpy()

    return results

def find_best_attack_code_from_full_record(original_dataset, full_attack_record, max_iter=5):

    all_qid_cid_pairs = set()
    for key_tup in full_attack_record.keys():
        qid, cid, attack_count = key_tup
        all_qid_cid_pairs.add((qid, cid))

    full_query_dict = original_dataset['query']
    full_code_dict = original_dataset['corpus']

    query_dict_to_emb = {}
    code_dict_to_emb = {}

    for key_tup in full_attack_record.keys():
        qid, cid, attack_count = key_tup
        query_dict_to_emb[qid] = full_query_dict[qid]
        code_dict_to_emb[cid] = full_code_dict[cid]

    query_emb = emb_with_codet5_plus(query_dict_to_emb, max_tokens=512, batch_size=10, mode='query')
    code_emb = emb_with_codet5_plus(code_dict_to_emb, max_tokens=512, batch_size=10, mode='corpus')
    adv_code_emb = emb_with_codet5_plus(full_attack_record, max_tokens=512, batch_size=10, mode='corpus')

    best_attacks = {}

    for key_tup in all_qid_cid_pairs:
        qid, cid = key_tup
        query_emb_vector = query_emb[qid]
        code_emb_vector = code_emb[cid]

        similarity = query_emb_vector @ code_emb_vector.T
        similarity = similarity / (np.linalg.norm(query_emb_vector) * np.linalg.norm(code_emb_vector))

        best_attacks[(qid, cid)] = {'similarity': similarity, 'attack_count': 0, 'adv_code': full_code_dict[cid]}


    for key_tup in full_attack_record.keys():
        qid, cid, attack_count = key_tup
        if attack_count > max_iter:
            continue
        query_emb_vector = query_emb[qid]
        adv_code_emb_vector = adv_code_emb[key_tup]
        similarity = query_emb_vector @ adv_code_emb_vector.T
        similarity = similarity / (np.linalg.norm(query_emb_vector) * np.linalg.norm(adv_code_emb_vector))

        
        if similarity > best_attacks[(qid, cid)]['similarity']:
            best_attacks[(qid, cid)] = {'similarity': similarity, 'attack_count': attack_count, 'adv_code': full_attack_record[key_tup]}

    unsuccessful_attacks = []
    
    for k, v in best_attacks.items():
        if v['attack_count'] == 0:
            unsuccessful_attacks.append(k)

    return best_attacks, unsuccessful_attacks

def main():
    # use argparse to get the path to the original dataset and the full attack record, and the path to store the best attack record
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_dataset', type=str, required=True, help='Path to the original dataset')
    parser.add_argument('--full_attack_record', type=str, required=True, help='Path to the full attack record')
    parser.add_argument('--best_attack_record', type=str, required=True, help='Path to store the best attack record')
    parser.add_argument('--max_iter', type=int, default=5, help='Max number of iterations for adversarial attack')

    

    args = parser.parse_args()

    print("Arguments:")
    print("original_dataset:", args.original_dataset)
    print("full_attack_record:", args.full_attack_record)
    print("best_attack_record:", args.best_attack_record)
    print("max_iter:", args.max_iter)
    print("=====================================")

    # load the original dataset and the full attack record
    if args.original_dataset.endswith('.pickle'):
        with open(args.original_dataset, 'rb') as f:
            original_dataset = pickle.load(f)
    elif args.original_dataset.endswith('.json'):
        with open(args.original_dataset, 'r') as f:
            original_dataset = json.load(f)

    # with open(args.original_dataset, 'r') as f:
    #     original_dataset = json.load(f)
    with open(args.full_attack_record, 'rb') as f:
        full_attack_record = pickle.load(f)

    # find the best attack code from the full attack record
    best_attacks, unsuccessful_attacks = find_best_attack_code_from_full_record(original_dataset, full_attack_record, max_iter=args.max_iter)

    print(f"Uncessful attacks: {len(unsuccessful_attacks)}")

    # save the best attack record
    with open(args.best_attack_record, 'wb') as f:
        pickle.dump(best_attacks, f)
    print(f"Best attack record saved to {args.best_attack_record}")

if __name__ == '__main__':
    main()

    
        
