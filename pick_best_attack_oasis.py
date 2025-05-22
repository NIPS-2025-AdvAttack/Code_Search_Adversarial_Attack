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

def emb_with_oasis( # Renamed slightly again for clarity
    text_dict: Dict[str, str],
    max_tokens: int = 2048, # Reinstated from previous signature
    batch_size: int = 64,
    mode: str = 'query', # Reinstated from previous signature
    loading_path: Optional[str] = None # Reinstated - IMPORTANT: Meaning changes here!
) -> Dict[str, np.ndarray]:
    """
    Generates embeddings for a dictionary of texts using the OASIS model
    via the sentence-transformers library, maintaining signature compatibility
    with the previous transformers-based functions.

    This method uses the default pooling strategy configured for the model
    in sentence-transformers (usually mean pooling).

    Args:
        text_dict: A dictionary where keys are identifiers and values are the text strings
                   (e.g., code snippets or queries).
        max_tokens: The maximum number of tokens to use. This will attempt to set
                    the max_seq_length of the loaded SentenceTransformer model.
        batch_size: The number of texts to process in parallel during encoding.
        mode: Included for signature compatibility. Typically ignored by sentence-transformers
              for symmetric embedding models like OASIS ('query' vs 'corpus').
        loading_path: If provided, this path is used to load the SentenceTransformer model.
                      **IMPORTANT**: This must be a Hugging Face model identifier OR a path
                      to a directory containing a model saved using the
                      `SentenceTransformer.save()` method, NOT a PyTorch Lightning checkpoint path.
                      If None, uses the default 'Kwaipilot/OASIS-code-embedding-1.5B'.

    Returns:
        A dictionary mapping the original keys from text_dict to their
        corresponding **unnormalized** embeddings as NumPy arrays (identical output format).
    """
    # --- Device Setup ---
    if torch.cuda.is_available():
        device = 'cuda' # sentence-transformers typically uses string identifiers
        print("Using GPU")
    else:
        device = 'cpu'
        print("Using CPU")

    # --- Determine Model Identifier ---
    model_identifier = loading_path if loading_path is not None else "Kwaipilot/OASIS-code-embedding-1.5B"

    # --- Model Loading ---
    print('='*50)
    print(f"Loading model '{model_identifier}' using SentenceTransformer library")
    if loading_path:
        print("NOTE: Using provided 'loading_path'. Ensure it's a SentenceTransformer-compatible path/ID.")
    print('='*50)
    try:
        model = SentenceTransformer(model_name_or_path=model_identifier, device=device)

        # --- Apply max_tokens ---
        # Attempt to set the max sequence length AFTER loading the model
        original_max_tokens = model.max_seq_length
        if max_tokens != original_max_tokens:
             try:
                 model.max_seq_length = max_tokens
                 print(f"Set model max_seq_length to: {model.max_seq_length} (original was {original_max_tokens})")
             except Exception as e:
                 print(f"Warning: Could not set max_seq_length on model. Using default {original_max_tokens}. Error: {e}")
        else:
            print(f"Model max_seq_length already matches requested max_tokens: {original_max_tokens}")


    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        print("Please ensure the path/identifier is correct and points to a SentenceTransformer compatible model.")
        return {}

    # --- Data Preparation ---
    keys = list(text_dict.keys())
    texts_to_embed = list(text_dict.values())

    # --- Embedding Generation ---
    # Note: 'mode' parameter is not used by model.encode here.
    print(f"Generating embeddings with batch size {batch_size} using SentenceTransformer...")
    print(f"(Parameter 'mode={mode}' is present for compatibility but generally unused by ST encode)")

    try:
        embeddings_np = model.encode(
            texts_to_embed,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    except Exception as e:
        print(f"Error during SentenceTransformer encoding: {e}")
        return {}

    # --- Post-processing ---
    if embeddings_np is None or embeddings_np.shape[0] != len(keys):
        print(f"Warning: Embedding generation failed or produced unexpected number of results. Expected {len(keys)}, Got {embeddings_np.shape[0] if embeddings_np is not None else 'None'}")
        return {}

    # Create the result dictionary (identical format)
    results = {key: emb/np.linalg.norm(emb) for key, emb in zip(keys, embeddings_np)}

    print("Embedding generation complete.")
    return results

def find_best_attack_code_from_full_record(original_dataset, full_attack_record):

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

    query_emb = emb_with_oasis(query_dict_to_emb, max_tokens=512, batch_size=32, mode='query')
    code_emb = emb_with_oasis(code_dict_to_emb, max_tokens=512, batch_size=32, mode='corpus')
    adv_code_emb = emb_with_oasis(full_attack_record, max_tokens=512, batch_size=32, mode='corpus')

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

    args = parser.parse_args()

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
    best_attacks, unsuccessful_attacks = find_best_attack_code_from_full_record(original_dataset, full_attack_record)

    print(f"Uncessful attacks: {len(unsuccessful_attacks)}")

    # save the best attack record
    with open(args.best_attack_record, 'wb') as f:
        pickle.dump(best_attacks, f)
    print(f"Best attack record saved to {args.best_attack_record}")

if __name__ == '__main__':
    main()

    
        
