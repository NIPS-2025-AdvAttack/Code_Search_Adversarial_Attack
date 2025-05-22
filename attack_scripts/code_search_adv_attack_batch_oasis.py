from transformers import BatchEncoding, AutoTokenizer, MPNetTokenizerFast, AutoModel, T5EncoderModel
import torch
from oasis_toks import valid_tok_idx_begin_with_space, valid_tok_idx_no_space, valid_special_cases
from parse_code_util import find_variables_and_functions_with_occurrences_python, find_variables_and_functions_with_occurrences_js, find_variables_and_functions_with_occurrences_cpp, find_variables_and_functions_with_occurrences_go, find_variables_and_functions_with_occurrences_java
import pdb
from tqdm import tqdm

import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim # Helper for cosine similarity
from typing import List, Tuple, Optional
import traceback
import torch.nn.functional as F
import string
import tempfile
from extract_cpp_locations import record_identifier_occurrences



# def find_variables_and_functions_with_occurrences_cpp(code_text):
#     """
#     Finds C++ identifiers and their occurrences using a temporary file.

#     Writes the input code string to a secure temporary file, calls a
#     processing function that requires a filename, and ensures the
#     temporary file is cleaned up afterwards.

#     Args:
#         code_text: A string containing the C++ source code.

#     Returns:
#         A dictionary mapping identifier names (str) to their counts (int),
#         as returned by record_identifier_occurrences. Returns an empty
#         dictionary if input is empty or an error occurs.
#     """
#     if not code_text:
#         print("Input code text is empty.")
#         return {}

#     func_and_var_names = {}
#     try:
#         # Create a named temporary file that is automatically deleted
#         # Use 'w' for write mode. Specify encoding for consistency.
#         # delete=True (default) ensures cleanup.
#         # suffix is optional but can be helpful for tools that check extensions.
#         with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.cpp', delete=True) as temp_f:
#             # Write the code text to the temporary file
#             temp_f.write(code_text)

#             # IMPORTANT: Flush the buffer to ensure the data is written to disk
#             # before the record_identifier_occurrences function tries to read it.
#             temp_f.flush()

#             # Get the path/name of the temporary file
#             temp_file_path = temp_f.name
#             # print(f"Created temporary file: {temp_file_path}") # For debugging

#             # Call the function that requires a file path
#             func_and_var_names = record_identifier_occurrences(temp_file_path)

#             # The file will be closed and deleted automatically when exiting the 'with' block

#     except IOError as e:
#         print(f"Error creating or writing to temporary file: {e}")
#         return {'func_names': {}, 'variables': {}} # Return empty dict on I/O error
#     except Exception as e:
#         # Catch potential errors from record_identifier_occurrences
#         print(f"An unexpected error occurred during processing: {e}")
#         return {'func_names': {}, 'variables': {}} # Return empty dict on other errors

#     return func_and_var_names


def best_match_hungarian(dict_of_dict):
    """
    Uses the Hungarian algorithm to find the best assignment.
    
    Note: This method assumes every major_key must be assigned a candidate,
    so if a major_key does not have an option for a candidate, you should handle
    that appropriately.
    
    Args:
        dict_of_dict (dict): Dictionary mapping major_key to a dict of candidate: influence.
    
    Returns:
        result (dict): Mapping from major_key to candidate.
        total_influence (float): Total influence of the assignment.
    """
    major_keys = list(dict_of_dict.keys())
    # Get all unique candidates
    candidate_set = set()
    for cand_dict in dict_of_dict.values():
        candidate_set.update(cand_dict.keys())
    candidates = list(candidate_set)
    
    # Create a cost matrix where rows correspond to major_keys and columns to candidates.
    # Initialize with a very high cost for disallowed assignments.
    high_cost = 1e9
    cost_matrix = np.full((len(major_keys), len(candidates)), high_cost)
    
    # Find maximum influence value to transform the maximization into minimization.
    max_influence = 0
    for cand_dict in dict_of_dict.values():
        if cand_dict:
            max_influence = max(max_influence, max(cand_dict.values()))
    
    # Fill in the cost matrix. For valid assignments, cost = max_influence - influence.
    for i, major in enumerate(major_keys):
        for j, candidate in enumerate(candidates):
            if candidate in dict_of_dict[major]:
                influence = dict_of_dict[major][candidate]
                cost_matrix[i, j] = max_influence - influence
    
    # Solve the assignment problem.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    result = {}
    total_influence = 0
    for i, j in zip(row_ind, col_ind):
        # Only assign if the cost is not the high_cost placeholder.
        if cost_matrix[i, j] < high_cost:
            candidate = candidates[j]
            result[major_keys[i]] = candidate
            total_influence += dict_of_dict[major_keys[i]][candidate]
    
    return result, total_influence

# Helper for cosine similarity calculation matching SentenceTransformer's typical usage
def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    Handles normalization.
    Returns:
        Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def grad_wrt_code_init_embeddings_batch(
    query_texts: List[str],
    code_texts: List[str],
    sentence_model: SentenceTransformer, # Use the SentenceTransformer model
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    max_length: int = 512
) -> Tuple[torch.Tensor, torch.Tensor]: # Modified return type annotation
    """
    Calculates the gradient of the cosine similarity between query and code
    sentence embeddings with respect to the code's **initial token embeddings**
    (output of the 'embed_tokens' lookup layer for Qwen2/OASIS).

    Args:
        query_texts: List of query strings (length M).
        code_texts: List of code strings (length N).
        sentence_model: The loaded SentenceTransformer model (expected to wrap Qwen2).
        device: The torch device to use.
        max_length: Max sequence length for tokenizer and model.

    Returns:
        A tuple containing:
        1. gradients (torch.Tensor): Gradients with shape (M, N, C, D), where:
            M = number of queries
            N = number of codes
            C = max_length (sequence length)
            D = hidden dimension of the transformer model (embedding dim)
           Represents d(similarity[p, q]) / d(code_initial_embedding[q, :, :])
        2. code_initial_embeddings (torch.Tensor): The initial token embeddings
           for the code texts (output of embed_tokens), shape (N, C, D).
    """
    M = len(query_texts)
    N = len(code_texts)

    # --- Ensure model is on the correct device and in eval mode ---
    sentence_model.to(device)
    sentence_model.eval()

    original_max_seq_length = sentence_model.max_seq_length # Store original value

    try:
        # --- Set max sequence length ---
        # Do this early as tokenizer uses it
        sentence_model.max_seq_length = max_length
        # print(f"Set SentenceTransformer max_seq_length to: {sentence_model.max_seq_length}")

        # --- Access underlying components (Adapted for Qwen2/OASIS) ---
        tokenizer = sentence_model.tokenizer
        try:
            # Assumes standard SentenceTransformer structure:
            # model[0] is the transformer module
            # model[1] is the pooling layer
            transformer_module = sentence_model[0]
            pooling_layer = sentence_model[1]
            transformer_model = transformer_module.auto_model # The underlying Qwen2Model

            # <<< MODIFICATION: Access 'embed_tokens' directly >>>
            if hasattr(transformer_model, 'embed_tokens'):
                embedding_layer = transformer_model.embed_tokens
                embedding_dim = embedding_layer.weight.shape[-1]
                # print(f"Accessed 'embed_tokens' layer. Embedding dim: {embedding_dim}")
            else:
                # Fallback attempt if needed, though 'embed_tokens' is expected
                embedding_layer = transformer_model.get_input_embeddings()
                if embedding_layer is None:
                    raise ValueError("Could not retrieve 'embed_tokens' or input embedding layer.")
                embedding_dim = embedding_layer.embedding_dim
                print(f"Accessed embedding layer via get_input_embeddings(). Embedding dim: {embedding_dim}")
            # <<< END MODIFICATION >>>

        except (AttributeError, IndexError, KeyError, TypeError) as e:
            raise ValueError(f"Cannot access underlying model components (transformer/pooling/embeddings). Error: {e}. Model structure might be non-standard.")


        # --- Tokenize ---
        # print("Tokenizing queries and codes...")
        query_input = tokenizer(query_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length).to(device)
        code_input = tokenizer(code_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length).to(device) # Shape: N x C

        # --- Query Forward Pass (no gradients needed w.r.t. queries) ---
        # print("Performing query forward pass...")
        with torch.no_grad():
            # Manual pass for queries
            query_outputs = sentence_model( # Use the full ST model for simplicity here if pooling logic is complex
                {'input_ids': query_input['input_ids'], 'attention_mask': query_input['attention_mask']}
            )
            query_emb = query_outputs['sentence_embedding'].float() # Shape: (M, D_pooled)
            # Note: D_pooled might be different from D (embedding_dim) if pooling changes dim

        # --- Code Processing for Gradient Calculation ---
        # print("Performing code forward pass for gradient calculation...")
        with torch.enable_grad():

            # 1. Get initial code embeddings from token IDs using the correct layer
            # Shape: (N, C, D) where D is embedding dim
            code_init_embeds = embedding_layer(code_input['input_ids'])

            # <<< Store this tensor for returning later >>>
            code_initial_embeddings_to_return = code_init_embeds.detach().clone()

            # 2. *** Detach, clone, and set requires_grad=True ***
            # This makes code_init_embeds the 'leaf' tensor for gradient calculation.
            code_init_embeds_leaf = code_init_embeds.detach().clone().requires_grad_(True)

            # 3. Pass these initial embeddings through the rest of the transformer model
            # Qwen2Model forward pass handles RoPE etc., based on positions derived
            # implicitly from the sequence length of inputs_embeds.
            transformer_outputs = transformer_model(
                inputs_embeds=code_init_embeds_leaf,
                attention_mask=code_input['attention_mask'],
                return_dict=True
            )
            code_last_hidden_state = transformer_outputs.last_hidden_state

            # 4. Apply pooling (must be part of the graph)
            code_pool_input = {'token_embeddings': code_last_hidden_state, 'attention_mask': code_input['attention_mask']}
            code_pooled_output = pooling_layer(code_pool_input)
            code_emb = code_pooled_output['sentence_embedding'].float() # Shape: (N, D_pooled)

            # --- Calculate Similarities ---
            # This is the scalar value (per pair) we backpropagate from
            similarities = cos_sim(query_emb, code_emb) # Shape: (M, N)

            # --- Iterative Gradient Calculation (Refined) ---
            # print("Calculating gradients iteratively w.r.t code initial embeddings...")
            # <<< MODIFICATION: Initialize final gradient tensor >>>
            # Shape: (M, N, C, D) where C=max_length, D=embedding_dim
            C = code_init_embeds_leaf.shape[1] # max_length
            D = code_init_embeds_leaf.shape[2] # embedding_dim
            gradients_final = torch.zeros((M, N, C, D), device=device, dtype=code_init_embeds_leaf.dtype)
            # <<< END MODIFICATION >>>

            # start_time_loop = time.time() # Optional timing
            for p in range(M):
                # Slight optimization: only need to retain graph within the inner loop
                # if the graph up to `similarities` calculation doesn't change.
                # It *does* change because `code_emb` depends on `code_init_embeds_leaf`,
                # so retain_graph=True is needed on sim_val.backward().
                for q in range(N):
                    # Zero out gradients for the leaf tensor *before* each backward pass
                    if code_init_embeds_leaf.grad is not None:
                        code_init_embeds_leaf.grad.zero_()

                    # Select the specific similarity score
                    sim_val = similarities[p, q]

                    # Compute gradient of this scalar w.r.t. code_init_embeds_leaf
                    # Retain graph because code_emb (derived from code_init_embeds_leaf)
                    # is reused for the next similarity calculation involving a different query 'p'
                    # or the same query 'p' but different code 'q'.
                    # If M=1, retain_graph is needed for the loop over q.
                    # If N=1, retain_graph is needed for the loop over p.
                    # If M>1 and N>1, it's definitely needed.
                    sim_val.backward(retain_graph=True)

                    # <<< MODIFICATION: Store the relevant gradient part >>>
                    # The gradient d(sim[p,q]) / d(code_init_embeds_leaf[k,:,:]) is non-zero only for k=q.
                    if code_init_embeds_leaf.grad is not None:
                        # Extract grad w.r.t the q-th code's initial embeddings
                        grad_for_code_q = code_init_embeds_leaf.grad[q].clone()
                        gradients_final[p, q] = grad_for_code_q
                    else:
                        # Should not happen if requires_grad was set and path exists
                        print(f"Warning: Grad is None for sim[{p},{q}] w.r.t initial embeds. Skipping storage.")
                    # <<< END MODIFICATION >>>

            # print(f"Gradient loop took {time.time() - start_time_loop:.2f}s") # Optional timing

            # Detach the final results to prevent further graph tracking unless intended
            gradients_final = gradients_final.detach()
            # code_initial_embeddings_to_return is already detached

            return gradients_final, code_initial_embeddings_to_return

    finally:
        # --- Restore original max_seq_length ---
        sentence_model.max_seq_length = original_max_seq_length
        # print(f"Restored SentenceTransformer max_seq_length to: {sentence_model.max_seq_length}")

def token_replace_options_batch(query_texts, code_texts, model, device=torch.device('cuda:0'), max_length=512):
    
    code_init_emb_grad, code_original_toks_init_emb = grad_wrt_code_init_embeddings_batch(query_texts, code_texts, model, device, max_length)
    code_init_emb_grad = code_init_emb_grad

    code_original_tok_influence = torch.einsum('xyz,ixyz->ixy', code_original_toks_init_emb, code_init_emb_grad).unsqueeze(-1) # This step changes the shape of the influence_of_orignial_tokens from (N, M, C) to (N, M, C, 1)

    transformer_model = model[0].auto_model
    embedding_layer = transformer_model.get_input_embeddings()

    all_possible_embeddings = embedding_layer.weight.detach()
    token_influence = torch.einsum('ij,xyzj->xyzi', all_possible_embeddings, code_init_emb_grad) # Note this is not delta so far

    real_influence = token_influence - code_original_tok_influence

    token_influence_rank = torch.argsort(real_influence, dim=-1, descending=True)
    return token_influence_rank.cpu(), real_influence.cpu()


def find_covered_token_indices(text_offset_mapping, span, max_length=512):
    start, end = span
    token_indices = []
    for idx, offsets in enumerate(text_offset_mapping):
        # Each 'offsets' is a tensor of shape [2] (start and end positions). Convert them to ints.
        token_start = offsets[0].item() if hasattr(offsets[0], "item") else offsets[0]
        token_end = offsets[1].item() if hasattr(offsets[1], "item") else offsets[1]
        
        # Check if this token overlaps with the substring.
        # (It overlaps if its end is after the substring's start and its start is before the substring's end.)
        if token_end > start and token_start < end:
            token_indices.append(idx)
    
    return token_indices

def find_replaceable_tokens(code_texts, tokenized_codes, max_length=512, pl='python'):
    if pl == 'python':
        find_variables_and_functions_with_occurrences = find_variables_and_functions_with_occurrences_python
    elif pl == 'cpp':
        find_variables_and_functions_with_occurrences = find_variables_and_functions_with_occurrences_cpp
    elif pl == 'js':
        find_variables_and_functions_with_occurrences = find_variables_and_functions_with_occurrences_js
    elif pl == 'go':
        find_variables_and_functions_with_occurrences = find_variables_and_functions_with_occurrences_go
    elif pl == 'java':
        find_variables_and_functions_with_occurrences = find_variables_and_functions_with_occurrences_java

    func_and_var_names = []
    for idx, code_text in enumerate(code_texts):
        try:
            func_and_var = find_variables_and_functions_with_occurrences(code_text)
        except:
            func_and_var = {'func_names': {}, 'variables': {}}
        func_and_var_names.append(func_and_var)

    results = []
    text_offset_mappings = tokenized_codes['offset_mapping']

    for idx, text_offset_mapping in enumerate(text_offset_mappings):
        func_names = func_and_var_names[idx]['func_names']
        var_names = func_and_var_names[idx]['variables']
        curr_result = {'func_names': {}, 'variables': {}}
        for func_name, occurrences in func_names.items():
            curr_result['func_names'][func_name] = []
            for occurrence in occurrences:
                start_loc, end_loc = occurrence['loc']
                curr_occurence_tokens = find_covered_token_indices(text_offset_mapping, (start_loc, end_loc), max_length)
                curr_result['func_names'][func_name].append(curr_occurence_tokens)

        for var_name, occurrences in var_names.items():
            curr_result['variables'][var_name] = []
            for occurrence in occurrences:
                start_loc, end_loc = occurrence['loc']
                curr_occurence_tokens = find_covered_token_indices(text_offset_mapping, (start_loc, end_loc), max_length)
                curr_result['variables'][var_name].append(curr_occurence_tokens)

        results.append(curr_result)
    return results

def find_top_k_replacements_for_certain_positions(current_influence, valid_token_idx, positions_to_consider, return_size=20):
    # Sum influences over the given positions for all tokens.
    cum_influence = current_influence[positions_to_consider, :].sum(dim=0)

    # Create a boolean mask with True for valid tokens and False otherwise.
    mask = torch.zeros(cum_influence.size(), dtype=torch.bool, device=cum_influence.device)
    valid_token_idx_tensor = torch.tensor(valid_token_idx, dtype=torch.long, device=cum_influence.device)
    mask[valid_token_idx_tensor] = True

    # Set tokens that are not valid to -1.
    cum_influence[~mask] = -1

    # Find the top k token indices with the maximum cumulative influence.
    topk_values, topk_indices = torch.topk(cum_influence, return_size)

    # Filter out tokens with influence -1 (invalid tokens).
    valid_mask = topk_values >= 0
    valid_topk_indices = topk_indices[valid_mask]
    valid_topk_values = topk_values[valid_mask]

    # If no valid token is found, return None.
    if valid_topk_indices.numel() == 0:
        return None

    # Build and return the mapping of token indices to their cumulative influence.
    result = {int(idx): float(val) for idx, val in zip(valid_topk_indices.tolist(), valid_topk_values.tolist())}
    return result

def find_replacements_with_influence_special_case(current_influence, valid_special_cases, special_positions, normal_position, return_size=20):
    if return_size != 1:
        raise ValueError("Currently only support return_size = 1")
    

    special_case_count = 0
    special_char = ""
    for k, v in special_positions.items():
        
        if len(v) >= 1:
            special_char = k
            special_case_count += 1


    if special_case_count > 1:
        # print("Only support one special case at a time")
        return {}
    
    cum_influence_left_param = current_influence[special_positions[special_char], :].sum(dim=0)

    cum_influence_normal = current_influence[normal_position, :].sum(dim=0)
    cumulative_influence_dict = {}

    for tup in valid_special_cases[special_char]:
        curr_cumulative_influence = cum_influence_left_param[tup[0]] + cum_influence_normal[tup[1]]
        if curr_cumulative_influence > 0:
            cumulative_influence_dict[tup] = curr_cumulative_influence

    # Sort the cumulative influence dictionary by values in descending order.
    sorted_cumulative_influence = sorted(cumulative_influence_dict.items(), key=lambda x: x[1], reverse=True)
    # Take only the top `return_size` items (or as many as possible).
    top_items = sorted_cumulative_influence[:return_size]
    result = {}
    left_param_key = tuple(special_positions[special_char])
    normal_key = tuple(normal_position)
    result[left_param_key] = {}
    result[normal_key] = {}

    for kv_tup in top_items:
        result[left_param_key][kv_tup[0][0]] = kv_tup[1]
        result[normal_key][kv_tup[0][1]] = kv_tup[1]

    return result

def find_replacements_with_influence(current_influence, valid_token_idx, positions_to_consider, return_size=20):
    # Sum influences over the given positions for all tokens at once.
    cum_influence = current_influence[positions_to_consider, :].sum(dim=0)

    # Create a boolean mask for valid tokens.
    mask = torch.zeros(cum_influence.size(), dtype=torch.bool, device=cum_influence.device)
    valid_token_idx_tensor = torch.tensor(valid_token_idx, dtype=torch.long, device=cum_influence.device)
    mask[valid_token_idx_tensor] = True

    # Set tokens that are not valid to -1.
    cum_influence[~mask] = -1

    # Get indices where the cumulative influence is positive.
    positive_mask = cum_influence > 0
    positive_indices = torch.nonzero(positive_mask, as_tuple=True)[0]

    # If no valid tokens have positive influence, return an empty dictionary.
    if positive_indices.numel() == 0:
        return {}

    # Sort the positive indices by their cumulative influence in descending order.
    sorted_order = torch.argsort(cum_influence[positive_indices], descending=True)
    sorted_indices = positive_indices[sorted_order]

    # Take only the top `return_size` indices (or as many as possible).
    top_indices = sorted_indices[:return_size]

    # Create a dictionary with sorted indices as keys and their cum_influence values as values.
    result = {int(idx.item()): float(cum_influence[idx].item()) for idx in top_indices}

    return result

def adversarial_attack_batch(query_texts, code_texts, model, device=torch.device('cuda:0'), max_length=512, pl='python'):

    token_influence_rank, token_influence = token_replace_options_batch(query_texts, code_texts, model, device, max_length)
    tokenizer = model.tokenizer

    tokenized_codes = tokenizer(code_texts, return_offsets_mapping=True, padding='max_length', truncation=True, max_length=max_length)


    # return tokenized_codes
    replaceable_tokens = find_replaceable_tokens(code_texts, tokenized_codes, max_length, pl)
    # The following code is for the case where we have a fixed number of tokens to replace

    result = {}

    for c_idx, code_text in enumerate(code_texts):
        curr_replaceable_tokens = replaceable_tokens[c_idx]
        curr_code_tokens = tokenized_codes['input_ids'][c_idx]

        token_pairs_to_replace = {'begin_toks': [], 'no_space_toks': [], 'special_character_case': []}
        
        for func_name in curr_replaceable_tokens['func_names']:
            func_tok_len = set()
            for i in curr_replaceable_tokens['func_names'][func_name]:
                func_tok_len.add(len(i))
            if len(func_tok_len) > 1 or len(func_tok_len) == 0:
                if len(func_tok_len) == 0:
                    print("func_tok_len is empty")
                    print(var_name)
                    print(c_idx)
                    print(code_text)
                    print('----------------')
                # Do not replace this var_name
                continue
            else:
                func_tok_len = func_tok_len.pop()
                for i in range(func_tok_len):
                    curr_occurrence = []
                    for sub_list in curr_replaceable_tokens['func_names'][func_name]:
                        curr_occurrence.append(sub_list[i])
                    curr_occurrence = tuple(curr_occurrence)
                    if i == 0:
                        token_pairs_to_replace['begin_toks'].append(curr_occurrence)
                    else:
                        token_pairs_to_replace['no_space_toks'].append(curr_occurrence)

        for var_name in curr_replaceable_tokens['variables']:
            var_tok_len = set()
            for i in curr_replaceable_tokens['variables'][var_name]:
                var_tok_len.add(len(i))
            if len(var_tok_len) > 1 or len(var_tok_len) == 0:
                if len(var_tok_len) == 0:
                    print("var_tok_len is empty")
                    print(var_name)
                    print(c_idx)
                    print(code_text)
                    print('----------------')
                # Do not replace this var_name
                continue
            else:
                var_tok_len = var_tok_len.pop()
                for i in range(var_tok_len):
                    curr_occurrence = []
                    for sub_list in curr_replaceable_tokens['variables'][var_name]:
                        curr_occurrence.append(sub_list[i])
                    curr_occurrence = tuple(curr_occurrence)
                    if i == 0:
                        skip_flag = False
                        curr_token_texts = [curr_code_tokens[curr_occurrence[q]] for q in range(len(curr_occurrence))]
                        curr_token_strs = [tokenizer.decode(curr_token_text, skip_special_tokens=True) for curr_token_text in curr_token_texts]

                        # check if " " is the first character for every str in curr_token_strs
                        begin_with_special_tok = {}
                        begin_with_space_count = 0
                        for curr_token_idx, curr_token_str in enumerate(curr_token_strs):
                            if " " == curr_token_str[0]:
                                begin_with_space_count += 1
                            elif curr_token_str[0] in valid_special_cases.keys():
                                if curr_token_str[0] not in begin_with_special_tok:
                                    begin_with_special_tok[curr_token_str[0]] = []
                                begin_with_special_tok[curr_token_str[0]].append(curr_occurrence[curr_token_idx])
                            else:

                                if curr_token_str[0] not in string.ascii_letters:
                                    skip_flag = True
                                    break
                        if skip_flag:
                            continue
                            
                        special_tok_idx = []
                        for special_char_k, tok_pos_list in begin_with_special_tok.items():
                            for tok_pos in tok_pos_list:
                                special_tok_idx.append(tok_pos)
                        if len(begin_with_special_tok) != 0:
                            curr_info = {'special_chars': {}, 'others': []}
                            for curr_token_idx, curr_tok_position_in_code_text in enumerate(curr_occurrence):
                                if curr_tok_position_in_code_text in special_tok_idx:
                                    curr_info['special_chars'] = begin_with_special_tok
                                else:
                                    curr_info['others'].append(curr_tok_position_in_code_text)
                            token_pairs_to_replace['special_character_case'].append(curr_info)
                        elif begin_with_space_count != 0:
                            token_pairs_to_replace['begin_toks'].append(curr_occurrence)
                        else:
                            token_pairs_to_replace['no_space_toks'].append(curr_occurrence)

                    else:
                        token_pairs_to_replace['no_space_toks'].append(curr_occurrence)         
                        

        for q_idx, query_text in enumerate(query_texts):
            curr_influence = token_influence[q_idx][c_idx]            
            replace_dict = {}
            for positions_to_consider in token_pairs_to_replace['begin_toks']:
                best_tok_idx = find_replacements_with_influence(curr_influence, valid_tok_idx_begin_with_space, positions_to_consider)
                if len(best_tok_idx) > 0:
                    replace_dict[positions_to_consider] = best_tok_idx
            
            for positions_to_consider in token_pairs_to_replace['no_space_toks']:
                best_tok_idx = find_replacements_with_influence(curr_influence, valid_tok_idx_no_space, positions_to_consider)
                if len(best_tok_idx) > 0:
                    replace_dict[positions_to_consider] = best_tok_idx

            
            
            for positions_to_consider in token_pairs_to_replace['special_character_case']:
                special_char_positions = positions_to_consider['special_chars']
                normal_position = positions_to_consider['others']

                left_param_replace = find_replacements_with_influence_special_case(curr_influence, valid_special_cases, special_char_positions, normal_position, return_size=1)
                if len(left_param_replace) > 0:
                    for k, v in left_param_replace.items():
                        if k in replace_dict:
                            raise ValueError("The left param position is already in the replace_dict")
                        replace_dict[k] = v

            best_match, max_influence = best_match_hungarian(replace_dict)

            tok_replace_dict = {}
            for tup in best_match:
                for tok_id in tup:
                    tok_replace_dict[tok_id] = best_match[tup]

            code_toks_after_replacement = []
            for tok_idx, tok in enumerate(curr_code_tokens):
                if tok_idx in tok_replace_dict:
                    code_toks_after_replacement.append(tok_replace_dict[tok_idx])
                else:
                    code_toks_after_replacement.append(tok)
            
            code_text_after_replacement = tokenizer.decode(code_toks_after_replacement, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            result[(q_idx, c_idx)] = {'replace_dict': tok_replace_dict, 'new_code': code_text_after_replacement}
        
    return result



def calculate_similarity_st( # Renamed function
    query_texts: List[str],
    code_texts: List[str],
    sentence_model: SentenceTransformer,
    device: Optional[torch.device] = None,
    max_length: int = 512,
    encode_batch_size: int = 32
) -> torch.Tensor: # Return type is torch.Tensor
    """
    Calculates the cosine similarity matrix between all query texts and all code texts.

    Args:
        query_texts: A list of query strings.
        code_texts: A list of code strings. Length can be different from query_texts.
        sentence_model: The loaded SentenceTransformer model instance.
        device: The torch device to use (e.g., 'cuda:0', 'cpu'). If None,
                attempts to use GPU if available, otherwise CPU.
        max_length: Max sequence length for the model (attempts to set temporarily).
        encode_batch_size: Batch size to use for the sentence_model.encode() call.

    Returns:
        A 2D torch.Tensor of shape (len(query_texts), len(code_texts)) where
        the element at index [i, j] is the cosine similarity between
        query_texts[i] and code_texts[j].
        Returns an empty tensor (0x0 shape) if either input list is empty or an
        error occurs during processing. The tensor will be on the specified device.
    """
    # --- Input Validation ---
    if not isinstance(query_texts, list) or not isinstance(code_texts, list):
        # Returning an empty tensor might be confusing here, raising Error is clearer
        raise TypeError("query_texts and code_texts must be lists of strings.")

    if not query_texts or not code_texts: # Check if either list is empty
        print("Warning: One or both input lists are empty.")
        # Return an empty tensor with 0 rows or 0 columns, placed on the target device
        # If device isn't determined yet, determine it first.
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Shape will be (0, M) or (N, 0) or (0, 0)
        return torch.empty((len(query_texts), len(code_texts)), device=device)

    num_queries = len(query_texts)
    num_codes = len(code_texts)

    # --- Device Setup ---
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Model Preparation ---
    sentence_model.to(device)
    sentence_model.eval()

    # --- Max Sequence Length Handling ---
    original_max_seq_length = None
    if hasattr(sentence_model, 'max_seq_length'):
        original_max_seq_length = sentence_model.max_seq_length
        try:
            # Set the max_seq_length attribute which model.encode() should respect
            sentence_model.max_seq_length = max_length
        except Exception as e:
             print(f"Warning: Could not set max_seq_length on model. Using original {original_max_seq_length}. Error: {e}")
    else:
        print(f"Warning: SentenceTransformer model object does not have a 'max_seq_length' attribute.")


    # Initialize return tensor as empty (will be replaced in try block)
    # Placing it on the target device early avoids potential issues later
    similarity_matrix = torch.empty((len(query_texts), len(code_texts)), device=device)

    try:
        # --- Prepare all texts for batch encoding ---
        all_texts = query_texts + code_texts

        # --- Generate Embeddings using model.encode() ---
        with torch.no_grad():
            # print(f"Encoding {len(all_texts)} texts...") # Optional debug
            embeddings = sentence_model.encode(
                all_texts,
                batch_size=encode_batch_size,
                convert_to_tensor=True,
                device=device
                # No need to normalize here, util.cos_sim handles it.
                # show_progress_bar=True # Optional
            )
            # embeddings should be a tensor shape (num_queries + num_codes, embedding_dim)

        # --- Check Embedding Shape ---
        expected_shape_dim0 = num_queries + num_codes
        if embeddings is None or embeddings.shape[0] != expected_shape_dim0:
             raise ValueError(f"model.encode() did not return the expected number of embeddings. Expected {expected_shape_dim0}, Got: {embeddings.shape[0] if embeddings is not None else 'None'}")

        # --- Split Embeddings ---
        query_embs = embeddings[0:num_queries]     # Shape: (num_queries, embedding_dim)
        code_embs = embeddings[num_queries : expected_shape_dim0] # Shape: (num_codes, embedding_dim)

        # --- Calculate Full Similarity Matrix using sentence_transformers.util ---
        # This efficiently computes cos(a, b) for all pairs a in query_embs, b in code_embs
        # print("Calculating similarity matrix...") # Optional debug
        similarity_matrix = util.cos_sim(query_embs, code_embs)
        # Result shape: (num_queries, num_codes)


    except Exception as e:
        print(f"An error occurred during similarity matrix calculation: {e}")
        traceback.print_exc()
        # Return an empty tensor matching the expected output dimensions but size 0 along one axis
        # This signals failure while maintaining type and device consistency.
        similarity_matrix = torch.empty((len(query_texts), len(code_texts)), device=device)


    finally:
        # --- Restore original max_seq_length ---
        if original_max_seq_length is not None and hasattr(sentence_model, 'max_seq_length'):
            try:
                 sentence_model.max_seq_length = original_max_seq_length
            except Exception as e:
                 print(f"Warning: Could not restore original max_seq_length. Error: {e}")

    return similarity_matrix


