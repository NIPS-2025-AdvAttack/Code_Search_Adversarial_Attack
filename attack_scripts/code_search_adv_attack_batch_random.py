from transformers import BatchEncoding, AutoTokenizer, MPNetTokenizerFast, AutoModel, T5EncoderModel
import torch
from codet5_toks import valid_tok_idx_begin_with_space, valid_tok_idx_no_space
from parse_code_util import find_variables_and_functions_with_occurrences_python
import pdb
from tqdm import tqdm

import numpy as np
from scipy.optimize import linear_sum_assignment
from extract_cpp_locations import record_identifier_occurrences
import tempfile
from typing import Dict, Any, Union, Tuple, List # Use Any if the return type of the helper is unknown


def generate_uniform_tensor(shape: Union[Tuple[int, ...], List[int]]) -> torch.Tensor:
  """
  Generates a PyTorch tensor with the specified shape, filled with random
  values uniformly distributed between -1 and 1.

  Args:
    shape: A tuple or list defining the dimensions of the desired tensor.
           For example: (3, 4) for a 3x4 matrix, or [2, 5, 5] for a 3D tensor.

  Returns:
    A PyTorch tensor of the given shape with values in the range [-1, 1).
  """
  # torch.rand generates values in the range [0, 1)
  # Multiply by 2 to get the range [0, 2)
  # Subtract 1 to shift the range to [-1, 1)
  tensor = 2 * torch.rand(shape) - 1
  return tensor

def find_variables_and_functions_with_occurrences_cpp(code_text):
    """
    Finds C++ identifiers and their occurrences using a temporary file.

    Writes the input code string to a secure temporary file, calls a
    processing function that requires a filename, and ensures the
    temporary file is cleaned up afterwards.

    Args:
        code_text: A string containing the C++ source code.

    Returns:
        A dictionary mapping identifier names (str) to their counts (int),
        as returned by record_identifier_occurrences. Returns an empty
        dictionary if input is empty or an error occurs.
    """
    if not code_text:
        print("Input code text is empty.")
        return {}

    func_and_var_names = {}
    try:
        # Create a named temporary file that is automatically deleted
        # Use 'w' for write mode. Specify encoding for consistency.
        # delete=True (default) ensures cleanup.
        # suffix is optional but can be helpful for tools that check extensions.
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.cpp', delete=True) as temp_f:
            # Write the code text to the temporary file
            temp_f.write(code_text)

            # IMPORTANT: Flush the buffer to ensure the data is written to disk
            # before the record_identifier_occurrences function tries to read it.
            temp_f.flush()

            # Get the path/name of the temporary file
            temp_file_path = temp_f.name
            # print(f"Created temporary file: {temp_file_path}") # For debugging

            # Call the function that requires a file path
            func_and_var_names = record_identifier_occurrences(temp_file_path)

            # The file will be closed and deleted automatically when exiting the 'with' block

    except IOError as e:
        print(f"Error creating or writing to temporary file: {e}")
        return {'func_names': {}, 'variables': {}} # Return empty dict on I/O error
    except Exception as e:
        # Catch potential errors from record_identifier_occurrences
        print(f"An unexpected error occurred during processing: {e}")
        return {'func_names': {}, 'variables': {}} # Return empty dict on other errors

    return func_and_var_names
    

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

# def cosine_similarity_matrix(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#     """
#     Compute the cosine similarity between each row of tensor P and each row of tensor Q.
    
#     Args:
#         P (torch.Tensor): Tensor of shape (i, d).
#         Q (torch.Tensor): Tensor of shape (j, d).
#         eps (float): A small epsilon for numerical stability (default: 1e-8).

#     Returns:
#         torch.Tensor: Cosine similarity matrix of shape (i, j), where each entry is the cosine
#                       similarity between a row in P and a row in Q.
#     """
#     # Compute dot products between rows of P and Q.
#     dot_products = torch.mm(P, Q.t())  # shape: (i, j)
    
#     # Compute the L2 norm for each row in P and Q.
#     norm_P = torch.norm(P, p=2, dim=1)  # shape: (i,)
#     norm_Q = torch.norm(Q, p=2, dim=1)  # shape: (j,)
    
#     # Create an outer product of the norms to get denominator matrix.
#     norm_products = norm_P.unsqueeze(1) * norm_Q.unsqueeze(0)  # shape: (i, j)
    
#     # Divide dot product by the product of norms, adding eps for numerical stability.
#     cosine_sim = dot_products / (norm_products + eps)
    
#     return cosine_sim

# def grad_with_respect_to_code_init_emb_batch(query_texts, code_texts, model, tokenizer, device=torch.device('cuda:0'), max_length=512):
#     """Let the length of the input queries be M, the length of the input codes be N, max_length be C, and the internal dimension of the model be D.
#     The function returns the gradient of the model's output with respect to the input embeddings of the codes.
#     The gradient is of shape (N, M, C, D)"""

#     query_input = tokenizer(query_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length).to(device)
#     code_input = tokenizer(code_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length).to(device)

#     # query_emb[i] is the embedding for the ith query in query_texts
#     # code_emb[i] is the embedding for the ith code in code_texts

#     query_emb = model(query_input.input_ids, query_input.attention_mask)

#     code_inputs_init_embeds = model.get_input_embeddings()(code_input.input_ids)
#     code_inputs_init_embeds = code_inputs_init_embeds.clone().detach().requires_grad_(True)
#     code_emb = model(input_ids=None, attention_mask=code_input.attention_mask, inputs_embeds=code_inputs_init_embeds)

#     similarities = cosine_similarity_matrix(query_emb, code_emb)

#     # Prepare a tensor to store the gradients.
#     # The resulting tensor will have shape (i, j, *code_input_init_embeds.shape).
#     gradients = torch.zeros(similarities.shape + code_inputs_init_embeds.shape, device=device)

#     # Loop over each similarity element.
#     for p in range(similarities.shape[0]):
#         for q in range(similarities.shape[1]):
#             # Zero out any existing gradients.
#             if code_inputs_init_embeds.grad is not None:
#                 code_inputs_init_embeds.grad.zero_()
            
#             # Select a single similarity element.
#             sim_val = similarities[p, q]
            
#             # Compute the gradient of this scalar w.r.t. code_input_init_embeds.
#             sim_val.backward(retain_graph=True)
#             # Save a clone of the gradient.grad tensor.
#             gradients[p, q] = code_inputs_init_embeds.grad.clone()

#     # Detach the gradients to ensure they're not connected to the graph.
#     gradients = gradients.detach()
#     # First, take the diagonal along dimensions 1 and 2.
#     diag = gradients.diagonal(offset=0, dim1=1, dim2=2)  # shape: [3, 512, 768, 4]

#     # Now, permute the dimensions to bring the diagonal to the second position.
#     collapsed_gradients = diag.permute(0, 3, 1, 2).contiguous()  # shape: [3, 4, 512, 768]

#     # Delete variables that are no longer needed.
#     del query_input, code_input, query_emb, code_emb, similarities, gradients

#     collapsed_gradients = collapsed_gradients.cpu()
#     code_inputs_init_embeds = code_inputs_init_embeds.cpu()

#     # Optionally, clear the GPU cache.
#     # torch.cuda.empty_cache()

#     return collapsed_gradients, code_inputs_init_embeds

def token_replace_options_batch(query_texts, code_texts, model, tokenizer, device=torch.device('cuda:0'), max_length=512):
    
    influence_shape = (len(query_texts), len(code_texts), max_length, tokenizer.vocab_size)
    random_influence = generate_uniform_tensor(influence_shape).unsqueeze(-1)

    return None, random_influence


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

def find_replacements_with_influence(current_influence, valid_token_idx, positions_to_consider, return_size=20):
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

def adversarial_attack_random(query_texts, code_texts, model, tokenizer, device=torch.device('cuda:0'), max_length=512, pl='python'):
    # print("Tokenize Code Texts")
    token_influence_rank, token_influence = token_replace_options_batch(query_texts, code_texts, model, tokenizer, device, max_length)
    tokenized_codes = tokenizer(code_texts, return_offsets_mapping=True, padding='max_length', truncation=True, max_length=max_length)
    # return tokenized_codes
    replaceable_tokens = find_replaceable_tokens(code_texts, tokenized_codes, max_length, pl)

    result = {}

    for c_idx, code_text in enumerate(code_texts):
        curr_replaceable_tokens = replaceable_tokens[c_idx]
        curr_code_tokens = tokenized_codes['input_ids'][c_idx]

        token_pairs_to_replace = {'begin_toks': [], 'no_space_toks': []}
        
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
                        curr_token_texts = [curr_code_tokens[curr_occurrence[q]] for q in range(len(curr_occurrence))]
                        curr_token_strs = [tokenizer.decode(curr_token_text, skip_special_tokens=True) for curr_token_text in curr_token_texts]
                        # check if " " is the first character for every str in curr_token_strs
                        begin_with_space_count = 0
                        for curr_token_str in curr_token_strs:
                            if " " == curr_token_str[0]:
                                begin_with_space_count += 1
                        if begin_with_space_count != 0:
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

            # pdb.set_trace()


            
            result[(q_idx, c_idx)] = {'replace_dict': tok_replace_dict, 'new_code': code_text_after_replacement}
        
    return result

        

def calculate_similarity(query_text, code_text, model, tokenizer, device=torch.device('cuda:0'), max_length=512):
    with torch.no_grad():
        query_input = tokenizer([query_text], return_tensors='pt', padding='max_length', truncation=True, max_length=max_length).to(device)
        code_input = tokenizer([code_text], return_tensors='pt', padding='max_length', truncation=True, max_length=max_length).to(device)
        query_emb = model(query_input.input_ids, query_input.attention_mask)[0].squeeze(0).cpu()
        code_emb = model(code_input.input_ids, code_input.attention_mask)[0].squeeze(0).cpu()

        result = torch.dot(query_emb, code_emb)
    
    # Release GPU memory from intermediate variables
    del query_input, code_input, query_emb, code_emb
    # torch.cuda.empty_cache()

    return result