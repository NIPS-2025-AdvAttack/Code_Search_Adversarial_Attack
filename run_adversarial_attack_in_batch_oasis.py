from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import sys
import argparse
import pickle
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo
import wandb
import subprocess
import time
import threading
from sentence_transformers import SentenceTransformer


# Append project-specific paths
from attack_scripts.code_search_adv_attack_batch_oasis import adversarial_attack_batch

def get_current_pacific_time():
    pacific_time = datetime.now(ZoneInfo("America/Los_Angeles"))
    return pacific_time.strftime("%m-%d-%H-%M")


def gpu_logging(stop_event):
    while not stop_event.is_set():
        try:
            # Get GPU stats from nvidia-smi
            result = subprocess.check_output(
                "nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits",
                shell=True
            )
            usage_str, power_str, mem_used_str, mem_total_str = result.decode("utf-8").strip().split(', ')
            wandb.log({
                "GPU Utilization (%)": float(usage_str),
                "GPU Power (Watts)": float(power_str),
                "GPU Memory Used (MB)": float(mem_used_str),
                "GPU Total Memory (MB)": float(mem_total_str)
            })
        except Exception as e:
            print("Error retrieving GPU stats:", e)
        time.sleep(1)

if __name__ == "__main__":
    # Initialize wandb for GPU monitoring and run configuration
    wandb.init(project="gpu_monitoring_when_running_adversarial_attack")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path", type=str, required=True, help="Path to the JSON file containing the queries")
    parser.add_argument("--code_path", type=str, required=True, help="Path to the JSON file containing the codes")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output pickle file")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for adversarial attack")
    parser.add_argument("--max_iter", type=int, default=5, help="Max number of iterations for adversarial attack")
    parser.add_argument("--max_length", type=int, default=512, help="Max length of the code")
    parser.add_argument("--pl", type=str, default="python", help="Programming language of the code")
    args = parser.parse_args()

    # Update wandb configuration with run parameters
    wandb.config.update({
        "query_path": args.query_path,
        "code_path": args.code_path,
        "output_path": args.output_path,
        "batch_size": args.batch_size,
        "max_iter": args.max_iter,
        "max_length": args.max_length,
        "pl": args.pl
    })

    # Start the GPU logging thread
    stop_event = threading.Event()
    gpu_thread = threading.Thread(target=gpu_logging, args=(stop_event,))
    gpu_thread.start()

    # Load queries and codes
    with open(args.query_path, "r") as f:
        queries = json.load(f)
    with open(args.code_path, "r") as f:
        codes = json.load(f)

    test_qids = list(queries.keys())
    test_cids = list(codes.keys())
    dataset = {"query": queries, "corpus": codes}
    recorded_codes = {}
    for cid in test_cids:
        for qid in test_qids:
            recorded_codes[(qid, cid)] = dataset['corpus'][cid]

    attack_iter = args.max_iter
    batch_size = args.batch_size

    device = torch.device('cuda:0')
    model_name = 'Kwaipilot/OASIS-code-embedding-1.5B'

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True).to(device)

    full_attack_record = {}
    total_iterations = attack_iter * len(test_qids) * math.ceil(len(test_cids) / batch_size)
    pbar = tqdm(total=total_iterations)

    

    for attack_count in range(attack_iter):
        for i in range(len(test_qids)):
            query_texts = [dataset['query'][test_qids[i]]]
            for j in range(0, len(test_cids), batch_size):
                code_texts = [recorded_codes[(test_qids[i], cid)] for cid in test_cids[j: j+batch_size]]
                attack_records = adversarial_attack_batch(
                    query_texts,
                    code_texts,
                    model,
                    device=device,
                    max_length=args.max_length,
                    pl=args.pl
                )
                for keys in attack_records:
                    curr_qid = test_qids[i]
                    curr_cid = test_cids[j + keys[1]]
                    full_attack_record[(curr_qid, curr_cid, attack_count + 1)] = attack_records[keys]['new_code']
                    recorded_codes[(curr_qid, curr_cid)] = attack_records[keys]['new_code']
                pbar.update(1)
        # Save intermediate results after each iteration
        with open(f"./test_files/recoreded_codes_after_iter{attack_count}_{get_current_pacific_time()}.pickle", 'wb') as f:
            pickle.dump(recorded_codes, f)
    pbar.close()

    with open(args.output_path, 'wb') as f:
        pickle.dump(full_attack_record, f)
    print("Done!")

    # Signal the GPU logging thread to stop and wait for it to finish
    stop_event.set()
    gpu_thread.join()

    # Finish the wandb run
    wandb.finish()