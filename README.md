# Adversarial Attacks on Neural Code Search

**This repository contains the code and data for the research paper investigating adversarial attacks against neural code language models (CLMs) used in code search tools.**

Reliable code retrieval is crucial for developer productivity and effective code reuse, significantly impacting software engineering teams and organizations. However, the neural code language models (CLMs) powering current search tools are susceptible to adversarial attacks targeting non-functional textual elements. In this paper, we introduce a language-agnostic adversarial attack method that exploits this vulnerability of CLMs. Our approach perturbs identifiers within a code snippet without altering its functionality to deceptively align the code with a target query. In particular, we demonstrate that modifications based on smaller models, such as CodeT5+, are highly transferable to larger or closed-source models, like Nomic-emb-code or Voyage-code-3. These modifications can increase the similarity between the query and an arbitrary irrelevant code snippet, consequently degrading the retrieval performance of state-of-the-art models. The experimental results highlight the fragility of current code search methods and underscore the need for more robust, semantic-aware approaches.

## Datasets for the Attack

In our paper, we used the adversarial attacks on three datasets:

* **[CosQA](https://huggingface.co/datasets/CoIR-Retrieval/cosqa):** We sampled 100 queries and 100 code snippets.
* **[CLARC](https://huggingface.co/datasets/ClarcTeam/CLARC):** We sampled 100 queries and 100 code snippets.
* **[HumanEval-X](https://huggingface.co/datasets/THUDM/humaneval-x):** We utilized the entire dataset for Python, C++, Java, JavaScript, and Go.

The sampled queries and code snippets from CosQA and CLARC, along with the HumanEval-X data used, can be found in the `data/` directory.

## Running the Adversarial Attack

The adversarial attack can be executed using one of the following scripts, depending on the base model for generating perturbations:

* `run_adversarial_attack_in_batch_codet5p.py`
* `run_adversarial_attack_in_batch_oasis.py`
* `run_adversarial_attack_in_batch_random.py` (for a random perturbation baseline)

To execute an attack, use the following command structure:

```bash
python3 run_adversarial_attack_in_batch_<model_name>.py \
    --query_path <path_to_attacking_queries> \
    --code_path <path_to_attacking_codes> \
    --output_path <path_to_store_the_results> \
    --max_iter <num_of_iterations_for_attack> \
    --pl <programming_language_to_use> \
    --batch_size <batch_size>
```

**Argument Explanations:**

* `model_name`: Specifies the model used for the attack. Choose from `codet5p`, `oasis`, or `random`.
* `path_to_attacking_queries`: Path to the file containing the queries.
* `path_to_attacking_codes`: Path to the file containing the code snippets.
* `output_path`: Path where the attack results will be saved.
* `max_iter`: The maximum number of iterations for the attack algorithm.
* `pl`: The programming language of the code snippets. Currently supported languages are `python`, `cpp`, `java`, `js`, and `go`.
* `batch_size`: The number of query-code pairs to process in each batch.

This command will run the adversarial attack on all combinations of (query, code) pairs from the specified input files. The results, including the adversarial code generated at each iteration, will be saved as a Python dictionary in a pickle file at the `output_path`. The dictionary keys in this file are tuples of `(query_id, code_id, iter_count)`.

**Example Command:**

```bash
python3 run_adversarial_attack_in_batch_codet5p.py \
    --query_path data/cosqa_query.json \
    --code_path data/cosqa_code.json \
    --output_path ./attack_record_for_cosqa_based_on_codet5p.pickle \
    --max_iter 5 \
    --pl python \
    --batch_size 20
```

## Extending to Other Models

Our current implementation directly supports attacks based on two pretrained models from Hugging Face: **CodeT5+** and **OASIS**.

To adapt the attack methodology for other models, please:
1.  Refer to the example implementations in the `attack_scripts/` directory.
2.  Modify the code to:
    * Update the set of valid tokens for the new model.
    * Ensure correct gradient calculation and influence scoring according to the target model's architecture.

## Extending to Other Programming Languages

The current implementation supports five programming languages: Python, C++, Java, JavaScript, and Go.

To add support for other programming languages, please:
1.  Examine the language parsing examples provided in the `parse_code/` directory.
2.  Implement the necessary parsing logic and add the corresponding files to enable support for the new language. This typically involves defining how to extract and replace identifiers for the target language.
