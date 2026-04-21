import os
import json
import csv # Added for CSV output
import time
import argparse
import pandas as pd
from irac_core import get_baseline_cot_and_answer, generate_hypotheses, probe_hypothesis
from hf_utils import DEFAULT_MODEL_HF # Changed import
from datasets import load_dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Command Line Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description="Run IRAC experiments on reasoning datasets")
    parser.add_argument("--subset", type=int, default=None, 
                        help="Test on first N questions only (for quick validation)")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix to add to output directory (e.g., '_v2' for experiment_results_truthful_qa_v2)")
    return parser.parse_args()

# --- Configuration ---
MODEL_NAME = DEFAULT_MODEL_HF  # Or your specific Hugging Face model
TEMPERATURE_BASELINE = 0.0
TEMPERATURE_HYPOTHESIS_GEN = 0.0 # Higher for more diverse hypotheses
TEMPERATURE_PROBE = 0.0
MAX_NEW_TOKENS = 10_000
NUM_HYPOTHESES_TO_PROBE_PER_QUESTION = 3 # How many of the generated hypotheses to test

# Dataset Configuration
# DATASET_NAME = "gsm8k"  # Options: "gsm8k", "truthful_qa"
# DATASET_CONFIG = "main"  # "main" for GSM8K, "generation" for TruthfulQA
# DATASET_SPLIT = "test"  # "test" for GSM8K, "train" for TruthfulQA (used as test set)
# MAX_QUESTIONS_TO_PROCESS = None # Set to None to process all questions in the split
DATASET_NAME = "strategy_qa"  # Options: "gsm8k", "truthful_qa", "strategy_qa"
DATASET_CONFIG = None  # "main" for GSM8K, "generation" for TruthfulQA, None for StrategyQA
DATASET_SPLIT = "test"  # "test" for GSM8K/StrategyQA, "validation" for TruthfulQA
MAX_QUESTIONS_TO_PROCESS = None # Set to None to process all questions in the split


# Output directory - dataset-specific
OUTPUT_DIR = f"/home/ec2-user/code/personal/logic_cot/experiment_results_{DATASET_NAME}"

GENERATE_HYPOTHESES_DYNAMICALLY = True # Must be True for large datasets

# --- Resume Capability ---

def get_processed_question_ids(csv_path):
    """
    Returns a set of question IDs that have already been processed.
    Enables resuming experiments from where they stopped.
    """
    if not os.path.exists(csv_path):
        return set()
    
    try:
        df = pd.read_csv(csv_path)
        if 'question_id' in df.columns:
            processed_ids = set(df['question_id'].tolist())
            print(f"Found {len(processed_ids)} already processed questions in {csv_path}")
            return processed_ids
    except Exception as e:
        print(f"Warning: Could not read existing CSV for resume: {e}")
        print("Starting fresh experiment.")
    
    return set()

# --- Dataset ---

def print_run_experiment_config():
    """Prints the configuration parameters for the experiment run."""
    print("\n--- Experiment Configuration ---")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"TEMPERATURE_BASELINE: {TEMPERATURE_BASELINE}")
    print(f"TEMPERATURE_HYPOTHESIS_GEN: {TEMPERATURE_HYPOTHESIS_GEN}")
    print(f"TEMPERATURE_PROBE: {TEMPERATURE_PROBE}")
    print(f"MAX_NEW_TOKENS: {MAX_NEW_TOKENS}")
    print(f"NUM_HYPOTHESES_TO_PROBE_PER_QUESTION: {NUM_HYPOTHESES_TO_PROBE_PER_QUESTION}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"DATASET_NAME: {DATASET_NAME} ({DATASET_CONFIG} - {DATASET_SPLIT})")
    print(f"MAX_QUESTIONS_TO_PROCESS: {MAX_QUESTIONS_TO_PROCESS if MAX_QUESTIONS_TO_PROCESS is not None else 'All'}")
    print(f"GENERATE_HYPOTHESES_DYNAMICALLY: {GENERATE_HYPOTHESES_DYNAMICALLY}")
    print("--------------------------------\n")

def load_questions_from_hf(dataset_name, config_name, split_name, max_questions=None):
    """Loads questions from a Hugging Face dataset (GSM8K)."""
    print(f"Loading dataset {dataset_name} ({config_name}) split {split_name} from Hugging Face...")
    dataset = load_dataset(dataset_name, config_name, split=split_name, cache_dir="/home/ec2-user/code/personal/hf_cache")
    questions_to_process = []
    
    count = 0
    for i, item in enumerate(dataset):
        if max_questions is not None and count >= max_questions:
            break
        # GSM8K has 'question' and 'answer' fields.
        # We'll use the 'question' field. The 'answer' field contains the full reasoning and final answer.
        # For this experiment, we are generating answers, not directly using the dataset's answer for prompting.
        questions_to_process.append({
            "id": f"{dataset_name}_{config_name}_{split_name}_{i}",
            "text": item["question"],
            "ground_truth_answer_details": item["answer"], # Store for potential later evaluation
            "few_shot_examples_for_hypothesis_gen": "Consider general mathematical reasoning principles and common student errors.", # Generic placeholder
            "dataset_type": "math",
            # Manual hypotheses are not practical for a large dataset
        })
        count += 1
    print(f"Loaded {len(questions_to_process)} questions.")
    return questions_to_process

def load_strategyqa_questions(split_name, max_questions=None):
    """Loads questions from StrategyQA dataset (ChilleD/StrategyQA)."""
    dataset_name = "ChilleD/StrategyQA"
    print(f"Loading StrategyQA dataset split {split_name} from Hugging Face...")
    dataset = load_dataset(dataset_name, split=split_name, cache_dir="/home/ec2-user/code/personal/hf_cache")
    questions_to_process = []
    
    count = 0
    for i, item in enumerate(dataset):
        if max_questions is not None and count >= max_questions:
            break
        
        # StrategyQA fields: qid, term, description, question, answer (bool), facts
        qid = item.get("qid", f"strategy_qa_{split_name}_{i}")
        term = item.get("term", "")
        description = item.get("description", "")
        answer_bool = item.get("answer", None)  # Boolean: True/False
        facts = item.get("facts", "")
        
        # Normalize boolean answer to string
        if answer_bool is True:
            ground_truth_answer = "true"
        elif answer_bool is False:
            ground_truth_answer = "false"
        else:
            ground_truth_answer = str(answer_bool).lower() if answer_bool is not None else "unknown"
        
        # Generate hypothesis context based on the multi-hop nature
        hypothesis_context = (
            f"This is a multi-hop reasoning question about '{term}'. "
            f"Consider implicit intermediate reasoning steps, unstated fact connections, "
            f"and background knowledge required to chain facts together."
        )
        
        questions_to_process.append({
            "id": f"strategy_qa_{split_name}_{qid}",
            "text": item["question"],
            "ground_truth_answer_details": {
                "answer": ground_truth_answer,
                "answer_bool": answer_bool,
                "facts": facts,
                "term": term,
                "description": description,
            },
            "category": term if term else "general",  # Use 'term' as category for analysis
            "type": "multi_hop",
            "few_shot_examples_for_hypothesis_gen": hypothesis_context,
            "dataset_type": "multi_hop_reasoning",
        })
        count += 1
    
    print(f"Loaded {len(questions_to_process)} StrategyQA questions.")
    unique_terms = set(q['category'] for q in questions_to_process if q['category'] != 'general')
    print(f"Unique terms/topics: {len(unique_terms)}")
    return questions_to_process

def load_truthfulqa_questions(config_name, split_name, max_questions=None):
    """Loads questions from TruthfulQA dataset."""
    dataset_name = "truthful_qa"
    print(f"Loading TruthfulQA dataset ({config_name}) split {split_name} from Hugging Face...")
    dataset = load_dataset(dataset_name, config_name, split=split_name, cache_dir="/home/ec2-user/code/personal/hf_cache")
    questions_to_process = []
    
    count = 0
    for i, item in enumerate(dataset):
        if max_questions is not None and count >= max_questions:
            break
        
        # TruthfulQA fields: question, best_answer, correct_answers, incorrect_answers, category, type
        category = item.get("category", "general")
        question_type = item.get("type", "unknown")
        
        # Generate category-specific hypothesis context
        hypothesis_context = f"Consider common misconceptions about {category}, false beliefs, imitative falsehoods, and factors that lead people to answer incorrectly due to popular but incorrect information."
        
        questions_to_process.append({
            "id": f"{dataset_name}_{config_name}_{split_name}_{i}",
            "text": item["question"],
            "ground_truth_answer_details": {
                "best_answer": item.get("best_answer", ""),
                "correct_answers": item.get("correct_answers", ""),
                "incorrect_answers": item.get("incorrect_answers", ""),
            },
            "category": category,
            "type": question_type,
            "few_shot_examples_for_hypothesis_gen": hypothesis_context,
            "dataset_type": "truthfulness",
        })
        count += 1
    
    print(f"Loaded {len(questions_to_process)} TruthfulQA questions.")
    print(f"Categories present: {len(set(q['category'] for q in questions_to_process))}")
    return questions_to_process

# --- Main Experiment Loop ---
def run_full_experiment(args=None):
    """
    Main experiment loop.
    
    Args:
        args: Parsed command line arguments (optional, will parse if not provided)
    """
    if args is None:
        args = parse_args()
    
    # Apply output suffix if provided
    output_dir = OUTPUT_DIR
    if args.output_suffix:
        output_dir = f"{OUTPUT_DIR}{args.output_suffix}"
    
    print_run_experiment_config() # Print config at the start
    
    # Print additional CLI args if specified
    if args.subset:
        print(f"[CLI] --subset: Testing on first {args.subset} questions only")
    if args.output_suffix:
        print(f"[CLI] --output-suffix: '{args.output_suffix}' -> {output_dir}")
    print()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_results_list_for_json_dump = [] # Keep for the _all_experiment_logs.json (optional full dump)
    output_csv_file = os.path.join(output_dir, "all_experiment_results.csv")

    # Load questions based on dataset type
    if DATASET_NAME.lower() == "truthful_qa":
        all_questions = load_truthfulqa_questions(
            DATASET_CONFIG, DATASET_SPLIT, MAX_QUESTIONS_TO_PROCESS
        )
    elif DATASET_NAME.lower() == "strategy_qa":
        all_questions = load_strategyqa_questions(
            DATASET_SPLIT, MAX_QUESTIONS_TO_PROCESS
        )
    else:  # Default to GSM8K or other HF datasets
        all_questions = load_questions_from_hf(
            DATASET_NAME, DATASET_CONFIG, DATASET_SPLIT, MAX_QUESTIONS_TO_PROCESS
        )

    # Resume capability - check for already processed questions
    processed_question_ids = get_processed_question_ids(output_csv_file)
    
    # Filter out already processed questions
    questions_to_test = [q for q in all_questions if q['id'] not in processed_question_ids]
    
    # Apply --subset limit if specified
    if args.subset and len(questions_to_test) > args.subset:
        print(f"[CLI] Limiting to first {args.subset} questions (from {len(questions_to_test)} remaining)")
        questions_to_test = questions_to_test[:args.subset]
    
    # Report resume status
    if processed_question_ids:
        print(f"\n{'='*60}")
        print("RESUME MODE ACTIVATED")
        print(f"{'='*60}")
        print(f"Total questions in dataset: {len(all_questions)}")
        print(f"Already processed: {len(processed_question_ids)}")
        print(f"Remaining to process: {len(questions_to_test)}")
        print(f"Progress: {len(processed_question_ids)}/{len(all_questions)} ({len(processed_question_ids)/len(all_questions)*100:.1f}%)")
        print(f"{'='*60}\n")
    else:
        print(f"Starting fresh experiment with {len(questions_to_test)} questions.\n")
    
    if not questions_to_test:
        print("All questions have been processed! Experiment complete.")
        return

    # Check if CSV file exists to write header only once
    csv_file_exists = os.path.isfile(output_csv_file)

    with open(output_csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question_id', 'experiment_data_json']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not csv_file_exists:
            csv_writer.writeheader()

        for i in tqdm(range(len(questions_to_test)), desc="Processing Questions"):
            question_data = questions_to_test[i]
            question_id = question_data["id"]
            question_text = question_data["text"]
            few_shot_text_for_hypo = question_data.get("few_shot_examples_for_hypothesis_gen", "")
            ground_truth_answer_details = question_data.get("ground_truth_answer_details", "")

            # print(f"\n--- Processing Question {i+1}/{len(questions_to_test)}: {question_id} ---")

            experiment_log = {
                "question_id": question_id,
                "question_text": question_text,
                "dataset_type": question_data.get("dataset_type", "unknown"),
                "category": question_data.get("category", "N/A"),
                "type": question_data.get("type", "N/A")
            }

            # Stage 1: Baseline CoT
            # print("  Getting baseline CoT and answer...")
            c_base, a_base, raw_baseline = get_baseline_cot_and_answer(
                question_text, MODEL_NAME, TEMPERATURE_BASELINE, MAX_NEW_TOKENS
            )
            experiment_log["baseline_cot"] = c_base
            experiment_log["baseline_answer"] = a_base
            experiment_log["raw_baseline_output"] = raw_baseline
            experiment_log["ground_truth_answer_details"] = ground_truth_answer_details
            # time.sleep(0.5) # Small delay

            # Stage 2.1: Hypothesis Generation
            hypotheses_to_probe = []
            if GENERATE_HYPOTHESES_DYNAMICALLY:
                # print("  Generating implicit hypotheses...")
                # Determine dataset type for hypothesis generation prompt selection
                dataset_type_map = {
                    "truthful_qa": "truthful_qa",
                    "strategy_qa": "strategy_qa",
                }
                dataset_type_for_hypo = dataset_type_map.get(DATASET_NAME.lower(), "gsm8k")
                
                generated_hypotheses, raw_hypo_gen = generate_hypotheses(
                    question_text, few_shot_text_for_hypo, MODEL_NAME, TEMPERATURE_HYPOTHESIS_GEN, MAX_NEW_TOKENS,
                    dataset_type=dataset_type_for_hypo
                )
                experiment_log["generated_hypotheses"] = generated_hypotheses
                experiment_log["raw_hypothesis_generation_output"] = raw_hypo_gen
                if generated_hypotheses:
                    hypotheses_to_probe = generated_hypotheses[:NUM_HYPOTHESES_TO_PROBE_PER_QUESTION]
                # time.sleep(0.5)
            else: # This branch is unlikely to be used if GENERATE_HYPOTHESES_DYNAMICALLY is True
                # print("  Using manually curated hypotheses...")
                hypotheses_to_probe = question_data.get("manual_hypotheses", [])[:NUM_HYPOTHESES_TO_PROBE_PER_QUESTION]
                experiment_log["used_manual_hypotheses"] = hypotheses_to_probe

            # Stage 2.2: Implicit Reliance Probing
            experiment_log["probes"] = []
            if not hypotheses_to_probe:
                print("  No hypotheses to probe.")
            for hypo_idx, hypothesis in enumerate(tqdm(hypotheses_to_probe, desc=f"Probing Hypotheses for {question_id}", leave=False)):
                # print(f"    Probing hypothesis {hypo_idx+1}/{len(hypotheses_to_probe)}: '{hypothesis[:50]}...'")
                probe_result = {"hypothesis": hypothesis}

                art_use, cot_use, ans_use, raw_use = probe_hypothesis(question_text, hypothesis, "use", MODEL_NAME, TEMPERATURE_PROBE, MAX_NEW_TOKENS)
                probe_result["articulation_use"], probe_result["cot_use"], probe_result["answer_use"], probe_result["raw_use_output"] = art_use, cot_use, ans_use, raw_use
                # time.sleep(0.5)

                art_ign, cot_ign, ans_ign, raw_ign = probe_hypothesis(question_text, hypothesis, "ignore", MODEL_NAME, TEMPERATURE_PROBE, MAX_NEW_TOKENS)
                probe_result["articulation_ignore"], probe_result["cot_ignore"], probe_result["answer_ignore"], probe_result["raw_ignore_output"] = art_ign, cot_ign, ans_ign, raw_ign
                # time.sleep(0.5)

                experiment_log["probes"].append(probe_result)

            # Save result for this question to CSV
            try:
                experiment_log_json_string = json.dumps(experiment_log)
                csv_writer.writerow({'question_id': question_id, 'experiment_data_json': experiment_log_json_string})
                # print(f"  Results for {question_id} appended to {output_csv_file}") # tqdm provides progress
            except Exception as e:
                print(f"ERROR: Could not serialize or write data for {question_id} to CSV: {e}")
            
            all_results_list_for_json_dump.append(experiment_log)

    # Save all results to a single JSON file too (optional, but was there before)
    consolidated_json_path = os.path.join(output_dir, "_all_experiment_logs.json")
    with open(consolidated_json_path, 'w') as f_json_dump:
        json.dump(all_results_list_for_json_dump, f_json_dump, indent=4)
    print(f"\nConsolidated JSON results saved to {consolidated_json_path}")
    print(f"All experiment results appended to CSV: {output_csv_file}")
    print("\n--- Experiment Run Complete ---")

if __name__ == "__main__":
    run_full_experiment()
