import os
import json
import csv # Added for CSV output
import time
from irac_core import get_baseline_cot_and_answer, generate_hypotheses, probe_hypothesis
from hf_utils import DEFAULT_MODEL_HF # Changed import
from datasets import load_dataset

# --- Configuration ---
MODEL_NAME = DEFAULT_MODEL_HF  # Or your specific Hugging Face model
TEMPERATURE_BASELINE = 0.0
TEMPERATURE_HYPOTHESIS_GEN = 0.7 # Higher for more diverse hypotheses
TEMPERATURE_PROBE = 0.0
NUM_HYPOTHESES_TO_PROBE_PER_QUESTION = 3 # How many of the generated hypotheses to test
OUTPUT_DIR = "experiment_results"
DATASET_NAME = "gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test" # or "train"
MAX_QUESTIONS_TO_PROCESS = None # Set to None to process all questions in the split

# --- Dataset ---

def load_questions_from_hf(dataset_name, config_name, split_name, max_questions=None):
    """Loads questions from a Hugging Face dataset."""
    print(f"Loading dataset {dataset_name} ({config_name}) split {split_name} from Hugging Face...")
    dataset = load_dataset(dataset_name, config_name, split=split_name)
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
            # Manual hypotheses are not practical for a large dataset
        })
        count += 1
    print(f"Loaded {len(questions_to_process)} questions.")
    return questions_to_process

GENERATE_HYPOTHESES_DYNAMICALLY = True # Must be True for large datasets

# --- Main Experiment Loop ---
def run_full_experiment():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_results_list_for_json_dump = [] # Keep for the _all_experiment_logs.json (optional full dump)
    output_csv_file = os.path.join(OUTPUT_DIR, "all_experiment_results.csv")

    questions_to_test = load_questions_from_hf(
        DATASET_NAME, DATASET_CONFIG, DATASET_SPLIT, MAX_QUESTIONS_TO_PROCESS
    )

    # Check if CSV file exists to write header only once
    csv_file_exists = os.path.isfile(output_csv_file)

    with open(output_csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question_id', 'experiment_data_json']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not csv_file_exists:
            csv_writer.writeheader()

        for i, question_data in enumerate(questions_to_test):
            question_id = question_data["id"]
            question_text = question_data["text"]
            few_shot_text_for_hypo = question_data.get("few_shot_examples_for_hypothesis_gen", "")
            ground_truth_answer_details = question_data.get("ground_truth_answer_details", "")

            print(f"\n--- Processing Question {i+1}/{len(questions_to_test)}: {question_id} ---")

            experiment_log = {"question_id": question_id, "question_text": question_text}

            # Stage 1: Baseline CoT
            print("  Getting baseline CoT and answer...")
            c_base, a_base, raw_baseline = get_baseline_cot_and_answer(
                question_text, MODEL_NAME, TEMPERATURE_BASELINE
            )
            experiment_log["baseline_cot"] = c_base
            experiment_log["baseline_answer"] = a_base
            experiment_log["raw_baseline_output"] = raw_baseline
            experiment_log["ground_truth_answer_details"] = ground_truth_answer_details
            time.sleep(0.5) # Small delay

            # Stage 2.1: Hypothesis Generation
            hypotheses_to_probe = []
            if GENERATE_HYPOTHESES_DYNAMICALLY:
                print("  Generating implicit hypotheses...")
                generated_hypotheses, raw_hypo_gen = Error in hypothesis generation: JSON decode error(
                    question_text, few_shot_text_for_hypo, MODEL_NAME, TEMPERATURE_HYPOTHESIS_GEN
                )
                experiment_log["generated_hypotheses"] = generated_hypotheses
                experiment_log["raw_hypothesis_generation_output"] = raw_hypo_gen
                if generated_hypotheses:
                    hypotheses_to_probe = generated_hypotheses[:NUM_HYPOTHESES_TO_PROBE_PER_QUESTION]
                time.sleep(0.5)
            else: # This branch is unlikely to be used if GENERATE_HYPOTHESES_DYNAMICALLY is True
                print("  Using manually curated hypotheses...")
                hypotheses_to_probe = question_data.get("manual_hypotheses", [])[:NUM_HYPOTHESES_TO_PROBE_PER_QUESTION]
                experiment_log["used_manual_hypotheses"] = hypotheses_to_probe

            # Stage 2.2: Implicit Reliance Probing
            experiment_log["probes"] = []
            if not hypotheses_to_probe:
                print("  No hypotheses to probe.")
            for hypo_idx, hypothesis in enumerate(hypotheses_to_probe):
                print(f"    Probing hypothesis {hypo_idx+1}/{len(hypotheses_to_probe)}: '{hypothesis[:50]}...'")
                probe_result = {"hypothesis": hypothesis}

                art_use, cot_use, ans_use, raw_use = probe_hypothesis(question_text, hypothesis, "use", MODEL_NAME, TEMPERATURE_PROBE)
                probe_result["articulation_use"], probe_result["cot_use"], probe_result["answer_use"], probe_result["raw_use_output"] = art_use, cot_use, ans_use, raw_use
                time.sleep(0.5)

                art_ign, cot_ign, ans_ign, raw_ign = probe_hypothesis(question_text, hypothesis, "ignore", MODEL_NAME, TEMPERATURE_PROBE)
                probe_result["articulation_ignore"], probe_result["cot_ignore"], probe_result["answer_ignore"], probe_result["raw_ignore_output"] = art_ign, cot_ign, ans_ign, raw_ign
                time.sleep(0.5)

                experiment_log["probes"].append(probe_result)

            # Save result for this question to CSV
            try:
                experiment_log_json_string = json.dumps(experiment_log)
                csv_writer.writerow({'question_id': question_id, 'experiment_data_json': experiment_log_json_string})
                print(f"  Results for {question_id} appended to {output_csv_file}")
            except Exception as e:
                print(f"ERROR: Could not serialize or write data for {question_id} to CSV: {e}")
            
            all_results_list_for_json_dump.append(experiment_log)

    # Save all results to a single JSON file too (optional, but was there before)
    consolidated_json_path = os.path.join(OUTPUT_DIR, "_all_experiment_logs.json")
    with open(consolidated_json_path, 'w') as f_json_dump:
        json.dump(all_results_list_for_json_dump, f_json_dump, indent=4)
    print(f"\nConsolidated JSON results saved to {consolidated_json_path}")
    print(f"All experiment results appended to CSV: {output_csv_file}")
    print("\n--- Experiment Run Complete ---")

if __name__ == "__main__":
    run_full_experiment()