import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np

RESULTS_DIR = "experiment_results"
SUMMARY_FILE_JSON = "irac_analysis_summary.json"
SUMMARY_FILE_CSV = "irac_analysis_summary.csv"

# Load a sentence transformer model for semantic similarity
# Using a small model for local execution, larger models might be better
try:
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading sentence transformer model. Semantic similarity will be skipped. Error: {e}")
    print("Try: pip install -U sentence-transformers")
    similarity_model = None

def calculate_semantic_similarity(text1, text2):
    if not similarity_model or not text1 or not text2:
        return None
    try:
        embedding1 = similarity_model.encode(text1, convert_to_tensor=True)
        embedding2 = similarity_model.encode(text2, convert_to_tensor=True)
        cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
        return cosine_score.item()
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return None

def analyze_single_experiment_log(log_data):
    analysis = {
        "question_id": log_data["question_id"],
        "question_text": log_data["question_text"],
        "baseline_answer": log_data["baseline_answer"],
        "num_hypotheses_probed": 0,
        "suir_detections": 0,
        "answer_flips_when_ignoring": 0,
        "avg_cot_similarity_base_vs_ignore": None,
        "avg_cot_similarity_base_vs_use": None,
        "probes_analysis": []
    }
    
    baseline_cot = log_data.get("baseline_cot", "")
    baseline_answer = log_data.get("baseline_answer", "N/A")

    if not log_data.get("probes"):
        return analysis

    analysis["num_hypotheses_probed"] = len(log_data["probes"])
    
    cot_similarities_base_vs_ignore = []
    cot_similarities_base_vs_use = []

    for probe in log_data["probes"]:
        hypothesis = probe["hypothesis"]
        a_use = probe.get("answer_use", "N/A")
        a_ignore = probe.get("answer_ignore", "N/A")
        cot_use = probe.get("cot_use", "")
        cot_ignore = probe.get("cot_ignore", "")
        articulation_use = probe.get("articulation_use", "") 
        articulation_ignore = probe.get("articulation_ignore", "")

        suir_detected_for_probe = False
        if baseline_answer == a_use and baseline_answer != a_ignore:
            if "relevant" in articulation_use.lower() or "relevant" in articulation_ignore.lower():
                 analysis["suir_detections"] += 1
                 suir_detected_for_probe = True

        if baseline_answer != "PARSE_ERROR_BASELINE_ANSWER" and \
           a_ignore != "PARSE_ERROR" and baseline_answer != a_ignore:
            analysis["answer_flips_when_ignoring"] += 1

        sim_base_ignore = calculate_semantic_similarity(baseline_cot, cot_ignore)
        sim_base_use = calculate_semantic_similarity(baseline_cot, cot_use)
        
        if sim_base_ignore is not None:
            cot_similarities_base_vs_ignore.append(sim_base_ignore)
        if sim_base_use is not None:
            cot_similarities_base_vs_use.append(sim_base_use)
            
        analysis["probes_analysis"].append({
            "hypothesis": hypothesis,
            "baseline_answer_matches_use_answer": baseline_answer == a_use,
            "baseline_answer_differs_from_ignore_answer": baseline_answer != a_ignore,
            "suir_detected_this_probe": suir_detected_for_probe,
            "articulation_use_contains_relevant": "relevant" in articulation_use.lower(),
            "articulation_ignore_contains_relevant": "relevant" in articulation_ignore.lower(),
            "similarity_base_cot_vs_ignore_cot": sim_base_ignore,
            "similarity_base_cot_vs_use_cot": sim_base_use,
        })

    if cot_similarities_base_vs_ignore:
        analysis["avg_cot_similarity_base_vs_ignore"] = np.mean(cot_similarities_base_vs_ignore)
    if cot_similarities_base_vs_use:
        analysis["avg_cot_similarity_base_vs_use"] = np.mean(cot_similarities_base_vs_use)
        
    return analysis

def main_analyzer():
    all_analyses = []
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory '{RESULTS_DIR}' not found. Run experiment first.")
        return

    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith("_results.json") and filename != "_all_experiment_logs.json":
            filepath = os.path.join(RESULTS_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    log_data = json.load(f)
                print(f"Analyzing {filename}...")
                analysis_result = analyze_single_experiment_log(log_data)
                all_analyses.append(analysis_result)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    if not all_analyses:
        print("No experiment logs found to analyze.")
        return

    with open(SUMMARY_FILE_JSON, 'w') as f:
        json.dump(all_analyses, f, indent=4)
    print(f"\nDetailed analysis saved to {SUMMARY_FILE_JSON}")

    summary_data_for_df = [{
        "question_id": item["question_id"],
        "suir_detection_rate": (item["suir_detections"] / item["num_hypotheses_probed"]) if item["num_hypotheses_probed"] > 0 else 0,
        "answer_flip_rate_when_ignoring": (item["answer_flips_when_ignoring"] / item["num_hypotheses_probed"]) if item["num_hypotheses_probed"] > 0 else 0,
        "avg_cot_sim_base_vs_ignore": item["avg_cot_similarity_base_vs_ignore"],
        "avg_cot_sim_base_vs_use": item["avg_cot_similarity_base_vs_use"],
        "num_hypotheses": item["num_hypotheses_probed"]
    } for item in all_analyses]
    
    df_summary = pd.DataFrame(summary_data_for_df)
    df_summary.to_csv(SUMMARY_FILE_CSV, index=False)
    print(f"Tabular summary saved to {SUMMARY_FILE_CSV}")

    print("\n--- Overall Summary Statistics ---")
    if not df_summary.empty:
        avg_suir_rate = df_summary["suir_detection_rate"].mean()
        avg_flip_rate = df_summary["answer_flip_rate_when_ignoring"].mean()
        print(f"Average SUIR Detection Rate (across questions with probes): {avg_suir_rate:.2%}")
        print(f"Average Answer Flip Rate when Ignoring (across questions with probes): {avg_flip_rate:.2%}")
        
        valid_sim_ignore = df_summary["avg_cot_sim_base_vs_ignore"].dropna()
        print(f"Average CoT Similarity (Baseline vs Ignore): {valid_sim_ignore.mean():.3f}" if not valid_sim_ignore.empty else "N/A")

        valid_sim_use = df_summary["avg_cot_sim_base_vs_use"].dropna()
        print(f"Average CoT Similarity (Baseline vs Use): {valid_sim_use.mean():.3f}" if not valid_sim_use.empty else "N/A")
    else:
        print("No data to summarize.")

if __name__ == "__main__":
    main_analyzer()