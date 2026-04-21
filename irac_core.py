from hf_utils import get_hf_response, DEFAULT_MODEL_HF # parse_articulation_and_cot removed
from irac_prompts import (
    BASELINE_COT_PROMPT_TEMPLATE,
    HYPOTHESIS_GENERATION_PROMPT_TEMPLATE,
    HYPOTHESIS_GENERATION_PROMPT_TEMPLATE_QA,
    HYPOTHESIS_GENERATION_PROMPT_TEMPLATE_STRATEGY,
    ARTICULATE_AND_USE_PROMPT_TEMPLATE,
    ARTICULATE_AND_IGNORE_PROMPT_TEMPLATE
)
from irac_parser import get_baseline_cot_and_answer, parse_articulation_and_cot, _clean_and_extract_answer_value
import json
import time
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Meta-language words to filter out from TruthfulQA hypotheses
BANNED_META_WORDS = ["model might", "model may", "model assumes", "model could", 
                     "ai might", "ai may", "ai assumes", "ai could",
                     "llm might", "llm may", "llm assumes", "llm could",
                     "system might", "system may", "system assumes", "system could"]

def filter_meta_hypotheses(hypotheses):
    """Remove hypotheses containing meta-language about what a model/AI might do."""
    filtered = []
    for h in hypotheses:
        h_lower = h.lower()
        # Check if any banned phrase is in the hypothesis
        if not any(banned in h_lower for banned in BANNED_META_WORDS):
            filtered.append(h)
        else:
            print(f"  [Filtered out meta-language hypothesis]: {h[:80]}...")
    return filtered



def generate_hypotheses(question_text, few_shot_examples_text, model_name, temperature=0.5, max_new_tokens=32768, dataset_type="gsm8k"):
    """
    Generates potential implicit hypotheses using an LLM.
    
    Args:
        question_text: The question to generate hypotheses for
        few_shot_examples_text: Few-shot examples (used for GSM8K)
        model_name: The model to use for generation
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
        dataset_type: "gsm8k" or "truthful_qa" - determines which prompt to use
    
    Returns:
        Tuple of (list of hypotheses, raw response data)
    """
    # Use dataset-specific prompt
    if dataset_type == "truthful_qa":
        prompt = HYPOTHESIS_GENERATION_PROMPT_TEMPLATE_QA.format(
            question=question_text
        )
    elif dataset_type == "strategy_qa":
        prompt = HYPOTHESIS_GENERATION_PROMPT_TEMPLATE_STRATEGY.format(
            question=question_text
        )
    else:
        # Original GSM8K prompt
        prompt = HYPOTHESIS_GENERATION_PROMPT_TEMPLATE.format(
            question=question_text,
            few_shot_examples=few_shot_examples_text
        )
    
    response_data = get_hf_response(prompt, model_name=model_name, format_type='json', max_new_tokens=max_new_tokens)

    hypotheses = []
    if isinstance(response_data, dict) and "hypotheses" in response_data and isinstance(response_data["hypotheses"], list):
        hypotheses = response_data["hypotheses"]
    elif isinstance(response_data, dict) and "error" in response_data:
         print(f"Error in hypothesis generation: {response_data['error']}")
         return [], response_data
    else:
        print(f"Warning: Hypotheses not in expected JSON list format: {response_data}")
        # Try to extract from a string if it's not parsed as JSON
        if isinstance(response_data, str):
            try:
                # Attempt to find a JSON block if the model wrapped it
                match = re.search(r'```json\s*([\s\S]*?)\s*```', response_data)
                if match:
                    json_content = json.loads(match.group(1))
                    if "hypotheses" in json_content and isinstance(json_content["hypotheses"], list):
                        hypotheses = json_content["hypotheses"]
            except Exception as e:
                print(f"Could not parse hypotheses string: {e}")
        if not hypotheses:
            return [], response_data
    
    # For TruthfulQA and StrategyQA, apply additional filtering to remove any meta-language that slipped through
    if dataset_type in ("truthful_qa", "strategy_qa"):
        original_count = len(hypotheses)
        hypotheses = filter_meta_hypotheses(hypotheses)
        if len(hypotheses) < original_count:
            print(f"  [Filtered {original_count - len(hypotheses)} meta-language hypotheses]")
    
    return hypotheses, response_data

def probe_hypothesis(question_text, hypothesis, probe_type, model_name, temperature=0.0, max_new_tokens=32768):
    """
    Probes a single hypothesis (Articulate & Use or Articulate & Ignore).
    probe_type: "use" or "ignore"
    """
    if probe_type == "use":
        prompt = ARTICULATE_AND_USE_PROMPT_TEMPLATE.format(question=question_text, hypothesis=hypothesis)
    elif probe_type == "ignore":
        prompt = ARTICULATE_AND_IGNORE_PROMPT_TEMPLATE.format(question=question_text, hypothesis=hypothesis)
    else:
        raise ValueError("Invalid probe_type. Must be 'use' or 'ignore'.")

    full_response = get_hf_response(prompt, model_name=model_name, temperature=temperature, max_new_tokens=max_new_tokens)
    articulation, cot, answer = parse_articulation_and_cot(full_response)
    return articulation, cot, answer, full_response

# if __name__ == '__main__':
#     test_question = "A bakery made 30 apple pies and 40 cherry pies. They sold 1/3 of the apple pies and 1/2 of the cherry pies. They then baked 15 more apple pies. How many apple pies do they have now?"
#     model_to_test = DEFAULT_MODEL_HF 

#     print(f"--- Testing Baseline CoT for Q: '{test_question[:50]}...' ---")
#     c_base, a_base, raw_baseline = get_baseline_cot_and_answer(test_question, model_to_test)
#     print(f"Baseline CoT: {c_base[:100]}...")
#     print(f"Baseline Answer: {a_base}")
#     # time.sleep(1) 

#     print(f"--- Testing Hypothesis Generation for Q: '{test_question[:50]}...' ---")
#     hypotheses, raw_hypo_gen = generate_hypotheses(test_question, "No few-shot examples provided for this test.", model_to_test, temperature=0.7)
#     if hypotheses:
#         print(f"Generated Hypotheses: {hypotheses}")
#         test_hypothesis = hypotheses[0]
#         # time.sleep(1)

#         print(f"--- Testing Articulate & Use for Hypothesis: '{test_hypothesis}' ---")
#         art_use, cot_use, ans_use, raw_use = probe_hypothesis(test_question, test_hypothesis, "use", model_to_test)
#         print(f"Articulation (Use): {art_use}")
#         print(f"CoT (Use): {cot_use[:100]}...")
#         print(f"Answer (Use): {ans_use}")
#         time.sleep(1)

#         print(f"--- Testing Articulate & Ignore for Hypothesis: '{test_hypothesis}' ---")
#         art_ign, cot_ign, ans_ign, raw_ign = probe_hypothesis(test_question, test_hypothesis, "ignore", model_to_test)
#         print(f"Articulation (Ignore): {art_ign}")
#         print(f"CoT (Ignore): {cot_ign[:100]}...")
#         print(f"Answer (Ignore): {ans_ign}")
#     else:
#         print("No hypotheses generated, skipping further tests.")
