from ollama_utils import get_ollama_response, parse_articulation_and_cot, DEFAULT_MODEL
from irac_prompts import (
    BASELINE_COT_PROMPT_TEMPLATE,
    HYPOTHESIS_GENERATION_PROMPT_TEMPLATE,
    ARTICULATE_AND_USE_PROMPT_TEMPLATE,
    ARTICULATE_AND_IGNORE_PROMPT_TEMPLATE
)
import json
import time
import re

def get_baseline_cot_and_answer(question_text, model_name, temperature=0.0):
    """Gets the baseline CoT and answer for a question."""
    prompt = BASELINE_COT_PROMPT_TEMPLATE.format(question=question_text)
    full_response = get_ollama_response(prompt, model=model_name, temperature=temperature)
    
    # Basic parsing for CoT and answer from baseline
    # This might need to be more robust depending on model's output format
    answer_marker_alt = "The final answer is"
    match = re.search(rf"{answer_marker_alt}\s*(.+)", full_response, re.IGNORECASE | re.DOTALL)
    if match:
        cot = full_response[:match.start()].strip()
        answer_text = match.group(1).strip()
        ans_match = re.match(r'^\(?([A-Z])\)?\.?', answer_text) # For (A) or A.
        answer = ans_match.group(1) if ans_match else answer_text
    else:
        print(f"Warning: Baseline answer parsing failed for: {full_response}")
        cot = full_response
        answer = "PARSE_ERROR_BASELINE_ANSWER"
        
    return cot, answer, full_response


def generate_hypotheses(question_text, few_shot_examples_text, model_name, temperature=0.5):
    """Generates potential implicit hypotheses using an LLM."""
    prompt = HYPOTHESIS_GENERATION_PROMPT_TEMPLATE.format(
        question=question_text,
        few_shot_examples=few_shot_examples_text
    )
    # Expecting JSON output, Ollama's format='json' can help
    response_data = get_ollama_response(prompt, model=model_name, temperature=temperature, format_type='json')
    
    if isinstance(response_data, dict) and "hypotheses" in response_data and isinstance(response_data["hypotheses"], list):
        return response_data["hypotheses"], response_data # return raw too
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
                        return json_content["hypotheses"], response_data
            except Exception as e:
                print(f"Could not parse hypotheses string: {e}")
        return [], response_data

def probe_hypothesis(question_text, hypothesis, probe_type, model_name, temperature=0.0):
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

    full_response = get_ollama_response(prompt, model=model_name, temperature=temperature)
    articulation, cot, answer = parse_articulation_and_cot(full_response)
    return articulation, cot, answer, full_response

if __name__ == '__main__':
    # Example usage (will make actual Ollama calls if Ollama is running)
    test_question = "A bakery made 30 apple pies and 40 cherry pies. They sold 1/3 of the apple pies and 1/2 of the cherry pies. They then baked 15 more apple pies. How many apple pies do they have now?"
    model_to_test = DEFAULT_MODEL # Make sure this model is pulled in Ollama

    print(f"--- Testing Baseline CoT for Q: '{test_question[:50]}...' ---")
    c_base, a_base, raw_baseline = get_baseline_cot_and_answer(test_question, model_to_test)
    print(f"Baseline CoT: {c_base[:100]}...")
    print(f"Baseline Answer: {a_base}")
    time.sleep(1) 

    print(f"--- Testing Hypothesis Generation for Q: '{test_question[:50]}...' ---")
    hypotheses, raw_hypo_gen = generate_hypotheses(test_question, "No few-shot examples provided for this test.", model_to_test, temperature=0.7)
    if hypotheses:
        print(f"Generated Hypotheses: {hypotheses}")
        test_hypothesis = hypotheses[0]
        time.sleep(1)

        print(f"--- Testing Articulate & Use for Hypothesis: '{test_hypothesis}' ---")
        art_use, cot_use, ans_use, raw_use = probe_hypothesis(test_question, test_hypothesis, "use", model_to_test)
        print(f"Articulation (Use): {art_use}")
        print(f"CoT (Use): {cot_use[:100]}...")
        print(f"Answer (Use): {ans_use}")
        time.sleep(1)

        print(f"--- Testing Articulate & Ignore for Hypothesis: '{test_hypothesis}' ---")
        art_ign, cot_ign, ans_ign, raw_ign = probe_hypothesis(test_question, test_hypothesis, "ignore", model_to_test)
        print(f"Articulation (Ignore): {art_ign}")
        print(f"CoT (Ignore): {cot_ign[:100]}...")
        print(f"Answer (Ignore): {ans_ign}")
    else:
        print("No hypotheses generated, skipping further tests.")