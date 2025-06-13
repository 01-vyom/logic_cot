import re
from hf_utils import get_hf_response # Assuming hf_utils is in the same directory or adjust import
from irac_prompts import BASELINE_COT_PROMPT_TEMPLATE # Assuming irac_prompts is in the same directory
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def _clean_and_extract_answer_value(text_payload: str) -> str:
    """
    Cleans a raw answer payload.
    1. Prioritizes content within \\boxed{} if found.
    2. If payload contains $$...$$, extracts content from the last one.
    3. Otherwise, uses the original payload.
    4. From the resulting text, extracts a numerical value or a short textual answer.
    """
    content_to_parse = text_payload

    # Priority 1: Extract from \boxed{...} if present in the payload
    boxed_match = re.search(r"\\boxed\{([\s\S]+?)\}", content_to_parse)
    if boxed_match:
        content_to_parse = boxed_match.group(1).strip()

    # Priority 2: Extract from $$...$$ if present in the (potentially \boxed-extracted) payload
    math_block_matches = list(re.finditer(r"\$\$([\s\S]+?)\$\$", content_to_parse, re.DOTALL))
    if math_block_matches:
        content_to_parse = math_block_matches[-1].group(1).strip()

    # Remove common currency symbols and units for a cleaner search for numbers
    # Also remove common leading phrases like "Answer is"
    cleaned_for_num_search = content_to_parse
    cleaned_for_num_search = re.sub(r"[$\£\€\¥₹]|(?:USD|EUR|GBP|JPY|INR|CAD|AUD)\b", "", cleaned_for_num_search, flags=re.IGNORECASE).strip()
    cleaned_for_num_search = re.sub(r"\b(?:dollars?|euros?|pounds?|yen|rupees?|units?|apples?|cm|m|kg|g|liters?|ml|is|the answer is|the final answer is|result is)\b", "", cleaned_for_num_search, flags=re.IGNORECASE).strip()
    cleaned_for_num_search = re.sub(r"^\W+|\W+$", "", cleaned_for_num_search) # Remove leading/trailing non-alphanumeric
    cleaned_for_num_search = re.sub(r"\.$", "", cleaned_for_num_search).strip() # Remove trailing period

    # If cleaned_for_num_search is just a number
    if re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned_for_num_search): # Matches integer or float, possibly negative
        return cleaned_for_num_search

    # Find all numbers in the cleaned string
    all_numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned_for_num_search)
    if all_numbers:
        # Return the last number found, as it's often the final answer in a phrase
        return all_numbers[-1]

    # Fallback for very short non-numeric answers (e.g., "Red", "Yes")
    if len(content_to_parse.split()) <= 3 and not re.search(r"[=()]", content_to_parse) and not all_numbers:
        return content_to_parse.strip()

    return content_to_parse.strip() # Absolute fallback

def get_baseline_cot_and_answer(question_text, model_name, temperature=0.0, max_new_tokens=32768):
    """Gets the baseline CoT and answer for a question."""
    prompt = BASELINE_COT_PROMPT_TEMPLATE.format(question=question_text)
    full_response = get_hf_response(prompt, model_name=model_name, temperature=temperature, max_new_tokens=max_new_tokens)
    
    # Pattern 1: "The final answer is ..."
    final_answer_marker_re = r"The final answer is\s*(.+)"
    match = re.search(final_answer_marker_re, full_response, re.IGNORECASE | re.DOTALL)
    if match:
        cot = full_response[:match.start()].strip()
        answer_payload = match.group(1).strip()
        ans_letter_match = re.match(r'^\(?([A-Z])\)?\.?', answer_payload) # For (A) or A.
        answer = ans_letter_match.group(1) if ans_letter_match else _clean_and_extract_answer_value(answer_payload)
        return cot, answer, full_response

    # Pattern 2: "**Answer:** ... \boxed{X}"
    boxed_answer_re = r"\*\*Answer:\*\*\s*.*?\s*\\boxed\{([\s\S]+?)\}"
    match = re.search(boxed_answer_re, full_response, re.IGNORECASE | re.DOTALL)
    if match:
        cot = full_response[:match.start()].strip()
        answer_payload = match.group(1).strip() # Content inside \boxed{}
        answer = _clean_and_extract_answer_value(answer_payload)
        return cot, answer, full_response

    # Pattern 3: "**Answer:** X" (more general, after specific boxed version)
    simple_answer_keyword_re = r"\*\*Answer:\*\*\s*(.+)"
    match = re.search(simple_answer_keyword_re, full_response, re.IGNORECASE | re.DOTALL)
    if match:
        cot = full_response[:match.start()].strip()
        answer_payload = match.group(1).strip()
        answer = _clean_and_extract_answer_value(answer_payload)
        return cot, answer, full_response

    # Pattern 4: Last $$...$$ block as answer (if no other markers found)
    math_block_re = r"\$\$([\s\S]+?)\$\$"
    all_math_block_matches = list(re.finditer(math_block_re, full_response, re.DOTALL))
    if all_math_block_matches:
        last_math_block_match = all_math_block_matches[-1]
        cot = full_response[:last_math_block_match.start()].strip()
        answer_payload = last_math_block_match.group(1).strip()
        answer = _clean_and_extract_answer_value(answer_payload)
        return cot, answer, full_response
        
    # Fallback if no specific answer pattern is found
    print(f"Warning: Baseline answer parsing failed for full response: {full_response}")
    cot = full_response # Whole response as CoT
    answer = "PARSE_ERROR_BASELINE_ANSWER"
    return cot, answer, full_response

def parse_articulation_and_cot(text_response):
    """
    Parses the model response to separate articulation of H_implicit
    from the subsequent Chain-of-Thought and answer.
    """
    articulation_end_marker = "ARTICULATION_END"
    parts = text_response.split(articulation_end_marker, 1)
    articulation = parts[0].strip() if len(parts) > 1 else ""
    cot_and_answer_text = parts[1].strip() if len(parts) > 1 else text_response # If marker not found, assume all is cot_and_answer

    if not articulation and articulation_end_marker not in text_response:
        print(f"Warning: ARTICULATION_END marker not found in response: {text_response[:200]}...")
        # Heuristic: if no marker, assume first paragraph or first few lines are articulation
        split_lines = text_response.split('\n\n', 1) # Split by first double newline
        if len(split_lines) > 1 and len(split_lines[0].split('\n')) < 5: # Arbitrary small number of lines for articulation
            articulation = split_lines[0].strip()
            cot_and_answer_text = split_lines[1].strip()
        else: # If no double newline or first part is too long, assume no distinct articulation
            articulation = "" 
            cot_and_answer_text = text_response.strip()

    final_answer_re = r"The final answer is\s*(.+)"
    match = re.search(final_answer_re, cot_and_answer_text, re.IGNORECASE | re.DOTALL)
    if match:
        cot = cot_and_answer_text[:match.start()].strip()
        answer_payload = match.group(1).strip()
        # Check for (A) or A. style answers first
        ans_letter_match = re.match(r'^\(?([A-Z])\)?\.?', answer_payload)
        if ans_letter_match:
            answer = ans_letter_match.group(1)
        else: # Otherwise, clean the payload
            answer = _clean_and_extract_answer_value(answer_payload)
    else:
        cot = cot_and_answer_text # Assume all remaining is CoT if "The final answer is" not found
        answer = "PARSE_ERROR_ARTICULATION_ANSWER"
        print(f"Warning: Could not parse answer using '{final_answer_re}' from: {cot_and_answer_text[:200]}...")

    return articulation, cot, answer