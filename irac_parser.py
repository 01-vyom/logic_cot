import re
from hf_utils import get_hf_response # Assuming hf_utils is in the same directory or adjust import
from irac_prompts import BASELINE_COT_PROMPT_TEMPLATE # Assuming irac_prompts is in the same directory
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def _extract_concise_answer_from_text(text: str) -> str:
    """
    Extracts a concise answer from TruthfulQA-style text.
    Handles cases where the text is a full sentence or paragraph.
    """
    text = text.strip()
    
    # Remove bold/italic markers
    text = re.sub(r'\*\*|\*|__', '', text)
    
    # Split into sentences
    sentences = re.split(r'\.(?:\s+|$)', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return text[:150] if len(text) <= 150 else text[:147] + "..."
    
    # Get first sentence (often the most direct answer)
    first_sentence = sentences[0]
    
    # If first sentence is very long (> 150 chars), try to extract the core claim
    if len(first_sentence) > 150:
        # Look for main clause before commas or semicolons
        main_clause = re.split(r'[,;]', first_sentence)[0].strip()
        if len(main_clause) > 20:  # Ensure it's substantial
            return main_clause
    
    return first_sentence if len(first_sentence) <= 200 else first_sentence[:197] + "..."

def _clean_and_extract_answer_value(text_payload: str) -> str:
    """
    Cleans a raw answer payload.
    Handles both numerical answers (GSM8K) and text answers (TruthfulQA).
    
    For numerical answers:
    1. Prioritizes content within \\boxed{} if found.
    2. If payload contains $$...$$, extracts content from the last one.
    3. If payload contains $...$, extracts content from the last one.
    4. Extracts numerical values.
    
    For text answers (TruthfulQA, Yes/No, short text):
    1. Handles Yes/No/True/False answers
    2. Extracts short text answers (up to 100 characters)
    3. Normalizes capitalization
    """
    content_to_parse = text_payload.strip()
    
    # Early detection for common TruthfulQA answer patterns
    # Check for Yes/No/True/False at the start
    yes_no_pattern = re.match(r'^(yes|no|true|false)\b', content_to_parse, re.IGNORECASE)
    if yes_no_pattern:
        return yes_no_pattern.group(1).capitalize()

    # Priority 1: Extract from \boxed{...} if present in the payload
    boxed_match = re.search(r"\\boxed\{([\s\S]+?)\}", content_to_parse)
    if boxed_match:
        content_to_parse = boxed_match.group(1).strip()

    # Priority 2: Extract from $$...$$ if present in the (potentially \boxed-extracted) payload
    math_block_matches = list(re.finditer(r"\$\$([\s\S]+?)\$\$", content_to_parse, re.DOTALL))
    if math_block_matches:
        content_to_parse = math_block_matches[-1].group(1).strip()

    # Priority 3: Extract from $...$ if present and no $$...$$ found
    elif not math_block_matches:
        single_dollar_matches = list(re.finditer(r"\$([\s\S]+?)\$", content_to_parse, re.DOTALL))
        if single_dollar_matches:
            # Filter out very short matches (likely variables like $x$, $y$)
            substantial_single_matches = [match for match in single_dollar_matches 
                                        if len(match.group(1).strip()) > 1]
            if substantial_single_matches:
                content_to_parse = substantial_single_matches[-1].group(1).strip()

    # Remove common currency symbols and units for a cleaner search for numbers
    # Also remove common leading phrases like "Answer is"
    cleaned_for_num_search = content_to_parse
    cleaned_for_num_search = re.sub(r"[$\£\€\¥₹]|(?:USD|EUR|GBP|JPY|INR|CAD|AUD)\b", "", cleaned_for_num_search, flags=re.IGNORECASE).strip()
    cleaned_for_num_search = re.sub(r"\b(?:dollars?|euros?|pounds?|yen|rupees?|units?|apples?|cm|m|kg|g|liters?|ml|is|the answer is|the final answer is|result is|therefore|thus|hence)\b", "", cleaned_for_num_search, flags=re.IGNORECASE).strip()
    
    # Remove common mathematical symbols that might interfere with number extraction
    cleaned_for_num_search = re.sub(r"[\\{}]", "", cleaned_for_num_search).strip()
    
    # Remove leading/trailing non-alphanumeric characters (but preserve negative signs and decimals)
    cleaned_for_num_search = re.sub(r"^[^\w\-\.]+|[^\w\.]+$", "", cleaned_for_num_search).strip()
    cleaned_for_num_search = re.sub(r"\.$", "", cleaned_for_num_search).strip() # Remove trailing period

    # If cleaned_for_num_search is just a number (including with commas for thousands)
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?", cleaned_for_num_search):
        # Remove commas and return the number
        return cleaned_for_num_search.replace(",", "")
    elif re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned_for_num_search): # Standard number format
        return cleaned_for_num_search

    # Find all numbers in the cleaned string (including comma-separated thousands)
    all_numbers = re.findall(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?", cleaned_for_num_search)
    if all_numbers:
        # Return the last number found, removing commas
        last_number = all_numbers[-1].replace(",", "")
        return last_number

    # Enhanced fallback for text-based answers (TruthfulQA)
    # Check if this looks like a short text answer (up to 100 chars, max 10 words)
    words = content_to_parse.split()
    if len(words) <= 10 and len(content_to_parse) <= 100 and not re.search(r"[=()]", content_to_parse):
        # Clean up and return the text answer
        # Remove leading/trailing punctuation except for meaningful ones
        cleaned_text = re.sub(r'^[^\w\s]+|[^\w\s.!?]+$', '', content_to_parse).strip()
        return cleaned_text

    # Fallback for very short non-numeric answers (e.g., "Red", "Yes", "No")
    if len(words) <= 3:
        return content_to_parse.strip()

    # If answer is longer than expected, try to extract just the last sentence
    sentences = re.split(r'[.!?]\s+', content_to_parse)
    if len(sentences) > 1 and len(sentences[-1].split()) <= 10:
        return sentences[-1].strip()

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

    # Pattern 5: Single dollar + boxed pattern: $\boxed{...}$
    single_dollar_boxed_re = r"\$\\boxed\{([\s\S]+?)\}\$"
    all_single_boxed_matches = list(re.finditer(single_dollar_boxed_re, full_response, re.DOTALL))
    if all_single_boxed_matches:
        last_single_boxed_match = all_single_boxed_matches[-1]
        cot = full_response[:last_single_boxed_match.start()].strip()
        answer_payload = last_single_boxed_match.group(1).strip()
        answer = _clean_and_extract_answer_value(answer_payload)
        return cot, answer, full_response

    # Pattern 6: Standalone boxed pattern: \boxed{...} (without dollar signs)
    standalone_boxed_re = r"\\boxed\{([\s\S]+?)\}"
    all_standalone_boxed_matches = list(re.finditer(standalone_boxed_re, full_response, re.DOTALL))
    if all_standalone_boxed_matches:
        last_standalone_boxed_match = all_standalone_boxed_matches[-1]
        cot = full_response[:last_standalone_boxed_match.start()].strip()
        answer_payload = last_standalone_boxed_match.group(1).strip()
        answer = _clean_and_extract_answer_value(answer_payload)
        return cot, answer, full_response

    # Pattern 7: Single dollar math blocks: $...$ (as fallback)
    single_dollar_re = r"\$([\s\S]+?)\$"
    all_single_dollar_matches = list(re.finditer(single_dollar_re, full_response, re.DOTALL))
    if all_single_dollar_matches:
        # Filter out very short matches that are likely not answers (e.g., single variables)
        substantial_matches = [match for match in all_single_dollar_matches 
                               if len(match.group(1).strip()) > 1]
        if substantial_matches:
            last_substantial_match = substantial_matches[-1]
            cot = full_response[:last_substantial_match.start()].strip()
            answer_payload = last_substantial_match.group(1).strip()
            answer = _clean_and_extract_answer_value(answer_payload)
            return cot, answer, full_response
    
    # Pattern 8: TruthfulQA-style - Extract from "**Conclusion**:" section
    conclusion_match = re.search(r'\*\*Conclusion\*\*:\s*(.+?)(?=\n\n|\Z)', full_response, re.DOTALL | re.IGNORECASE)
    if conclusion_match:
        conclusion_text = conclusion_match.group(1).strip()
        # Use the concise answer extractor
        answer = _extract_concise_answer_from_text(conclusion_text)
        cot = full_response[:conclusion_match.start()].strip()
        return cot, answer, full_response
    
    # Pattern 8b: Look for "**Key Takeaway**:" section (alternative format)
    key_takeaway_match = re.search(r'\*\*Key Takeaway\*\*:\s*(.+?)(?=\n\n|\Z)', full_response, re.DOTALL | re.IGNORECASE)
    if key_takeaway_match:
        takeaway_text = key_takeaway_match.group(1).strip()
        answer = _extract_concise_answer_from_text(takeaway_text)
        cot = full_response[:key_takeaway_match.start()].strip()
        return cot, answer, full_response
    
    # Pattern 9: Extract last 1-2 sentences as answer (TruthfulQA fallback)
    # This handles cases where there's no explicit conclusion marker
    sentences = re.split(r'(?<=[.!?])\s+', full_response.strip())
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
    
    if len(sentences) >= 2:
        # Try last 2 sentences combined (often forms a complete answer)
        last_two = ' '.join(sentences[-2:])
        if len(last_two) <= 300:
            # Use the concise answer extractor
            answer = _extract_concise_answer_from_text(last_two)
            # Find where these sentences start in the original text
            last_sentence_start = full_response.rfind(sentences[-2])
            if last_sentence_start > 0:
                cot = full_response[:last_sentence_start].strip()
                return cot, answer, full_response
        
        # If last 2 sentences are too long, just use the last one
        last_sentence = sentences[-1]
        if len(last_sentence) <= 200:
            answer = _extract_concise_answer_from_text(last_sentence)
            last_sentence_start = full_response.rfind(last_sentence)
            if last_sentence_start > 0:
                cot = full_response[:last_sentence_start].strip()
                return cot, answer, full_response
    
    # Pattern 10: As a last resort, use the first sentence as answer (some models put answer first)
    if sentences and len(sentences[0]) <= 200:
        first_sentence = sentences[0]
        answer = _extract_concise_answer_from_text(first_sentence)
        if answer and len(answer) > 5:  # Ensure it's substantial
            cot = full_response[len(first_sentence):].strip()
            return cot, answer, full_response
        
    # Absolute fallback if no specific answer pattern is found
    print(f"Warning: Baseline answer parsing failed for full response: {full_response[:200]}...")
    cot = full_response # Whole response as CoT
    answer = "PARSE_ERROR_BASELINE_ANSWER"
    return cot, answer, full_response

def parse_articulation_and_cot(text_response):
    """
    Parses the model response to separate articulation of H_implicit
    from the subsequent Chain-of-Thought and answer.
    Enhanced to handle various response formats more robustly.
    """
    articulation_end_marker = "ARTICULATION_END"
    text_response = text_response.strip()
    
    # Check if response starts directly with "The final answer is..."
    final_answer_at_start = re.match(r"^\s*The final answer is\s*(.+)", text_response, re.IGNORECASE | re.DOTALL)
    if final_answer_at_start:
        # Model jumped straight to answer without articulation
        print(f"Info: Response starts directly with final answer, assuming no articulation provided.")
        articulation = ""
        cot = ""  # No CoT provided
        answer_payload = final_answer_at_start.group(1).strip()
        ans_letter_match = re.match(r'^\(?([A-Z])\)?\.?', answer_payload)
        answer = ans_letter_match.group(1) if ans_letter_match else _clean_and_extract_answer_value(answer_payload)
        return articulation, cot, answer

    # Standard parsing with marker
    parts = text_response.split(articulation_end_marker, 1)
    articulation = parts[0].strip() if len(parts) > 1 else ""
    cot_and_answer_text = parts[1].strip() if len(parts) > 1 else text_response

    # Enhanced fallback logic when marker is not found
    if not articulation and articulation_end_marker not in text_response:
        print(f"Warning: ARTICULATION_END marker not found in response: {text_response[:200]}...")
        
        # Strategy 1: Look for "The final answer is" and split there
        final_answer_match = re.search(r"The final answer is\s*(.+)", text_response, re.IGNORECASE | re.DOTALL)
        if final_answer_match:
            # Everything before "The final answer is" could be articulation + CoT
            pre_answer_text = text_response[:final_answer_match.start()].strip()
            
            # Try to split articulation from CoT using paragraph breaks
            split_lines = pre_answer_text.split('\n\n', 1)
            if len(split_lines) > 1 and len(split_lines[0].split('\n')) <= 5:
                # First paragraph seems like articulation
                articulation = split_lines[0].strip()
                cot_and_answer_text = split_lines[1].strip() + "\n\n" + text_response[final_answer_match.start():].strip()
            else:
                # Use first few lines as articulation if they're short enough
                lines = pre_answer_text.split('\n')
                if len(lines) > 2 and len(lines[0]) < 200:  # First line seems like articulation
                    articulation = lines[0].strip()
                    cot_and_answer_text = '\n'.join(lines[1:]).strip() + "\n\n" + text_response[final_answer_match.start():].strip()
                else:
                    # Can't reliably separate, assume no distinct articulation
                    articulation = ""
                    cot_and_answer_text = text_response.strip()
        else:
            # No final answer pattern found, use original fallback
            split_lines = text_response.split('\n\n', 1)
            if len(split_lines) > 1 and len(split_lines[0].split('\n')) < 5:
                articulation = split_lines[0].strip()
                cot_and_answer_text = split_lines[1].strip()
            else:
                articulation = ""
                cot_and_answer_text = text_response.strip()

    # Parse CoT and answer from the remaining text
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
        # Try to extract answer using the same patterns as baseline parsing
        answer = extract_answer_from_text(cot_and_answer_text)
        if answer != "PARSE_ERROR":
            cot = cot_and_answer_text # Keep all as CoT if we found an answer through other means
        else:
            cot = cot_and_answer_text
            answer = "PARSE_ERROR_ARTICULATION_ANSWER"
            print(f"Warning: Could not parse answer from articulation response: {cot_and_answer_text[:200]}...")

    return articulation, cot, answer

def extract_answer_from_text(text):
    """
    Helper function to extract answers using the same patterns as baseline parsing.
    Returns the extracted answer or "PARSE_ERROR" if no pattern matches.
    """
    # Try the same patterns used in get_baseline_cot_and_answer
    patterns_to_try = [
        r"\$\\boxed\{([\s\S]+?)\}\$",           # $\boxed{...}$
        r"\\boxed\{([\s\S]+?)\}",               # \boxed{...}
        r"\$\$([\s\S]+?)\$\$",                  # $$...$$
        r"\$([\s\S]+?)\$"                       # $...$
    ]
    
    for pattern in patterns_to_try:
        matches = list(re.finditer(pattern, text, re.DOTALL))
        if matches:
            # Filter substantial matches for single dollar patterns
            if pattern == r"\$([\s\S]+?)\$":
                substantial_matches = [match for match in matches if len(match.group(1).strip()) > 1]
                if substantial_matches:
                    answer_payload = substantial_matches[-1].group(1).strip()
                    return _clean_and_extract_answer_value(answer_payload)
            else:
                answer_payload = matches[-1].group(1).strip()
                return _clean_and_extract_answer_value(answer_payload)
    
    return "PARSE_ERROR"
