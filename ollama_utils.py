import ollama
import json
import re

DEFAULT_MODEL = 'qwen3:0.6b' # Or your preferred Ollama model

def get_ollama_response(prompt, model=DEFAULT_MODEL, temperature=0.0, format_type=None):
    """
    Sends a prompt to the Ollama API and returns the response.
    Args:
        prompt (str): The prompt to send to the model.
        model (str): The Ollama model to use.
        temperature (float): The temperature for generation.
        format_type (str, optional): 'json' if the output should be in JSON format.

    Returns:
        str or dict: The model's response, or a dict if format_type is 'json'.
    """
    try:
        options = {
            "temperature": temperature
        }
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options=options,
            format=format_type if format_type == 'json' else '' # API expects empty string for non-json
        )
        if format_type == 'json':
            # The API should already return a dict if format='json' and the content is valid JSON
            # but sometimes the content might still be a string that needs parsing
            content = response['message']['content']
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from model response: {content}")
                # Attempt to extract JSON from a potentially messy string
                match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode extracted JSON: {match.group(1)}")
                        return {"error": "JSON decode error", "raw_response": content}
                return {"error": "JSON decode error", "raw_response": content}
        else:
            return response['message']['content']
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        if format_type == 'json':
            return {"error": str(e)}
        return f"Error: {e}"

def parse_articulation_and_cot(text_response):
    """
    Parses the model response to separate articulation of H_implicit
    from the subsequent Chain-of-Thought and answer.
    Uses a simple delimiter.
    """
    articulation_end_marker = "ARTICULATION_END"
    if articulation_end_marker in text_response:
        parts = text_response.split(articulation_end_marker, 1)
        articulation = parts[0].strip()
        cot_and_answer_text = parts[1].strip()
    else:
        # Fallback if marker is not found (less ideal)
        print("Warning: ARTICULATION_END marker not found. Attempting basic split.")
        lines = text_response.split('\n')
        articulation = lines[0] if lines else "" # Assume first line is articulation
        cot_and_answer_text = "\n".join(lines[1:]) if len(lines) > 1 else ""

    # Further split CoT and answer
    answer_marker_alt = "The final answer is"
    # Regex to find variations of the answer marker, case insensitive
    match = re.search(rf"{answer_marker_alt}\s*(.+)", cot_and_answer_text, re.IGNORECASE | re.DOTALL)
    if match:
        cot = cot_and_answer_text[:match.start()].strip()
        answer_text = match.group(1).strip()
        # Try to extract common answer patterns like (A), A, etc.
        ans_match = re.match(r'^\(?([A-Z])\)?\.?', answer_text)
        answer = ans_match.group(1) if ans_match else answer_text # Fallback to full text if no pattern
    else:
        # Fallback if specific answer marker is not found
        cot = cot_and_answer_text # Assume all remaining is CoT
        answer = "PARSE_ERROR" # Or handle differently
        print(f"Warning: Could not parse answer from: {cot_and_answer_text}")

    return articulation, cot, answer

if __name__ == '__main__':
    # Test functions
    # print("Testing basic Ollama call:")
    # test_prompt = "What is the capital of France?"
    # basic_response = get_ollama_response(test_prompt)
    # print(f"Prompt: {test_prompt}\nResponse: {basic_response}\n")

    # print("Testing JSON Ollama call (model might not produce valid JSON easily):")
    # json_prompt = "List three colors in JSON format like {\"colors\": [\"color1\", \"color2\", \"color3\"]}."
    # json_response = get_ollama_response(json_prompt, format_type='json')
    # print(f"Prompt: {json_prompt}\nResponse: {json_response}\n")

    print("Testing articulation and CoT parsing:")
    test_response_text = """This factor is relevant. It simplifies the problem.
ARTICULATION_END
Step 1: Do this.
Step 2: Then do that.
The final answer is (A)
"""
    articulation, cot, answer = parse_articulation_and_cot(test_response_text)
    print(f"Original Text:\n{test_response_text}")
    print(f"\nParsed Articulation: {articulation}")
    print(f"Parsed CoT: {cot}")
    print(f"Parsed Answer: {answer}\n")

    test_response_text_no_marker = """This factor is relevant because it is common sense.
Step 1: Do this.
Step 2: Then do that.
The final answer is B.
"""
    articulation, cot, answer = parse_articulation_and_cot(test_response_text_no_marker)
    print(f"Original Text (no marker):\n{test_response_text_no_marker}")
    print(f"\nParsed Articulation: {articulation}")
    print(f"Parsed CoT: {cot}")
    print(f"Parsed Answer: {answer}")