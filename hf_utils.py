import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re

DEFAULT_MODEL_HF = 'Qwen/Qwen3-0.6B' # Default Hugging Face model (e.g., 'gpt2', 'mistralai/Mistral-7B-v0.1')

# Global cache for models and tokenizers to avoid reloading them repeatedly
model_cache = {}
tokenizer_cache = {}

def get_hf_response(prompt_text, model_name=DEFAULT_MODEL_HF, temperature=0.0, max_new_tokens=512, format_type=None):
    """
    Sends a prompt to a Hugging Face model and returns the response.
    Args:
        prompt_text (str): The prompt to send to the model.
        model_name (str): The Hugging Face model identifier.
        temperature (float): The temperature for generation. 0.0 means greedy.
        max_new_tokens (int): Maximum number of new tokens to generate.
        format_type (str, optional): 'json' if the output is expected to be JSON.

    Returns:
        str or dict: The model's response, or a dict if format_type is 'json'.
    """
    try:
        # device = "cuda" if torch.cuda.is_available() else "cpu" # device_map="auto" handles this

        if model_name not in model_cache:
            print(f"Loading model: {model_name} to {device}...")
            model_cache[model_name] = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",  # Automatically select appropriate dtype
                device_map="auto"    # Automatically distribute model across available devices (GPUs/CPU)
            )
        model = model_cache[model_name]

        if model_name not in tokenizer_cache:
            print(f"Loading tokenizer: {model_name}...")
            tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
        tokenizer = tokenizer_cache[model_name]
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Determine input device based on model's placement by device_map
        input_device = next(model.parameters()).device

        # Prepare prompt using chat template if available
        final_prompt_text_for_model = prompt_text
        try:
            messages_for_template = [{"role": "user", "content": prompt_text}]
            final_prompt_text_for_model = tokenizer.apply_chat_template(
                messages_for_template,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e_template:
            # Fallback if chat template application fails (e.g., model is not chat-tuned or tokenizer doesn't support it)
            print(f"Warning: Could not apply chat template for {model_name} (falling back to direct prompt): {e_template}")
            # final_prompt_text_for_model remains prompt_text

        inputs = tokenizer([final_prompt_text_for_model], return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 2048).to(input_device)
        
        gen_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.pad_token_id}
        if temperature > 0.0: 
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        else: # Greedy decoding
            gen_kwargs["do_sample"] = False
            
        output_sequences = model.generate(**inputs, **gen_kwargs)
        
        # Decode only the newly generated tokens
        generated_text = tokenizer.decode(output_sequences[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

        if format_type == 'json':
            try:
                return json.loads(generated_text)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from model response: {generated_text}")
                match = re.search(r'```json\s*([\s\S]*?)\s*```', generated_text, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode extracted JSON: {match.group(1)}")
                        return {"error": "JSON decode error", "raw_response": generated_text}
                return {"error": "JSON decode error", "raw_response": generated_text}
        else:
            return generated_text
    except Exception as e:
        print(f"Error communicating with Hugging Face model: {e}")
        if format_type == 'json':
            return {"error": str(e)}
        return f"Error: {e}"

def parse_articulation_and_cot(text_response):
    """
    Parses the model response to separate articulation of H_implicit
    from the subsequent Chain-of-Thought and answer.
    Uses a simple delimiter. (This function is generic and can be reused)
    """
    articulation_end_marker = "ARTICULATION_END"
    if articulation_end_marker in text_response:
        parts = text_response.split(articulation_end_marker, 1)
        articulation = parts[0].strip()
        cot_and_answer_text = parts[1].strip()
    else:
        print("Warning: ARTICULATION_END marker not found. Attempting basic split.")
        lines = text_response.split('\n')
        articulation = lines[0] if lines else ""
        cot_and_answer_text = "\n".join(lines[1:]) if len(lines) > 1 else ""

    match = re.search(r"The final answer is\s*(.+)", cot_and_answer_text, re.IGNORECASE | re.DOTALL)
    if match:
        cot = cot_and_answer_text[:match.start()].strip()
        answer_text = match.group(1).strip()
        ans_match = re.match(r'^\(?([A-Z])\)?\.?', answer_text)
        answer = ans_match.group(1) if ans_match else answer_text
    else:
        cot = cot_and_answer_text
        answer = "PARSE_ERROR"
        print(f"Warning: Could not parse answer from: {cot_and_answer_text}")
    return articulation, cot, answer

if __name__ == '__main__':
    print("Testing HF model call:")
    test_prompt = "What is the capital of France? The final answer is " # Prompting for a specific end
    # Note: Small models like gpt2 might not follow "The final answer is" well without more specific prompting or fine-tuning.
    basic_response = get_hf_response(test_prompt, model_name='gpt2', temperature=0.1, max_new_tokens=50)
    print(f"Prompt: {test_prompt}\nResponse: {basic_response}\n")

    print("Testing JSON HF call (model needs to be prompted to produce JSON):")
    json_prompt = "List three colors in JSON format like {\"colors\": [\"color1\", \"color2\", \"color3\"]}. Output only the JSON. JSON: "
    json_response = get_hf_response(json_prompt, model_name='gpt2', temperature=0.1, format_type='json', max_new_tokens=100)
    print(f"Prompt: {json_prompt}\nResponse: {json_response}\n")

    print("Testing articulation and CoT parsing:")
    test_response_text_hf = """This factor is relevant. It simplifies the problem.
ARTICULATION_END
Step 1: Do this.
Step 2: Then do that.
The final answer is (C)
"""
    articulation, cot, answer = parse_articulation_and_cot(test_response_text_hf)
    print(f"Original Text:\n{test_response_text_hf}")
    print(f"\nParsed Articulation: {articulation}")
    print(f"Parsed CoT: {cot}")
    print(f"Parsed Answer: {answer}\n")