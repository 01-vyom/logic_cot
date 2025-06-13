import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# DEFAULT_MODEL_HF = 'Qwen/Qwen3-4B' # Default Hugging Face model (e.g., 'gpt2', 'mistralai/Mistral-7B-v0.1')
DEFAULT_MODEL_HF = 'Qwen/Qwen3-32B' # Default Hugging Face model (e.g., 'gpt2', 'mistralai/Mistral-7B-v0.1')

# Global cache for models and tokenizers to avoid reloading them repeatedly
model_cache = {}
tokenizer_cache = {}

def get_hf_response(prompt_text, model_name=DEFAULT_MODEL_HF, temperature=0.0, max_new_tokens=32768, format_type=None):
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
            # print(f"Loading model: {model_name} to {device}...")
            model_cache[model_name] = AutoModelForCausalLM.from_pretrained(
                model_name,
                # attn_implementation="flash_attention_2",
                torch_dtype="auto",  # Automatically select appropriate dtype
                device_map="auto",    # Automatically distribute model across available devices (GPUs/CPU)
                cache_dir="/orange/daisyw/v.pathak/hf_cache"
            )
        model = model_cache[model_name].eval()
        model = torch.compile(model)

        if model_name not in tokenizer_cache:
            print(f"Loading tokenizer: {model_name}...")
            tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name,cache_dir="/orange/daisyw/v.pathak/hf_cache")
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
                add_generation_prompt=True,
                # enable_thinking=True # Qwen specific, might need conditional logic or try-except
            )
            # More robust way to add Qwen-specific args if tokenizer supports them
            if "qwen" in model_name.lower():
                try:
                    final_prompt_text_for_model = tokenizer.apply_chat_template(
                        messages_for_template,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True
                    )
                    # print(f"Info: Applied chat template with enable_thinking=True for {model_name}.")
                except TypeError: # If enable_thinking is not a valid kwarg for this tokenizer
                    # print(f"Info: Applied chat template (without enable_thinking) for {model_name}.")
                    pass
            else:
                 print(f"Info: Applied chat template for {model_name}.")

        except Exception as e_template:
            # Fallback if chat template application fails (e.g., model is not chat-tuned or tokenizer doesn't support it)
            print(f"Warning: Could not apply chat template for {model_name} (falling back to direct prompt): {e_template}")
            # final_prompt_text_for_model remains prompt_text

        inputs = tokenizer([final_prompt_text_for_model], return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 2048).to(input_device)

        gen_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.pad_token_id}
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        with torch.no_grad():
            output_sequences = model.generate(**inputs, **gen_kwargs)

        # Extract only the newly generated token IDs
        generated_token_ids = output_sequences[0][inputs.input_ids.shape[-1]:].tolist()

        # Qwen-specific token ID for </think> is 151668
        # This is a more robust way to handle thinking blocks for Qwen models
        processed_text = ""
        if "qwen" in model_name.lower():
            think_tag_end_token_id = 151668 
            try:
                # Find the last occurrence of the </think> token ID
                last_think_end_idx_in_generated = len(generated_token_ids) - 1 - generated_token_ids[::-1].index(think_tag_end_token_id)
                # Decode content after the last </think> token
                content_token_ids = generated_token_ids[last_think_end_idx_in_generated + 1:]
                processed_text = tokenizer.decode(content_token_ids, skip_special_tokens=True).strip()
                
                # Optional: decode and print thinking content for debugging
                # thinking_token_ids = generated_token_ids[:last_think_end_idx_in_generated + 1]
                # thinking_content_decoded = tokenizer.decode(thinking_token_ids, skip_special_tokens=True).strip()
                # print(f"Info: Decoded thinking content: '{thinking_content_decoded[:200]}...'")
                # print(f"Info: Extracted content after last </think> token_id. Preview: '{processed_text[:100]}...'")
            except ValueError: # If </think> token is not found
                processed_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
        else: # For non-Qwen models, decode all generated tokens
            processed_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

        if format_type == 'json':
            try:
                return json.loads(processed_text)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON directly from processed model response: '{processed_text[:200]}...'")
                match = re.search(r'```json\s*([\s\S]*?)\s*```', processed_text, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode extracted JSON: {match.group(1)}")
                        return {"error": "JSON decode error from extracted markdown", "raw_response": processed_text}
                # If no markdown, try to find the first '{' and last '}' as a last resort
                try:
                    json_start_index = processed_text.find('{')
                    json_end_index = processed_text.rfind('}')
                    if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                        potential_json_str = processed_text[json_start_index : json_end_index+1]
                        parsed_json = json.loads(potential_json_str)
                        # print(f"Info: Successfully parsed JSON by finding {{...}} block from: '{potential_json_str[:100]}...'")
                        return parsed_json
                except json.JSONDecodeError:
                    pass # Fall through to return error
                return {"error": "JSON decode error, no clear JSON block or markdown", "raw_response": processed_text}
        else:
            return processed_text # Return the processed text (potentially without <think> block)
    except Exception as e:
        print(f"Error communicating with Hugging Face model: {e}")
        if format_type == 'json':
            return {"error": str(e)}
        return f"Error: {e}"