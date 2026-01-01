import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import warnings
from functools import lru_cache
warnings.filterwarnings("ignore", category=FutureWarning)


# DEFAULT_MODEL_HF = 'Qwen/Qwen3-4B' # Default Hugging Face model (e.g., 'gpt2', 'mistralai/Mistral-7B-v0.1')
DEFAULT_MODEL_HF = 'Qwen/Qwen3-32B' # Default Hugging Face model (e.g., 'gpt2', 'mistralai/Mistral-7B-v0.1')

# Global cache for models and tokenizers
model_cache = {}
tokenizer_cache = {}
compiled_model_cache = {}

# Cache for processed chat templates and tokenized inputs
@lru_cache(maxsize=1000)
def get_chat_template(model_name: str, prompt_text: str, enable_thinking: bool = False):
    """Cache chat template processing to avoid repeated computation."""
    if model_name not in tokenizer_cache:
        return None

    tokenizer = tokenizer_cache[model_name]
    try:
        messages = [{"role": "user", "content": prompt_text}]
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }

        if enable_thinking and "qwen" in model_name.lower():
            try:
                kwargs["enable_thinking"] = True
            except TypeError:
                pass

        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        return None

def load_model_optimized(model_name: str):
    """Load model with optimized settings."""
    if model_name in model_cache:
        return model_cache[model_name]

    print(f"Loading model: {model_name}...")

    # Use more aggressive optimization flags
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir="/home/ec2-user/code/personal/hf_cache",
        # Additional optimizations
        use_cache=True,  # Enable KV cache
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        trust_remote_code=True,  # Allow custom model code
        # attn_implementation="flash_attention_2",  # Uncomment if available
    )

    model.eval()

    # Enable memory efficient attention if available
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()

    model_cache[model_name] = model
    return model

def get_compiled_model(model_name: str):
    """Get compiled model with caching."""
    if model_name not in compiled_model_cache:
        model = load_model_optimized(model_name)
        # Compile with mode='reduce-overhead' for better performance on repeated calls
        compiled_model_cache[model_name] = torch.compile(
            model,
            mode='reduce-overhead',
            fullgraph=True,
            dynamic=False
        )
    return compiled_model_cache[model_name]

def load_tokenizer_optimized(model_name: str):
    """Load tokenizer with optimized settings."""
    if model_name in tokenizer_cache:
        return tokenizer_cache[model_name]

    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="/home/ec2-user/code/personal/hf_cache",
        use_fast=True,  # Use fast tokenizer implementation
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer_cache[model_name] = tokenizer
    return tokenizer

def decode_qwen_response(tokenizer, generated_token_ids: list, model_name: str) -> str:
    """Optimized Qwen response decoding."""
    if "qwen" not in model_name.lower():
        return tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

    think_tag_end_token_id = 151668
    try:
        # Use list comprehension and reverse search for better performance
        reversed_ids = generated_token_ids[::-1]
        last_think_end_idx = len(generated_token_ids) - 1 - reversed_ids.index(think_tag_end_token_id)
        content_token_ids = generated_token_ids[last_think_end_idx + 1:]
        return tokenizer.decode(content_token_ids, skip_special_tokens=True).strip()
    except ValueError:
        return tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

def parse_json_response(text: str) -> dict:
    """Optimized JSON parsing with multiple fallback strategies."""
    # First try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try markdown code block extraction
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding JSON object boundaries
    json_start = text.find('{')
    json_end = text.rfind('}')
    if json_start != -1 and json_end != -1 and json_start < json_end:
        try:
            return json.loads(text[json_start:json_end + 1])
        except json.JSONDecodeError:
            pass

    return {"error": "JSON decode error", "raw_response": text}

def get_hf_response(prompt_text, model_name=DEFAULT_MODEL_HF, temperature=0.0,
                   max_new_tokens=32768, format_type=None, use_compilation=True):
    """
    Optimized Hugging Face model inference with multiple performance improvements.

    Args:
        prompt_text (str): The prompt to send to the model.
        model_name (str): The Hugging Face model identifier.
        temperature (float): The temperature for generation.
        max_new_tokens (int): Maximum number of new tokens to generate.
        format_type (str, optional): 'json' if the output is expected to be JSON.
        use_compilation (bool): Whether to use torch.compile for optimization.

    Returns:
        str or dict: The model's response, or a dict if format_type is 'json'.
    """
    try:
        # Load model and tokenizer
        if use_compilation:
            model = get_compiled_model(model_name)
        else:
            model = load_model_optimized(model_name)

        tokenizer = load_tokenizer_optimized(model_name)
        device = next(model.parameters()).device

        # Use cached chat template processing
        final_prompt = get_chat_template(
            model_name,
            prompt_text,
            enable_thinking="qwen" in model_name.lower()
        )

        if final_prompt is None:
            final_prompt = prompt_text
            print(f"Warning: Using direct prompt for {model_name}")

        # Optimize tokenization
        max_length = getattr(tokenizer, 'model_max_length', 2048)
        inputs = tokenizer(
            final_prompt,
            return_tensors="pt",
            padding=False,  # Don't pad single inputs
            truncation=True,
            max_length=max_length
        ).to(device, non_blocking=True)

        # Optimized generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": True,  # Enable KV cache
            "do_sample": temperature > 0.0,
        }

        if temperature > 0.0:
            gen_kwargs.update({
                "temperature": temperature,
                "top_p": 0.9,  # Add nucleus sampling for better quality
            })

        # Use torch.inference_mode for better performance
        with torch.inference_mode():
            output_sequences = model.generate(**inputs, **gen_kwargs)

        # Extract and decode generated tokens
        generated_token_ids = output_sequences[0][inputs.input_ids.shape[-1]:].tolist()
        processed_text = decode_qwen_response(tokenizer, generated_token_ids, model_name)

        # Handle JSON formatting
        if format_type == 'json':
            return parse_json_response(processed_text)

        return processed_text

    except Exception as e:
        print(f"Error in model inference: {e}")
        if format_type == 'json':
            return {"error": str(e)}
        return f"Error: {e}"

# Utility function to warm up the model (optional but recommended)
def warmup_model(model_name=DEFAULT_MODEL_HF, warmup_prompt="Hello, world!"):
    """
    Warm up the model with a simple prompt to optimize subsequent calls.
    This helps with torch.compile optimization and GPU memory allocation.
    """
    print(f"Warming up model: {model_name}")
    get_hf_response(warmup_prompt, model_name, max_new_tokens=10)
    print("Model warmup complete")

# Memory cleanup utility
def clear_model_cache():
    """Clear model cache to free up memory."""
    global model_cache, tokenizer_cache, compiled_model_cache
    model_cache.clear()
    tokenizer_cache.clear()
    compiled_model_cache.clear()
    get_chat_template.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model cache cleared")
