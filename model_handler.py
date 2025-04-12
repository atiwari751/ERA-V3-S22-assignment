import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer, ensuring CPU compatibility."""
    # Load tokenizer
    base_model_name = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create offload directory if it doesn't exist
    offload_dir = "offload_dir"
    os.makedirs(offload_dir, exist_ok=True)
    
    # Load base model with 8-bit quantization to reduce memory usage
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="auto",
        offload_folder=offload_dir,  # Add offload directory
        load_in_8bit=True,          # Use 8-bit quantization
        low_cpu_mem_usage=True      # Optimize for low memory
    )
    
    # Load adapter weights
    model = PeftModel.from_pretrained(
        base_model, 
        "phi2-grpo-qlora-final",
        device_map="auto",
        offload_folder=offload_dir  # Add offload directory
    )
    
    # Set to evaluation mode
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Generate a streaming response from the model for the given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Store the input length to identify the response part
    input_length = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
    
    # Initialize generation
    generated_ids = inputs.input_ids.clone()
    past_key_values = None
    response_text = ""
    
    # Add stop sequences to detect natural endings
    stop_sequences = ["\n\n", "\nExercise:", "\nQuestion:"]
    
    # Generate tokens one by one
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # Forward pass
            outputs = model(
                input_ids=generated_ids[:, -1:] if past_key_values is not None else generated_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # Get logits and past key values
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            
            # Sample next token
            next_token_logits = logits[:, -1, :]
            next_token_logits = next_token_logits / 0.7  # Apply temperature
            
            # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.9
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = -float('Inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated ids
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Decode the current token
            current_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Extract only the new part (response)
            if len(current_text) > input_length:
                new_text = current_text[input_length:]
                # Get only the new characters
                new_chars = new_text[len(response_text):]
                response_text = new_text
                
                # Yield the new characters for streaming
                yield new_chars
            
            # Stop if we generate an EOS token
            if next_token[0, 0].item() == tokenizer.eos_token_id:
                break
                
            # Check for natural stopping points
            if any(stop_seq in response_text for stop_seq in stop_sequences):
                # If we find a stop sequence, only keep text up to that point
                for stop_seq in stop_sequences:
                    if stop_seq in response_text:
                        stop_idx = response_text.find(stop_seq)
                        if stop_idx > 0:  # Only trim if we have some content
                            yield response_text[len(response_text)-1:stop_idx]  # Yield any remaining text
                            return
