import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import time
import sys

def stream_response(model, tokenizer, prompt, max_new_tokens=256):
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
    was_truncated = False  # Track if response was truncated
    
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
                # Print only the new characters
                new_chars = new_text[len(response_text):]
                sys.stdout.write(new_chars)
                sys.stdout.flush()
                response_text = new_text
                
                # Add a small delay to simulate typing
                time.sleep(0.01)
            
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
                            was_truncated = True
                            response_text = response_text[:stop_idx]
                            sys.stdout.write("\n")  # Add a newline for cleaner output
                            sys.stdout.flush()
                            return response_text, was_truncated
    
    # Return the full response
    return response_text, was_truncated

def main():
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned Phi-2 models")
    parser.add_argument("--base-only", action="store_true", help="Use only the base model")
    parser.add_argument("--finetuned-only", action="store_true", help="Use only the fine-tuned model")
    parser.add_argument("--adapter-path", type=str, default="./phi2-grpo-qlora-final", 
                        help="Path to the fine-tuned adapter")
    args = parser.parse_args()
    
    # Load the base model and tokenizer
    base_model_name = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Configure tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load models based on arguments
    models = {}
    
    if not args.finetuned_only:
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        models["Base Phi-2"] = base_model
    
    if not args.base_only:
        print(f"Loading fine-tuned model from {args.adapter_path}...")
        # Load the base model first (with same quantization as during training)
        base_model_for_ft = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        # Load the adapter on top of the base model
        finetuned_model = PeftModel.from_pretrained(base_model_for_ft, args.adapter_path)
        models["Fine-tuned Phi-2"] = finetuned_model
    
    # Interactive prompt loop
    print("\n" + "="*50)
    print("Interactive Phi-2 Model Comparison (Streaming Mode)")
    print("Type 'exit' to quit")
    print("="*50 + "\n")
    
    while True:
        # Get user input
        user_prompt = input("\nEnter your prompt: ")
        if user_prompt.lower() == 'exit':
            break
        
        print("\n" + "-"*50)
        
        # Generate responses from each model
        for model_name, model in models.items():
            print(f"\n{model_name} response:")
            response, was_truncated = stream_response(model, tokenizer, user_prompt)
            if was_truncated:
                print("\n[Note: Response was truncated at a natural stopping point]")
            print("\n" + "-"*30)
        
        print("-"*50)

if __name__ == "__main__":
    main()
