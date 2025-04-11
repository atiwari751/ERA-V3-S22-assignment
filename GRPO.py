from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import torch
import os
from collections import defaultdict

# Set environment variables for better logging
os.environ["WANDB_PROJECT"] = "phi2-grpo-finetuning"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the OpenAssistant dataset
raw_data = load_dataset("OpenAssistant/oasst1", split="train")

# Preprocess the dataset using logic from preprocess.py
# Group messages by conversation_id
conversations = defaultdict(list)
for item in raw_data:
    conversations[item["message_tree_id"]].append(item)

# Prepare preference pairs
pairs = []
for tree_id, msgs in conversations.items():
    prompt = next((m for m in msgs if m["role"] == "prompter" and m["parent_id"] is None), None)
    if not prompt:
        continue
    
    # Find direct replies to the prompt
    replies = [m for m in msgs if m["parent_id"] == prompt["message_id"]]
    
    # If we don't have ranking info or not enough replies, try to use other heuristics
    if len([r for r in replies if r.get("ranking")]) < 2:
        # If we have at least 2 replies, use them based on likes or other metrics
        if len(replies) >= 2:
            # Sort by likes if available, otherwise just take any two
            if all("like_count" in r for r in replies):
                ranked = sorted(replies, key=lambda x: x.get("like_count", 0), reverse=True)
            else:
                ranked = replies[:2]  # Just take the first two
            
            chosen = ranked[0]["text"]
            rejected = ranked[-1]["text"]
            
            pairs.append({
                "prompt": prompt["text"],
                "chosen": chosen,
                "rejected": rejected
            })
        continue
    
    # Original logic for replies with ranking
    ranked = sorted(replies, key=lambda x: x["ranking"])
    chosen = ranked[0]["text"]
    rejected = ranked[-1]["text"]

    pairs.append({
        "prompt": prompt["text"],
        "chosen": chosen,
        "rejected": rejected
    })

# Convert to Hugging Face dataset format for preference learning
preference_dataset = Dataset.from_list(pairs)

print(f"Created {len(preference_dataset)} preference pairs for GRPO")

# Debug: Print a sample pair if available
if len(preference_dataset) > 0:
    print("\nSample preference pair:")
    print(f"Prompt: {preference_dataset[0]['prompt'][:100]}...")
    print(f"Chosen: {preference_dataset[0]['chosen'][:100]}...")
    print(f"Rejected: {preference_dataset[0]['rejected'][:100]}...")
else:
    print("WARNING: No preference pairs were created. Check the dataset structure.")

# Load model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Define a reward function that rewards helpful, concise responses
# and penalizes responses similar to rejected ones
def reward_func(completions, **kwargs):
    return [len(c.split()) for c in completions]  # reward by word count

# Configure tokenizer for chat format
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Configure GRPO training
training_args = GRPOConfig(
    output_dir="phi2-grpo-openassistant",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-6,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    remove_unused_columns=False,
    report_to="wandb",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
)

# Initialize the GRPO trainer with preference dataset
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=preference_dataset,
    reward_funcs=reward_func,
)

# Set the tokenizer on the trainer after initialization
trainer.tokenizer = tokenizer

# Start training
trainer.train()

# Save the final model
trainer.save_model("phi2-grpo-openassistant-final")
