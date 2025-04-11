from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import torch
import os

# Set environment variables for better logging
os.environ["WANDB_PROJECT"] = "phi2-grpo-finetuning"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the OpenAssistant dataset
dataset = load_dataset("trl-internal-testing/oasst_preference_dataset", split="train")

# Load model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Configure tokenizer for chat format
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Process the dataset to create prompt-response pairs
def preprocess_function(examples):
    # For OpenAssistant, we need to format the conversations properly
    # This is a simplified version - you may need to adjust based on the exact structure
    prompts = []
    responses = []
    
    for message in examples["messages"]:
        if len(message) >= 2:  # Ensure there's at least a prompt and response
            prompt = message[0]["content"]
            response = message[1]["content"]
            prompts.append(prompt)
            responses.append(response)
    
    return {"prompt": prompts, "response": responses}

# Process the dataset
processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# Define a reward function that rewards helpful, concise responses
def reward_function(responses, prompts=None, **kwargs):
    rewards = []
    for response in responses:
        # Example reward criteria:
        # 1. Length-based component (prefer responses between 100-500 chars)
        length_score = min(1.0, max(0.0, 1.0 - abs(len(response) - 300) / 300))
        
        # 2. Quality heuristics (simple examples)
        has_structure = 0.5 if any(marker in response for marker in ["First", "Second", "Finally", "In conclusion"]) else 0.0
        is_detailed = 0.5 if len(response) > 200 else 0.0
        
        # Combine reward components
        reward = length_score + has_structure + is_detailed
        rewards.append(reward)
    
    return rewards

# Configure GRPO training
training_args = GRPOConfig(
    output_dir="phi2-grpo-openassistant",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-6,
    max_length=512,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    evaluation_strategy="steps",
    fp16=True,
    remove_unused_columns=False,
    report_to="wandb",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
)

# Initialize the GRPO trainer
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=processed_dataset,
    reward_funcs=reward_function,
    packing=False,
)

# Start training
trainer.train()

# Save the final model
trainer.save_model("phi2-grpo-openassistant-final")
