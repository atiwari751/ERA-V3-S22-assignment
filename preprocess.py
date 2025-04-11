from datasets import load_dataset, Dataset
from collections import defaultdict

# Load dataset
raw_data = load_dataset("OpenAssistant/oasst1", split="train")

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
    
    # Find direct replies with ranking
    replies = [m for m in msgs if m["parent_id"] == prompt["message_id"] and m.get("ranking")]

    if len(replies) < 2:
        continue

    # Sort replies by rank
    ranked = sorted(replies, key=lambda x: x["ranking"])
    
    # Create one preference pair (you can create more pairs per prompt if you want)
    chosen = ranked[0]["text"]
    rejected = ranked[-1]["text"]

    pairs.append({
        "prompt": prompt["text"],
        "chosen": chosen,
        "rejected": rejected
    })

# Convert to Hugging Face dataset format
preference_dataset = Dataset.from_list(pairs)
preference_dataset.save_to_disk("oasst_preference_for_grpo")
