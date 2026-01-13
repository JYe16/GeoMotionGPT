import torch

def load_labels(label_path, tokenizer):
    """
    Return a list of label sequences, each shape (1, seq_len).
    We'll *append* these to the prompt in the training loop,
    and do manual shifting afterwards.
    """
    prompt = "Describe the action performed by the human skeleton with details in the video and select the exact action category from the following categories: "
    desc = []
    lines = open(label_path, "r").readlines()
    for i in range(0, len(lines), 2):
        # E.g.: lines[i]:   "Action: drinking water"
        #       lines[i+1]: "Description: The person picks up a bottle..."
        action_line = lines[i][:-1]
        desc_line = lines[i + 1].strip()  # e.g., "The person picks up a bottle..."
        full_text = desc_line + f" Thus, the exact action category is: ###{action_line}###</s>."

        tokenized = tokenizer(full_text, add_special_tokens=False)
        # shape => (seq_len,)
        label_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long).unsqueeze(0)
        # shape => (1, seq_len)
        desc.append(label_ids)

        prompt += action_line
        if i != len(lines) - 2:
            prompt += ', '
    prompt += '.'
    return desc, prompt
