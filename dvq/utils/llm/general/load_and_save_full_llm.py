import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

def load_and_save_full_llm(base_model_path, lora_adapter_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    # 1. Load base model first
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # or float32
        device_map="auto"  # can calculate "auto" if VRAM is sufficient
    )

    # 2. Mount LoRA adapter
    model = PeftModel.from_pretrained(model, lora_adapter_path)

    # 3. Merge & unload LoRA layers
    merged_model = model.merge_and_unload()  # Critical step!
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, use_fast=True)

    # 4. Save full weights (sharded by 2 GB, safetensors format)
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="4GB"
    )
    tokenizer.save_pretrained(output_path)

    print("Done! Full parameter model has been saved to ", output_path)


def merge_lora_with_motion_tokens(
        base_model_path: str,
        lora_path: str,
        nb_code: int,
        output_path: str):
    """
    Do not modify files in base_model_path:
    1) Reload original model & tokenizer
    2) Temporarily expand vocab (pad + motion token)
    3) Merge LoRA -> Save to output_path
    """
    # 1️⃣ Base tokenizer / model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model     = AutoModelForCausalLM.from_pretrained(base_model_path)

    # 2️⃣ pad_token (if missing)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # 3️⃣ motion tokens
    motion_tokens = [f"<motion_{i:04d}>" for i in range(nb_code)]
    tokenizer.add_tokens(motion_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # 4️⃣ Load LoRA, merge
    peft_model = PeftModel.from_pretrained(model, lora_path)
    merged     = peft_model.merge_and_unload()   # -> Standard HF model

    # 5️⃣ Save to new directory
    merged.save_pretrained(output_path, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output_path)
    print(f"✅ Full model saved to {output_path}")

def merge_lora_with_motion_tokens_2cb(
        base_model_path: str,
        lora_path: str,
        nb_code: int,
        output_path: str):
    """
    Merge a LoRA checkpoint (trained with additional motion tokens)
    into the original base model and save a full-precision model.

    Steps:
        1) Try to load the tokenizer from the LoRA path (preferred).
        2) If that fails, load the base tokenizer and rebuild motion tokens.
        3) Resize the base model's token embeddings to match the tokenizer.
        4) Load the LoRA adapter and merge it into the base model.
        5) Save the merged full model and tokenizer to `output_path`.
    """
    # 1) Prefer the tokenizer that was saved together with the LoRA checkpoint.
    #    This tokenizer should already contain all motion tokens.
    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_path, use_fast=False)
        print(f"[merge] Loaded tokenizer from LoRA path: {lora_path}")
        loaded_from_lora = True
    except Exception as e:
        print(f"[merge] Failed to load tokenizer from LoRA path ({e}). "
              f"Falling back to base tokenizer from: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
        loaded_from_lora = False

    # 2) If tokenizer is NOT from LoRA, we need to rebuild the pad token
    #    and motion tokens exactly as in LoRA training.
    if not loaded_from_lora:
        # Ensure pad token exists by actually adding a new token.
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            print("[merge] Added pad_token '<|pad|>' to tokenizer.")

        # Rebuild motion tokens; make sure naming/format matches LoRA training.
        motion_tokens = [f"<motion_{i:04d}>" for i in range(nb_code)]
        num_added = tokenizer.add_tokens(motion_tokens)
        print(f"[merge] Added {num_added} motion tokens to tokenizer.")
    else:
        # When using LoRA's tokenizer, we MUST NOT change its vocab size,
        # otherwise it will no longer match the LoRA checkpoint.
        if tokenizer.pad_token_id is None:
            # Do NOT add a new token here; just reuse eos_token as pad if possible.
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print("[merge] Set pad_token to eos_token to avoid changing vocab size.")
            else:
                print("[merge][warning] Tokenizer has neither pad_token nor eos_token. "
                      "This may affect training/inference, but vocab size is kept intact.")

    # 3) Load the base model from its original path.
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    old_vocab_size = model.get_input_embeddings().weight.size(0)
    new_vocab_size = len(tokenizer)

    # 4) Resize model embeddings only if sizes do not match.
    if old_vocab_size != new_vocab_size:
        print(f"[merge] Resizing token embeddings: {old_vocab_size} -> {new_vocab_size}")
        model.resize_token_embeddings(new_vocab_size)
    else:
        print(f"[merge] Model vocab size already matches tokenizer: {new_vocab_size}")

    # 5) Load the LoRA adapter and merge it into the base model.
    peft_model = PeftModel.from_pretrained(model, lora_path)
    merged = peft_model.merge_and_unload()  # → plain HF model (no PEFT wrapper)

    # 6) Save the merged full model and tokenizer to a new directory.
    merged.save_pretrained(output_path, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output_path)
    print(f"[merge] ✅ Full merged model saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default="llama1b",)
    parser.add_argument('--model', type=str, default="0805-llama1b")
    args = parser.parse_args()


    base_model_path   = "../../../model/original/" + args.base_model  # ⚠️ 换成你自己的
    lora_adapter_path = "../../../model/lora/" + args.model + '-lora'          # 就是截图里的目录
    output_path       = "../../../model/" + args.model + "-lang"

    load_and_save_full_llm(base_model_path, lora_adapter_path, output_path)