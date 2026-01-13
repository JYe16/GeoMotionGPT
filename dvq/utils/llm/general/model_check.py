import torch

def inspect_model_gradients(model, motion_ids_tensor):
    """
    Comprehensive check of model training status:
    1. Print parameters where requires_grad is True.
    2. Run a Dummy Backward to verify if gradients only flow to Motion Tokens.
    """
    print("\n" + "=" * 50)
    print("🔍 Model Freeze State Deep Inspection")
    print("=" * 50)

    # --- 1. Static Check: Which parameters are marked as trainable ---
    print("\n[1. Static Check] List of Parameters with requires_grad=True:")
    trainable_count = 0
    total_count = 0

    for name, param in model.named_parameters():
        total_count += param.numel()
        if param.requires_grad:
            print(f"  ✅ Trainable: {name:<60} | Shape: {list(param.shape)}")
            trainable_count += param.numel()

    ratio = trainable_count / total_count * 100
    print(f"\n  📊 Stats: {trainable_count:,} / {total_count:,} params are trainable ({ratio:.2f}%)")
    print("  ⚠️  Note: It is normal for Embeddings and LM Head to show 'Trainable'.")
    print("      Must confirm if Hook is effective via the dynamic check below.")

    # --- 2. Dynamic Check: Are gradients really filtered by Hook ---
    print("\n[2. Dynamic Check] Running Dummy Backward to verify Gradient Hooks...")

    device = model.device
    emb_layer = model.get_input_embeddings()
    lm_head = model.lm_head

    # Construct a dummy input: [Normal Token, Motion Token]
    # Find a normal Token ID that is definitely not Motion (e.g. 100)
    normal_token_id = 100
    while normal_token_id in motion_ids_tensor:
        normal_token_id += 1

    motion_token_id = motion_ids_tensor[0].item()  # Take the first Motion Token

    print(f"  🧪 Test Input: [Normal ID: {normal_token_id}] vs [Motion ID: {motion_token_id}]")

    dummy_input = torch.tensor([[motion_token_id, normal_token_id]], device=device)
    dummy_labels = dummy_input.clone()

    # Clear gradients
    model.zero_grad()

    # Forward + Backward
    outputs = model(input_ids=dummy_input, labels=dummy_labels)
    loss = outputs.loss
    loss.backward()

    # --- Verify Input Embeddings ---
    print(f"\n  🔎 Checking Input Embeddings (layer: {type(emb_layer).__name__}):")
    if emb_layer.weight.grad is None:
        print("  ❌ CRITICAL ERROR: Embedding layer has NO gradient at all! (Check if inputs require grad)")
    else:
        grad = emb_layer.weight.grad
        normal_grad_norm = grad[normal_token_id].norm().item()
        motion_grad_norm = grad[motion_token_id].norm().item()

        print(f"    - Normal Token Grad Norm: {normal_grad_norm:.6f} (Expecting 0.0)")
        print(f"    - Motion Token Grad Norm: {motion_grad_norm:.6f} (Expecting > 0.0)")

        if normal_grad_norm == 0.0 and motion_grad_norm > 0.0:
            print("    ✅ Embedding Hook is WORKING! (Text frozen, Motion training)")
        elif normal_grad_norm > 0.0:
            print("    ❌ FAILURE: Normal tokens are receiving gradients! Hook NOT working.")
        else:
            print("    ❓ WARNING: Motion tokens have 0 gradient. Maybe bad initialization or loss is 0?")

    # --- Verify LM Head (Output) ---
    if lm_head.weight is emb_layer.weight:
        print("\n  ℹ️  LM Head is TIED to Input Embeddings. (Skipping separate check)")
    else:
        print(f"\n  🔎 Checking LM Head (layer: {type(lm_head).__name__}):")
        if lm_head.weight.grad is None:
            print("  ❌ CRITICAL ERROR: LM Head has NO gradient!")
        else:
            grad = lm_head.weight.grad
            normal_grad_norm = grad[normal_token_id].norm().item()
            motion_grad_norm = grad[motion_token_id].norm().item()

            print(f"    - Normal Token Grad Norm: {normal_grad_norm:.6f} (Expecting 0.0)")
            print(f"    - Motion Token Grad Norm: {motion_grad_norm:.6f} (Expecting > 0.0)")

            if normal_grad_norm == 0.0 and motion_grad_norm > 0.0:
                print("    ✅ LM Head Hook is WORKING!")
            else:
                print("    ❌ FAILURE on LM Head.")

    # Clear test gradients to avoid affecting subsequent training
    model.zero_grad()
    print("\n" + "=" * 50 + "\n")