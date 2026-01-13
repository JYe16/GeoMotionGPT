import torch

def inspect_model_gradients(model, motion_ids_tensor):
    """
    全面检查模型的训练状态：
    1. 打印 requires_grad 为 True 的参数。
    2. 运行一次 Dummy Backward，验证梯度是否真的只流向了 Motion Token。
    """
    print("\n" + "=" * 50)
    print("🔍 模型解冻状态深度检查 (Deep Inspection)")
    print("=" * 50)

    # --- 1. 静态检查: 哪些参数被标记为可训练 ---
    print("\n[1. Static Check] List of Parameters with requires_grad=True:")
    trainable_count = 0
    total_count = 0

    for name, param in model.named_parameters():
        total_count += param.numel()
        if param.requires_grad:
            print(f"  ✅ Trainable: {name:<60} | Shape: {list(param.shape)}")
            trainable_count += param.numel()

    ratio = trainable_count / total_count * 100
    print(f"\n  📊 统计: {trainable_count:,} / {total_count:,} params are trainable ({ratio:.2f}%)")
    print("  ⚠️  注意：对于 Embeddings 和 LM Head，显示 'Trainable' 是正常的。")
    print("      必须通过下面的动态检查来确认 Hook 是否生效。")

    # --- 2. 动态检查: 梯度是否真的被 Hook 过滤了 ---
    print("\n[2. Dynamic Check] Running Dummy Backward to verify Gradient Hooks...")

    device = model.device
    emb_layer = model.get_input_embeddings()
    lm_head = model.lm_head

    # 构造一个假输入：[普通Token, MotionToken]
    # 找一个肯定不是 Motion 的普通 Token ID (比如 100)
    normal_token_id = 100
    while normal_token_id in motion_ids_tensor:
        normal_token_id += 1

    motion_token_id = motion_ids_tensor[0].item()  # 取第一个动作 Token

    print(f"  🧪 Test Input: [Normal ID: {normal_token_id}] vs [Motion ID: {motion_token_id}]")

    dummy_input = torch.tensor([[motion_token_id, normal_token_id]], device=device)
    dummy_labels = dummy_input.clone()

    # 清空梯度
    model.zero_grad()

    # 前向 + 反向
    outputs = model(input_ids=dummy_input, labels=dummy_labels)
    loss = outputs.loss
    loss.backward()

    # --- 验证 Input Embeddings ---
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

    # --- 验证 LM Head (Output) ---
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

    # 清理用于测试的梯度，以免影响后续训练
    model.zero_grad()
    print("\n" + "=" * 50 + "\n")