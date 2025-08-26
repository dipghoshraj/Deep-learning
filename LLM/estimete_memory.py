def estimate_fp16_memory(P, B, S, D, L, overhead_MB=500, use_checkpointing=False):
    weights_MB = P * 2 / 1e6
    adam_MB = P * 8 / 1e6
    grad_MB = P * 2 / 1e6
    activation_factor = 2.5 if use_checkpointing else 4
    activations_MB = B * S * D * L * activation_factor * 2 / 1e6
    total_MB = weights_MB + adam_MB + grad_MB + activations_MB + overhead_MB
    return round(total_MB, 2)

estimate_fp16_memory(P=88000000, B=4, S=512, D=512, L=12, use_checkpointing=False)