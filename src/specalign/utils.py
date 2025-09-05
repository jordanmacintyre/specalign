import torch


def onpolicy_generation(model, input_ids, k):
    # Generate exactly up to k_max new tokens (may stop early on EOS)
    model_status = model.training  # Is the model currently in training mode
    model.eval()

    # Generate on-policy data
    with torch.no_grad():
        generated_tokens = model.generate(
            input_ids=input_ids,
            max_new_tokens=k,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=False,
            output_scores=False,
        )

    # Return model to training mode if applicable
    if model_status:
        model.train()

    return generated_tokens.sequences
