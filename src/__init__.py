"""Reusable code for the time-complexity LLM experiments.

Modules
-------
prepare_data   : de-duplicate CodeComplex and split it into stratified cross-validation folds.
load_model     : load Llama 3.1 8B Instruct (4-bit) via Unsloth.
prompts        : the prompt shared by fine-tuning and fine-tuned inference.
train_model    : QLoRA fine-tuning.
evaluate       : map raw model outputs to classes and compute accuracy / F1 / confusion matrix.
"""
