"""The prompt used for QLoRA fine-tuning and for fine-tuned inference.

Both paths must send the model the *same* prompt, so it lives here.

Unlike the in-context (zero-/few-shot) prompts, the fine-tuned model is treated as
a clean 7-way classifier: the prompt lists the exact class labels and asks for the
label *only*, with no extra commentary. Those labels match the training targets in
data/*.jsonl, so the prompt and the target speak the same vocabulary.
"""

CLASSES = "constant, logn, linear, nlogn, quadratic, cubic, exponential"


def build_prompt(src):
    return (
        "Analyze the worst-case time complexity of the following code.\n"
        f"Answer with exactly one of these labels and nothing else: {CLASSES}.\n\n"
        "Code:\n"
        f"{src}"
    )
