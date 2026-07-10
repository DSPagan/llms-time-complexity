from .prompts import build_prompt


def train_model(
    train_data_path: str,
    model,
    tokenizer,
    output_dir: str = "outputs",
    num_epochs: int = 2,
    lora_r: int = 16,
    max_seq_length: int = 2048,
):
    """
    Fine-tune a language model using QLoRA on a dataset of code snippets with time complexity annotations.

    Args:
        train_data_path (str): Path to the training .jsonl file.
        model: The base model to fine-tune.
        tokenizer: The tokenizer corresponding to the model.
        output_dir (str): Directory where the fine-tuned model will be saved.
        num_epochs (int): Number of training epochs.
        lora_r (int): Rank of the LoRA adaptation layers.
        max_seq_length (int): Maximum input sequence length.

    Returns:
        (model, trainer_stats): the fine-tuned model and the training statistics.
        The model is returned because QLoRA wraps it here, so the caller must use
        this object (not the original) for inference.
    """

    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    import json

    # Load the training data
    with open(train_data_path, "r") as f:
        train_data = [json.loads(line.strip()) for line in f]

    # Build a chat dataset: one user turn (the prompt) + one assistant turn (the label)
    rows = [
        {"conversations": [
            {"role": "user", "content": build_prompt(item["src"])},
            {"role": "assistant", "content": item["complexity"]},
        ]}
        for item in train_data
    ]

    def formatting_prompts_func(examples):
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    dataset = Dataset.from_list(rows).map(formatting_prompts_func, batched=True)

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        args = SFTConfig(
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = num_epochs,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,
            report_to = "none",
        ),
    )

    # Train only on the assistant's response tokens
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    trainer_stats = trainer.train()
    return model, trainer_stats
