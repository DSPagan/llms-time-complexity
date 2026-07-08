def load_model(
    model_path: str = "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length: int = 2048
):
    """
    Load a quantized pretrained model and tokenizer using Unsloth.

    Args:
        model_path (str): Path to the model directory or model name from Hugging Face Hub.
        max_seq_length (int): Maximum input sequence length for the model.

    Returns:
        model: The loaded model instance.
        tokenizer: The corresponding tokenizer with the chat template applied.
    """

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length, # maximum sequence length for the model
        load_in_4bit = True, # use 4-bit quantization for memory efficiency
        dtype = None,
    )

    # Apply the correct chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )

    return model, tokenizer