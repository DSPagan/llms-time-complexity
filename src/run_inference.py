from .prompts import build_prompt


def run_inference(
    input_data: str,
    model,
    tokenizer,
    save_path: str = None,
    max_seq_length: int = 2048,
    max_new_tokens: int = 1024,
):   
    """
    Run inference on a list of code snippets to estimate their time complexity using a pretrained model.

    Args:
        input_data (str): Path to the input file in .txt or .py format, or code to analyze.
        model: The language model to use for inference.
        tokenizer: The tokenizer corresponding to the model.
        save_path (str): Path to save the results in .jsonl format, by default no file is saved.
        max_seq_length (int): Maximum length of the input tokens.
        max_new_tokens (int): Maximum number of tokens to generate for the output.

    Saves:
        A .jsonl file with one entry per code snippet, including:
            - "src": the code snippet
            - "complexity": the true complexity label
            - "model": the model's prediction
    """

    import json
    from pathlib import Path

    def gen_pred(prompt):
        messages = [
            {"role": "user", "content": prompt}
            ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        if len(inputs[0]) > max_seq_length:
            return None

        tokens = model.generate(input_ids=inputs,
                                do_sample=False, # deterministic generation
                                max_new_tokens=max_new_tokens, # maximum number of tokens to generate
                                use_cache=True, # use cache for faster generation
                                no_repeat_ngram_size=4) # to avoid repetition

        result = tokenizer.decode(tokens[0], skip_special_tokens=True)
        result = result.split("assistant")[-1].strip()  # keep only the assistant's response

        return result
        

    if isinstance(input_data, str) and Path(input_data).suffix in [".py", ".txt"]:
        results = []

        with open(input_data, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                code = line.strip()

                # Change the prompt to match the training data format
                prompt = build_prompt(code)

                result = gen_pred(prompt)

                if result is None:
                    print(f"Skipping code snippet {i+1} due to length constraints.")
                    continue

                entry = {
                    "src": code,
                    "model": result,
                }
                
                results.append(entry)

        if save_path is not None:
            with open(save_path, 'w') as out_file:
                for entry in results:
                    json.dump(entry, out_file)
                    out_file.write('\n')

        return results
            

    elif isinstance(input_data, str):
        # If input_data is not a file, assume it's a code snippet
        prompt = build_prompt(input_data.strip())

        result = gen_pred(prompt)

        if result is None:
            print("The input code snippet is too long for the model to process.")
            return None

        return result
    

    else:
        raise ValueError("Unsupported input type.")
