import json

def load_alpaca_data(file_path):
    """
    Load Alpaca dataset from a JSON file.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def format_for_llama2(example):
    """
    Format a single example for LLaMA 2 fine-tuning.
    """
    if example['input']:
        return f"<s>[INST] {example['instruction']}\n\n{example['input']} [/INST] {example['output']}</s>"
    else:
        return f"<s>[INST] {example['instruction']} [/INST] {example['output']}</s>"

def format_for_phi(example):
    """
    Format a single example for Phi fine-tuning.
    """
    if example['input']:
        return f"Instruction: {example['instruction']}\nInput: {example['input']}\nResponse: {example['output']}"
    else:
        return f"Instruction: {example['instruction']}\nResponse: {example['output']}"

def convert_dataset(alpaca_data, output_llama2, output_phi):
    """
    Convert the Alpaca dataset to LLaMA 2 and Phi formats and save them.
    """
    llama2_data = [format_for_llama2(example) for example in alpaca_data]
    phi_data = [format_for_phi(example) for example in alpaca_data]

    # Save LLaMA 2 format
    with open(output_llama2, 'w') as f:
        for item in llama2_data:
            f.write(f"{item}\n")

    # Save Phi format
    with open(output_phi, 'w') as f:
        json.dump(phi_data, f, indent=2)

    print(f"Converted {len(alpaca_data)} examples.")
    print(f"LLaMA 2 format saved to: {output_llama2}")
    print(f"Phi format saved to: {output_phi}")