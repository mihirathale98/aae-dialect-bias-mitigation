import modal


cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
image = (
    image
    .apt_install("git")
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "torch==2.5.1",
        "transformers==4.48.3",
    )
)
image = image.run_commands(
    "CXX=g++ pip install flash-attn --no-build-isolation"
)
app_image = image.run_commands(
    "pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton",
    "pip install --no-deps cut_cross_entropy unsloth_zoo",
    "pip install sentencepiece protobuf datasets huggingface_hub hf_transfer",
).pip_install(
    "psutil",
    "Pillow",
    'gguf',
    'protobuf',
).apt_install(
    'cmake',
    'libcurl4-openssl-dev',
)


with app_image.imports():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import get_peft_model, LoraConfig, TaskType
    from datasets import load_dataset, Dataset
    import json
    from torch.utils.data import DataLoader
    import tqdm
    import os

app = modal.App("finetune-model", image=app_image)

# Create a secret for Hugging Face token
huggingface_token = modal.Secret.from_name("huggingface-secret")
model_volume = modal.Volume.from_name("model_volume", create_if_missing=True)
data_volume = modal.Volume.from_name("data_volume", create_if_missing=True)

## Upload data
# with data_volume.batch_upload() as batch:
#     batch.put_file("../data/aae_llama.json", "/data/aae_llama.json")
#     batch.put_file("../data/aae_phi.json", "/data/aae_phi.json")

@app.function(
    cpu=(8.0, 8.0),
    memory=32768,
    gpu="A100-80GB:1",  # Request an A100 GPU with 40GB memory
    timeout=24 * 60 * 60,     # Set timeout to 24 hours
    secrets=[huggingface_token],
    volumes={"/model": model_volume, "/data": data_volume},

)
def finetune():

    os.environ["HUGGINGFACE_HUB_CACHE"] = "/model/hf_cache"

    def load_llama2_dataset(file_path):
        """Load LLaMA 2 formatted dataset from a text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            examples = [line.strip() for line in f if line.strip()]
        return Dataset.from_dict({"text": examples})

    def load_phi_dataset(file_path):
        """Load Phi formatted dataset from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        return Dataset.from_dict({"text": examples})

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Load dataset
    if 'llama' in model_name:
        dataset = load_llama2_dataset("/data/data/aae_llama.json")
    elif 'phi' in model_name:
        dataset = load_phi_dataset("/data/data/aae_phi.json")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"])
    tokenizer.pad_token = tokenizer.eos_token

    # Split dataset into train/test
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Tokenize in batches
    def tokenize_in_batches(examples, batch_size=1000):
        """Efficiently tokenize dataset in batches."""
        tokenized_data = {"input_ids": [], "attention_mask": []}
        
        # Set a safe max_length - using 4096 as a reasonably safe value
        # This is the key fix to avoid the OverflowError
        max_length = min(4096, tokenizer.model_max_length)
        
        for i in tqdm.tqdm(range(0, len(examples["text"]), batch_size)):
            batch_texts = examples["text"][i:i + batch_size]
            tokenized_output = tokenizer(
                batch_texts,
                padding="longest",  # Use longest padding to reduce padding size
                truncation=True,
                max_length=max_length,  # Use our safe max_length
                return_tensors="pt",
            )
            tokenized_data["input_ids"].extend(tokenized_output["input_ids"].tolist())
            tokenized_data["attention_mask"].extend(tokenized_output["attention_mask"].tolist())
        return tokenized_data

    # Apply tokenization in batches to the train/test splits
    train_data = tokenize_in_batches(dataset["train"])
    test_data = tokenize_in_batches(dataset["test"])

    # Convert tokenized data into Dataset format
    tokenized_train_dataset = Dataset.from_dict(train_data)
    tokenized_test_dataset = Dataset.from_dict(test_data)

    # Create a data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )


    # Load model with quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=False,  # Using full precision since we have 40GB GPU
        cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"]
    )

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        inference_mode=False,
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"/model/lora_finetuned_{model_name.replace('/', '_')}",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=1e-5,
        bf16=True,
        logging_steps=5,
        save_steps=100,
        report_to="none",
        save_total_limit=4,
        remove_unused_columns=False,
    )

    # Create the Trainer and start finetuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

    # Save the finetuned LoRA adapters
    model.save_pretrained(f"/model/lora_finetuned_{model_name.replace('/', '_')}")
    tokenizer.save_pretrained(f"/model/lora_finetuned_{model_name.replace('/', '_')}")

    return "Finetuning completed successfully!"


@app.local_entrypoint()
def main():
    # Run the finetuning job on Modal
    finetune.remote()