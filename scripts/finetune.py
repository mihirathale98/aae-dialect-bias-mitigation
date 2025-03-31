import modal
import json

# Define a Modal app for our finetuning job
app = modal.App("finetune-model")

# Create an image with the required dependencies
image = modal.Image.debian_slim().pip_install([
    "transformers>=4.43.2",  # Required for Llama 3.1 and Phi-2
    "peft", 
    "torch>=2.0.0", 
    "accelerate", 
    "bitsandbytes",  # For quantization
    "sentencepiece",  # Required for tokenization
    "datasets"        # For dataset handling
])

# Create a secret for Hugging Face token
huggingface_token = modal.Secret.from_name("huggingface-secret")
model_volume = modal.Volume.from_name("model_volume", create_if_missing=True)
data_volume = modal.Volume.from_name("data_volume", create_if_missing=True)

# with data_volume.batch_upload() as batch:
    
#     batch.put_file("../data/aae_llama.json", "/data/aae_llama.json")
#     batch.put_file("../data/aae_phi.json", "/data/aae_phi.json")

@app.function(
    image=image,
    gpu="A100-80GB:1",  # Request an A100 GPU with 80GB memory
    timeout=3600,     # Set timeout to 1 hour
    secrets=[huggingface_token],
    volumes={"/model": model_volume, "/data": data_volume},
)
def finetune():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import get_peft_model, LoraConfig, TaskType
    from datasets import load_dataset
    
    # Load dataset from data volume
    dataset_path = "/data/data/aae_llama.json"
    dataset = load_dataset("json", data_files=dataset_path)
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True  # Use 8-bit quantization to save memory
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
    

    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch")
    
    # Create a data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"/model/lora_finetuned_{model_name.replace('/', '_')}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=5,
        save_steps=50,
        report_to="none",
        save_total_limit=2,
        remove_unused_columns=False,
    )
    
    # Create the Trainer and start finetuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
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