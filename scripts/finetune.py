import modal


# Define a Modal app for our finetuning job
app = modal.App("finetune-llama31-lora")

# Create an image with the required dependencies
image = modal.Image.debian_slim().pip_install([
    "transformers>=4.43.2",  # Required for Llama 3.1
    "peft", 
    "torch>=2.0.0", 
    "accelerate", 
    "bitsandbytes",  # For quantization
    "sentencepiece",  # Required for tokenization
    "datasets"        # For dataset handling
])

# Create a secret for Hugging Face token
huggingface_token = modal.Secret.from_name("huggingface-secret")

@app.function(
    image=image,
    gpu="A100-80GB:1",  # Request an A100 GPU with 80GB memory
    timeout=3600,     # Set timeout to 1 hour
    secrets=[huggingface_token]
)
def finetune():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import get_peft_model, LoraConfig, TaskType
    from datasets import Dataset
    # Model and tokenizer setup for Llama 3.1
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
    )
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
        r=16,                  # Rank
        lora_alpha=32,         # Alpha parameter
        lora_dropout=0.1,      # Dropout probability
        bias="none",           # Bias configuration
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target specific modules
        inference_mode=False,
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # Print trainable parameters info
    
    # Prepare training data
    instruction_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{response}<|eot_id|>"
    
    training_samples = [
        {"instruction": "Explain quantum computing in simple terms.", 
         "response": "Quantum computing uses quantum bits or qubits that can exist in multiple states simultaneously, unlike classical bits that are either 0 or 1. This allows quantum computers to process certain types of problems much faster than classical computers."},
        {"instruction": "Write a short poem about artificial intelligence.", 
         "response": "Silicon dreams in neural webs,\nLearning patterns, making guesses.\nMimicking thought with math and code,\nAI walks a human road."}
    ]
    
    # Format the training data
    formatted_data = []
    for sample in training_samples:
        formatted_text = instruction_template.format(
            instruction=sample["instruction"],
            response=sample["response"]
        )
        formatted_data.append({"text": formatted_text})
    
    # Create a proper dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")
    
    # Create a data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're not doing masked language modeling
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./lora_finetuned_llama31",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=5,
        save_steps=50,
        report_to="none",
        save_total_limit=2,
        remove_unused_columns=False,  # Important for custom datasets
    )
    
    # Create the Trainer and start finetuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save the finetuned LoRA adapters
    model.save_pretrained("./lora_finetuned_llama31")
    tokenizer.save_pretrained("./lora_finetuned_llama31")
    
    return "Finetuning completed successfully!"

@app.local_entrypoint()
def main():
    # Run the finetuning job on Modal
    finetune.remote()
