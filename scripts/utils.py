# utils.py
import torch
from modal import method

class ModelFinetuner:
    @method()
    def finetune_model(
        self,
        model_name: str,
        dataset_path: str,
        output_dir: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        max_seq_length: int,
    ):
        from datasets import load_dataset
        from transformers import (
            AutoModelForCausalLM, 
            AutoTokenizer, 
            BitsAndBytesConfig,
            TrainingArguments
        )
        from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
        from trl import SFTTrainer
        
        print(f"Fine-tuning {model_name}")
        print("Loading dataset...")
        dataset = load_dataset('csv', data_files=dataset_path, split="train")
        
        print("Configuring quantization...")
        compute_dtype = getattr(torch, "float16")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
        
        print(f"Loading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map={"": 0},
            trust_remote_code=True
        )
        
        model = prepare_model_for_kbit_training(model)
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        print("Configuring LoRA...")
        if "llama" in model_name.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "phi" in model_name.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
        else:
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules
        )
        
        model = get_peft_model(model, lora_config)
        model.config.use_cache = False
        
        print("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            save_steps=100,
            logging_steps=10,
            learning_rate=learning_rate,
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
        )
        
        print("Initializing trainer...")
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=lora_config,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_args,
            packing=True
        )
        
        print("Starting training...")
        trainer.train()
        
        print("Saving model...")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        return f"{model_name} fine-tuning completed successfully!"