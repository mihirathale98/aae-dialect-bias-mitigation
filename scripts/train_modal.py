import argparse
import modal
from modal import Stub, Image, gpu, method

stub = Stub("llm-finetuning")

image = (
    Image.debian_slim()
    .pip_install([
        "torch>=2.0.0", 
        "transformers>=4.30.0", 
        "datasets>=2.12.0", 
        "peft>=0.4.0", 
        "accelerate>=0.20.0", 
        "bitsandbytes>=0.40.0", 
        "trl>=0.4.7",
        "wandb"
    ])
)

gpu_config = gpu.A100(count=1, memory=80)

from utils import ModelFinetuner

@stub.local_entrypoint()
def main(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_seq_length: int = 2048,
):
    finetuner = ModelFinetuner()
    
    result = finetuner.finetune_model.remote(
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        max_seq_length=max_seq_length,
    )
    
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLM models using LoRA on Modal")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to fine-tune (e.g., 'meta-llama/Llama-2-7b-chat-hf' or 'microsoft/phi-2')")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for training")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    
    args = parser.parse_args()
    main(**vars(args))
