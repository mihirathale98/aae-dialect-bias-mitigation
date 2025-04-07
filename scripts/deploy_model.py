import modal
import os
from typing import List, Dict, Any

# Define constants
MODEL_ID = "unsloth/llama-3-8b-Instruct-bnb-4bit"  # Using Unsloth's 4-bit quantized model
LORA_ADAPTER_ID = "path/to/your/lora/adapter"  # Replace with your LoRA adapter path or HF repo
API_KEY = "your-api-key"  # You can set this as a Modal secret in production
VLLM_PORT = 8000

# Create a volume to cache models
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Create the base image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.7.2",  # Fast inference engine
        "unsloth[llama]==0.6.0",  # Unsloth for optimizations
        "huggingface_hub[hf_transfer]==0.26.2",  # For model downloading
        "transformers>=4.43.0",  # For model loading
        "peft>=0.9.0",  # For LoRA adapters
        "accelerate",  # For inference optimizations
        "bitsandbytes>=0.42.0",  # For quantization
        "torch>=2.3.0",  # PyTorch
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # Faster model transfers
)

# Create the Modal app
app = modal.App("llama3-lora-inference", image=image)

# Function to load the model and LoRA adapter
@app.function(
    gpu="A10G",  # You can change to H100 for better performance
    timeout=600,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
def load_model_with_lora():
    from unsloth import FastLanguageModel
    import torch
    from peft import PeftModel

    # Load the base model
    print("Loading the base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=4096,  # Set context length
        load_in_4bit=True,    # Use 4-bit quantization to reduce memory
        precision="bfloat16"  # Use bfloat16 precision for faster inference
    )
    
    print("Optimizing model for inference...")
    # Optimize for inference
    model = FastLanguageModel.for_inference(model)
    
    print("Loading the LoRA adapter...")
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(
        model,
        LORA_ADAPTER_ID,
    )
    
    print("LoRA adapter loaded successfully")
    return model, tokenizer

# Deploy a vLLM server with the merged model for high-performance inference

@app.function(
    gpu="A10G",  # You can change to H100 for better performance
    timeout=600,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    concurrency_limit=4,  # Limit concurrent requests for stability,
    keep_warm=True,
)
def run_inference(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    from unsloth import FastLanguageModel
    import torch
    from peft import PeftModel
    
    # Load model with LoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=4096,
        load_in_4bit=True,
        precision="bfloat16"
    )
    
    # Optimize for inference
    model = FastLanguageModel.for_inference(model)
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(
        model,
        LORA_ADAPTER_ID,
    )
    
    # Format the input for Llama 3
    messages = [{"role": "user", "content": prompt}]
    prompt_formatted = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate response
    input_ids = tokenizer(prompt_formatted, return_tensors="pt").input_ids.to("cuda")
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=temperature > 0,
        )
    
    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response.replace(prompt_formatted, "").strip()
    
    return response

# Deploy a web server
@app.function(
    gpu="A10G",  # You can change to H100 for better performance
    timeout=600,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    concurrency_limit=4,
)
@modal.web_server()
def serve():
    from fastapi import FastAPI, HTTPException, Depends, status
    from fastapi.security import APIKeyHeader
    from pydantic import BaseModel
    import time
    import uvicorn
    
    class CompletionRequest(BaseModel):
        prompt: str
        max_tokens: int = 512
        temperature: float = 0.7
        
    class CompletionResponse(BaseModel):
        completion: str
        time_taken: float
    
    app = FastAPI(title="Llama 3 with LoRA Inference API")
    api_key_header = APIKeyHeader(name="X-API-Key")
    
    # Simple API key validation
    def get_api_key(api_key: str = Depends(api_key_header)):
        if api_key != API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key",
            )
        return api_key
    
    @app.post("/complete", response_model=CompletionResponse)
    async def complete(request: CompletionRequest, api_key: str = Depends(get_api_key)):
        start_time = time.time()
        
        try:
            # Call the inference function
            completion = run_inference.remote(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            
            time_taken = time.time() - start_time
            return CompletionResponse(completion=completion, time_taken=time_taken)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference error: {str(e)}",
            )
    
    @app.get("/health")
    async def health_check():
        return {"status": "ok"}
    
    return app

# For local testing
@app.local_entrypoint()
def main():
    # Test inference
    test_prompt = "Explain the benefits of LoRA fine-tuning for LLMs"
    print(f"Running inference with prompt: {test_prompt}")
    
    # Call the remote function
    result = run_inference.remote(test_prompt)
    print(f"Response: {result}")