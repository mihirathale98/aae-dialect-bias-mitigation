import modal
import os

# Define constants
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_ADAPTER_ID = "/model/lora_finetuned_meta-llama_Meta-Llama-3-8B-Instruct"
API_KEY = modal.Secret.from_name("modal_api_key")

# Create volumes for caching
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
model_volume = modal.Volume.from_name("model_volume")

# Create the base image with necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "unsloth",        # For optimizations
        "transformers",   # For model loading
        "peft",           # For LoRA adapters
        "accelerate",     # For inference optimizations
        "bitsandbytes",   # For quantization
        "torch",          # PyTorch
        "fastapi",        # For API
        "uvicorn",        # For API
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # Faster model transfers
)

# Create the Modal app
app = modal.App("llama3-lora-inference-simplified", image=image)

# Define a Modal class to keep the model loaded and warm
@app.cls(
    gpu="A10G",  # GPU required for model loading and inference
    timeout=3600,  # Longer timeout to keep the container alive
    volumes={
        "/model": model_volume,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    concurrency_limit=4,  # Limit concurrent requests
    keep_warm=1,  # Keep one instance always running
)
class LlamaWithLoRA:
    def __init__(self):
        from unsloth import FastLanguageModel
        import torch
        from peft import PeftModel
        
        print("Initializing model...")
        # Set environment variable for caching
        os.environ["HUGGINGFACE_HUB_CACHE"] = "/model/hf_cache"
        
        # Load model with LoRA
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=4096,
            load_in_4bit=True,  # Use 4-bit quantization to reduce memory
        )
        
        # Optimize for inference
        self.model = FastLanguageModel.for_inference(self.model)
        
        # Load the LoRA adapter
        self.model = PeftModel.from_pretrained(
            self.model,
            LORA_ADAPTER_ID,
        )
        print("Model initialized and ready!")
        
    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Run inference with the already-loaded model"""
        import torch
        
        # Format the input for Llama 3
        messages = [{"role": "user", "content": prompt}]
        prompt_formatted = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate response
        input_ids = self.tokenizer(prompt_formatted, return_tensors="pt").input_ids.to("cuda")
        
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=temperature > 0,
            )
        
        # Decode the output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response.replace(prompt_formatted, "").strip()
        
        return response

@app.function(
    # No GPU needed for API server
    timeout=3600,  # Extended timeout for long-running server
    volumes={
        "/model": model_volume,  # Still need volumes for paths
    },
    keep_warm=1,  # Keep server always running
)
@modal.web_server(port=8000)
def serve():
    """Serve the model via a FastAPI endpoint"""
    from fastapi import FastAPI, HTTPException, Depends, status
    from fastapi.security import APIKeyHeader
    from pydantic import BaseModel
    import time
    
    # Initialize our persistent model instance
    model = LlamaWithLoRA()
    
    class CompletionRequest(BaseModel):
        prompt: str
        max_tokens: int = 1024
        temperature: float = 0.0
        
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
            # Call the method on our persistent model instance
            completion = model.generate.remote(
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
        return {"status": "ok", "model_loaded": True}
    
    return app

# For local testing
@app.local_entrypoint()
def main():
    # Test inference using the persistent class
    model = LlamaWithLoRA()
    
    # Test multiple inferences in a loop to demonstrate persistence
    test_prompts = [
        "Break down why LoRA fine-tunin' good for them big language models.",
        "Explain the advantage of 4-bit quantization for inference.",
        "What are the main benefits of Modal for ML model deployment?"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nRunning inference #{i+1} with prompt: {prompt}")
        
        # Call the remote method
        result = model.generate.remote(prompt)
        print(f"Response: {result}")
        
    print("\nAll inferences completed successfully!")