import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

vllm_image = vllm_image.env({"VLLM_USE_V1": "1"})

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_PATH = "/model/lora_finetuned_meta-llama_Meta-Llama-3-8B-Instruct"


vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

model_volume = modal.Volume.from_name("model_volume")

app = modal.App("vllm-app")

N_GPU = 1  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count
API_KEY = "super-secret-key"  # api key, for auth. for production use, replace with a modal.Secret

MINUTES = 60  # seconds

VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"A10G:{N_GPU}",
    # how many requests can one replica handle? tune carefully!
    allow_concurrent_inputs=4,
    # how long should we stay up with no requests?
    scaledown_window=5 * MINUTES,
    volumes={
        "/root/.cache/vllm": vllm_cache_vol,
        "/model": model_volume,
    },
    keep_warm=1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES)
def serve():
    import subprocess
    import os

    os.environ["HUGGINGFACE_HUB_CACHE"] = "/model/hf_cache"

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--enable-lora",
        f"--lora-modules llama-3-8b-instruct-lora={LORA_PATH}",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        API_KEY,
    ]

    subprocess.Popen(" ".join(cmd), shell=True)

