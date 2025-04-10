import os
import streamlit as st
from openai import AsyncOpenAI
import asyncio
import time
from typing import AsyncGenerator, Generator

openai_client = AsyncOpenAI()
modal_client = AsyncOpenAI(api_key='super-secret-key', base_url="https://mihirathale98--vllm-app-serve.modal.run/v1")
together_client = AsyncOpenAI(api_key=os.environ.get("TOGETHER_API_KEY"),
  base_url="https://api.together.xyz/v1")

st.set_page_config(
    page_title="Model Comparison Demo",
    layout="wide"
)

st.title("Language Model Comparison")

user_input = st.text_area("Enter your prompt:", height=100)

with st.sidebar:
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    max_tokens = st.number_input("Max Tokens", min_value=100, max_value=4000, value=1000, step=100)
    model_left = st.selectbox("Model 1", ["llama-3-8b-instruct", "llama-3-8b-instruct-lora"], key="model_left")
    model_right = st.selectbox("Model 2", ["llama-3-8b-instruct","llama-3-8b-instruct-lora"], key="model_right")

async def stream_from_openai(
    prompt: str, 
    model: str = "gpt-4o-mini",
    temperature: float = 0.7, 
    max_tokens: int = 1000
) -> AsyncGenerator[str, None]:
    """Stream responses from OpenAI API using AsyncOpenAI."""

    if model == "llama-3-8b-instruct":
        client = together_client
        model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    elif model == "llama-3-8b-instruct-lora":
        client = modal_client
    # client = openai_client
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content
    except Exception as e:
        yield f"Error: {str(e)}"

def get_streaming_response(
    prompt: str, 
    model: str = "gpt-4o-mini",
    temperature: float = 0.7, 
    max_tokens: int = 1000
) -> Generator[str, None, None]:
    """Create a synchronous generator for streaming OpenAI responses."""
    async def async_generator():
        async for chunk in stream_from_openai(prompt, model, temperature, max_tokens):
            yield chunk
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        agen = async_generator()
        while True:
            try:
                chunk = loop.run_until_complete(agen.__anext__())
                yield chunk
            except StopAsyncIteration:
                break
    except Exception as e:
        yield f"Error: {str(e)}"
    finally:
        loop.close()

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Model 1: {model_left}")
    model1_output = st.empty()

with col2:
    st.subheader(f"Model 2: {model_right}")
    model2_output = st.empty()

if st.button("Compare Models", type="primary"):
    if user_input:
        model1_response = ""
        model2_response = ""
        
        with col1:
            metrics1 = st.empty()
        with col2:
            metrics2 = st.empty()
        
        gen1 = get_streaming_response(user_input, model_left, temperature, max_tokens)
        gen2 = get_streaming_response(user_input, model_right, temperature, max_tokens)
        
        gen1_done = False
        gen2_done = False
        
        with st.spinner("Generating responses..."):
            while not (gen1_done and gen2_done):
                if not gen1_done:
                    try:
                        chunk1 = next(gen1)
                        model1_response += chunk1
                        model1_output.markdown(model1_response)
                    except StopIteration:
                        gen1_done = True
                
                if not gen2_done:
                    try:
                        chunk2 = next(gen2)
                        model2_response += chunk2
                        model2_output.markdown(model2_response)
                    except StopIteration:
                        gen2_done = True
                
                time.sleep(0.01)
    
    else:
        st.warning("Please enter a prompt first.")