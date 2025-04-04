import transformers
import torch
from tqdm import tqdm
import os
import json
from .api_wrapper import *
from torch.utils.data import Dataset, DataLoader
from typing import List, Union, Dict, Any, Optional, Tuple

def init_pipeline(model_id: str,
                  task: str='text-generation',
                  token: str='',
                  **kwargs):
    '''
    Initialize inference pipelines for different models.
        Parameters:
            model_id (str): The model ID to be used.
            task (str): The task to be performed by the model.
            token (str): The token to be used for the model.
        Returns:
            pipeline (transformers.Pipeline): The initialized pipeline.
            terminators (list): The list of terminators for the model.
    '''
    assert torch.cuda.is_available(), "This model needs a GPU to run ..."
    device = torch.cuda.current_device()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, padding_side='left', token=token)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, token=token, device_map="auto")
    # Initialize the pipeline
    if 'llama-3' in model_id.lower():
        terminators = [tokenizer.eos_token_id,
                       tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    else:
        assert 'mistral' in model_id.lower() or 'gemma' in model_id.lower() or 'mixtral' in model_id.lower(), "Only Mistral/llama/gemma models are supported ..."
        terminators = [tokenizer.eos_token_id]

    tokenizer.pad_token_id = model.config.eos_token_id
    pipeline = transformers.pipeline(task=task,
                                     model=model,
                                     tokenizer=tokenizer,)
                                    #  device=device)
    print(f"Initialized pipeline for {model_id}.")
    # print(f"Initialized pipeline for {model_id} on device {device}.")
    return pipeline, terminators


class InferenceDataset(Dataset):
    """Dataset for batched inference with transformer models"""
    
    def __init__(
        self, 
        user_prompts: List[str], 
        sys_prompts: List[str] = None, 
        model_name: str = "",
        instruction_tuned: bool = True,
        tokenizer: transformers.PreTrainedTokenizer = None,
        max_length: int = 2048
    ):
        """
        Initialize the dataset with prompts
        
        Args:
            user_prompts: List of user prompts
            sys_prompts: List of system prompts (optional)
            model_name: Name of the model for formatting
            instruction_tuned: Whether the model is instruction-tuned
            tokenizer: Tokenizer for non-instruction-tuned models
            max_length: Max input length for tokenization
        """
        self.user_prompts = user_prompts
        self.sys_prompts = sys_prompts if sys_prompts and len(sys_prompts) > 0 else [""] * len(user_prompts)
        self.model_name = model_name.lower()
        self.instruction_tuned = instruction_tuned
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-compute formatted queries
        self.batched_queries = self._format_queries()
        
    def _format_queries(self):
        """Format queries based on model type"""
        if self.instruction_tuned:
            if all(not sys for sys in self.sys_prompts):
                # No system prompts
                return [[{'role': 'user', 'content': self.user_prompts[i]}] for i in range(len(self.user_prompts))]
            else:
                # With system prompts
                if 'llama' in self.model_name:
                    return [
                        [{"role": "system", "content": self.sys_prompts[i]}, {'role': 'user', 'content': self.user_prompts[i]}] 
                        for i in range(len(self.user_prompts))
                    ]
                elif any(model in self.model_name for model in ['mistral', 'gemma', 'mixtral']):
                    return [
                        [{'role': 'user', 'content': self.sys_prompts[i]+'\n\n'+self.user_prompts[i]}] 
                        for i in range(len(self.user_prompts))
                    ]
                else:
                    raise ValueError(f"Unsupported model architecture: {self.model_name}")
        else:
            # For non-instruction-tuned models, tokenize inputs
            if self.tokenizer is None:
                raise ValueError("Tokenizer must be provided for non-instruction-tuned models")
            
            # Apply custom template if available, otherwise use simple concatenation
            try:
                from templates import apply_template
                return apply_template([[p] for p in self.user_prompts], model_name=self.model_name, urial='inst_1k_v4.help')
            except ImportError:
                # Simple concatenation as fallback
                return [f"{self.sys_prompts[i]}\n\n{self.user_prompts[i]}" for i in range(len(self.user_prompts))]
    
    def __len__(self):
        return len(self.user_prompts)
    
    def __getitem__(self, idx):
        if self.instruction_tuned:
            return self.batched_queries[idx]
        else:
            # For non-instruction models, return pre-tokenized inputs
            inputs = self.tokenizer(
                self.batched_queries[idx],
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )
            return {k: v.squeeze(0) for k, v in inputs.items()}


def inference(
    pipeline: transformers.Pipeline,
    user_prompts: List[str],
    terminators: List[int],
    model: Optional[transformers.AutoModelForCausalLM] = None,
    tokenizer: Optional[transformers.AutoTokenizer] = None,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
    sys_prompts: List[str] = [],
    max_new_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 1,
    num_return_sequences: int = 1,
    instruction_tuned: bool = True,
    num_workers: int = 2,  # New parameter for DataLoader workers
    pin_memory: bool = True,  # New parameter for faster GPU transfer
    mixed_precision: bool = True,  # New parameter for mixed precision
    **kwargs
) -> List[str]:
    '''
    Optimized inference using PyTorch Dataset for non-instruction models.
    For instruction-tuned models, processes each prompt individually to avoid template issues.
    
    Parameters:
        pipeline (transformers.Pipeline): The initialized pipeline.
        user_prompts (List[str]): List of user prompts.
        terminators (List[int]): List of terminator token IDs.
        model (transformers.AutoModelForCausalLM): Model for non-pipeline inference.
        tokenizer (transformers.AutoTokenizer): Tokenizer for the model.
        save_dir (str): Directory to save outputs.
        save_name (str): Filename for saved outputs.
        sys_prompts (List[str]): List of system prompts.
        max_new_tokens (int): Maximum new tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Top-p sampling parameter.
        batch_size (int): Batch size for inference.
        num_return_sequences (int): Number of sequences to return per prompt.
        instruction_tuned (bool): Whether the model is instruction-tuned.
        num_workers (int): Number of worker processes for DataLoader.
        pin_memory (bool): Whether to pin memory for faster GPU transfer.
        mixed_precision (bool): Whether to use mixed precision for inference.
        
    Returns:
        List[str]: Generated outputs.
    '''
    assert isinstance(user_prompts, list), "The user prompts must be a list"
    assert isinstance(sys_prompts, list), "The system prompts must be a list"
    
    # Set do_sample based on temperature
    do_sample = temperature > 0
    
    # Ensure system prompts match user prompts in length if provided
    if sys_prompts and sys_prompts[0]:
        assert len(user_prompts) == len(sys_prompts), "Number of user prompts must equal number of system prompts"
    
    # Get model name
    try:
        model_name = pipeline.model.config._name_or_path if pipeline else model.config._name_or_path
    except:
        model_name = "unknown_model"
    
    # Create dataset
    dataset = InferenceDataset(
        user_prompts=user_prompts,
        sys_prompts=sys_prompts,
        model_name=model_name,
        instruction_tuned=instruction_tuned,
        tokenizer=tokenizer if not instruction_tuned else None,
        max_length=max_new_tokens
    )
    
    # Try to load existing outputs if save_dir and save_name are provided
    outputs = []
    start_idx = 0
    
    if save_dir and save_name:
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, save_name)
        input_path = os.path.join(
            save_dir, 
            save_name.replace('output.json', 'input.json').replace('outputs.json', 'inputs.json')
        )
        
        try:
            with open(output_path, 'r') as f:
                outputs = json.load(f)
                start_idx = len(outputs)
        except (FileNotFoundError, json.JSONDecodeError):
            outputs = []
        
        # Save inputs
        with open(input_path, 'w') as f:
            json.dump(dataset.batched_queries, f, indent=4)
    
    # Different processing for instruction-tuned vs non-instruction-tuned models
    if instruction_tuned:
        # For instruction-tuned models, process each prompt individually
        # to avoid template formatting issues with batch processing
        
        # Skip already processed prompts
        queries = dataset.batched_queries[start_idx:]
        
        # Set up mixed precision if requested and supported
        amp_dtype = torch.float16 if mixed_precision and torch.cuda.is_available() else torch.float32
        
        # Process each prompt
        with torch.cuda.amp.autocast(enabled=mixed_precision and torch.cuda.is_available()):
            for i in tqdm(range(0, len(queries), batch_size)):
                batch_queries = queries[i:i+batch_size]
                batch_outputs = []
                
                # Process each prompt in the batch individually to avoid template issues
                for query in batch_queries:
                    with torch.no_grad():
                        result = pipeline(
                            query,  # Pass individual message list
                            max_new_tokens=max_new_tokens,
                            eos_token_id=terminators,
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p,
                            return_full_text=False,
                            num_return_sequences=num_return_sequences
                        )
                    
                    # Format result based on return type
                    if isinstance(result, list):
                        # Multiple return sequences
                        texts = [item["generated_text"] for item in result]
                    else:
                        # Single return sequence
                        texts = [result["generated_text"]]
                    
                    batch_outputs.append(texts)
                
                outputs.extend(batch_outputs)
                
                # Save intermediate results
                if save_dir and save_name:
                    with open(output_path, 'w') as f:
                        json.dump(outputs, f, indent=4)
    else:
        # For non-instruction-tuned models, use DataLoader for efficient batching
        
        # Set up DataLoader
        effective_workers = num_workers if batch_size > 1 and len(user_prompts) > batch_size else 0
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=effective_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            shuffle=False
        )
        
        # Skip already processed batches
        skip_batches = start_idx // batch_size
        
        # Set up mixed precision
        with torch.cuda.amp.autocast(enabled=mixed_precision and torch.cuda.is_available()):
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                if batch_idx < skip_batches:
                    continue
                
                # Move inputs to the correct device
                batch = {k: v.to(model.device) for k, v in batch.items()}
                
                input_len = batch['input_ids'].shape[1]
                
                # Generate with non-pipeline model
                with torch.no_grad():
                    output_ids = model.generate(
                        **batch,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        num_return_sequences=num_return_sequences,
                        eos_token_id=terminators
                    )
                
                # Decode outputs
                batch_outputs = []
                for i in range(0, len(output_ids), num_return_sequences):
                    sequence_outputs = []
                    for j in range(num_return_sequences):
                        if i + j < len(output_ids):
                            text = tokenizer.decode(
                                output_ids[i + j][input_len:], 
                                skip_special_tokens=True
                            )
                            sequence_outputs.append(text)
                    batch_outputs.append(sequence_outputs)
                
                outputs.extend(batch_outputs)
                
                # Save intermediate results
                if save_dir and save_name:
                    with open(output_path, 'w') as f:
                        json.dump(outputs, f, indent=4)
    
    # Return the first generated sequence for each prompt if multiple were requested
    return [o[0] for o in outputs]

def pipe_and_infer(model_id: str,
                   save_dir: str=None,
                   save_name: str=None,
                   use_api: bool=False,
                   top_p: float=1.0,
                   max_new_tokens: int=4096,
                   **kwargs):
    '''
    Initialize and run inference on a model.
        Parameters:
            model_id (str): The model ID to be used.
            save_dir (str): The directory to save the outputs.
            save_name (str): The name of the file to save the outputs.
            use_api (bool): Whether to use the API for inference
            **kwargs: Additional keyword arguments for the inference and pipeline function.
        Returns:
            outputs (list): The list of outputs from the model.
    '''
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    if not use_api:
        if 'it' in model_id.lower() or 'inst' in model_id.lower() or 'chat' in model_id.lower():
            kwargs['instruction_tuned'] = True
            pipeline, terminators = init_pipeline(model_id=model_id,
                                            **kwargs)
            outputs = inference(pipeline=pipeline,
                                terminators=terminators,
                                save_dir=save_dir,
                                save_name=save_name,
                                **kwargs)
            # unload the model
            pipeline.model = None
        else:
            kwargs['instruction_tuned'] = False
            model = transformers.AutoModelForCausalLM.from_pretrained(model_id, token=kwargs['token'], device_map="auto")
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, padding_side='left', **kwargs)
            tokenizer.pad_token_id = model.config.eos_token_id
            outputs = inference(model=model,
                                pipeline=None,
                                terminators=None,
                                tokenizer=tokenizer,
                                save_dir=save_dir,
                                save_name=save_name,
                                **kwargs)
            # unload the model
            model = None
            
    else:
        try:
            with open(os.path.join(save_dir, save_name), 'r') as f:
                outputs = json.load(f)
        except:
            outputs = []
        start_idx = len(outputs)
        llama_client = LlamaWrapper(url=kwargs['url'],
                                    api_key=kwargs['api_key'],
                                    deployment=kwargs['deployment'])
        if kwargs['sys_prompts']:
            assert len(kwargs['user_prompts']) == len(kwargs['sys_prompts']), "The number of user prompts must be equal to the number of system prompts"
        for i in tqdm(range(len(kwargs['user_prompts']))):
            if i < start_idx:
                continue
            user_prompt = kwargs['user_prompts'][i]
            output = llama_client.run(system_prompt = kwargs['sys_prompts'][i],
                                      user_prompt=user_prompt,
                                      model_version=model_id,
                                      temperature=kwargs['temperature'],
                                      top_p=top_p,
                                      max_new_tokens=max_new_tokens)
            outputs.append(output)
            with open(os.path.join(save_dir, save_name), 'w') as f:
                json.dump(outputs, f, indent=4)
        # check no response
        flag = True
        count = 0
        while flag and count < 3:
            count+=1
            print("Checking for API NO RESPONSE ...")
            flag = False
            for i in tqdm(range(len(outputs))):
                if outputs[i] == "API NO RESPONSE":
                    # try to rerun the API
                    user_prompt = kwargs['user_prompts'][i]
                    output = llama_client.run(system_prompt = kwargs['sys_prompts'][i],
                                            user_prompt=user_prompt,
                                            model_version=model_id,
                                            temperature=kwargs['temperature'],
                                            top_p=top_p,
                                            max_new_tokens=max_new_tokens)
                    outputs[i] = output
                    with open(os.path.join(save_dir, save_name), 'w') as f:
                        json.dump(outputs, f, indent=4)
                    flag = True
        
    return outputs
