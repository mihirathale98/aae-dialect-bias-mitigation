import transformers
import torch
from tqdm import tqdm
import os
import json
from .api_wrapper import *


def init_pipeline(model_id: str,
                  task: str='text-generation',
                  token: str='',
                  lora_weights: str=None,
                  lora_config: dict=None,
                  **kwargs):
    '''
    Initialize inference pipelines for different models with optional LoRA weights.
        Parameters:
            model_id (str): The model ID to be used.
            task (str): The task to be performed by the model.
            token (str): The token to be used for the model.
            lora_weights (str): Path to LoRA weights or HF repo ID containing LoRA weights.
            lora_config (dict): Configuration for LoRA adapter loading (optional).
        Returns:
            pipeline (transformers.Pipeline): The initialized pipeline.
            terminators (list): The list of terminators for the model.
    '''
    import torch
    import transformers
    from peft import PeftModel, PeftConfig
    
    assert torch.cuda.is_available(), "This model needs a GPU to run ..."
    device = torch.cuda.current_device()
    
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, padding_side='left', token=token)
    
    # Load base model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, 
        token=token, 
        device_map="auto"
    )
    
    # Load LoRA weights if provided
    if lora_weights:
        print(f"Loading LoRA weights from {lora_weights}")
        
        # If lora_config is provided, use it for custom loading parameters
        if lora_config is None:
            lora_config = {}  # Use default configuration if not provided
            
        # Load the model with LoRA weights
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            **lora_config
        )
        
        # Optionally merge weights for faster inference if specified in config
        if lora_config.get('merge_weights', False):
            print("Merging LoRA weights with base model for optimized inference")
            model = model.merge_and_unload()
    
    # Determine terminators based on model type
    if 'llama-3' in model_id.lower():
        terminators = [tokenizer.eos_token_id,
                      tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    else:
        assert 'mistral' in model_id.lower() or 'gemma' in model_id.lower() or 'mixtral' in model_id.lower(), "Only Mistral/llama/gemma models are supported ..."
        terminators = [tokenizer.eos_token_id]

    # Set pad token to eos token if not already set
    tokenizer.pad_token_id = model.config.eos_token_id
    
    # Initialize the pipeline
    pipeline = transformers.pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
    )
    
    print(f"Initialized pipeline for {model_id}" + 
          (f" with LoRA weights from {lora_weights}" if lora_weights else ""))
    
    return pipeline, terminators

def inference(pipeline: transformers.Pipeline,
              user_prompts: list,
              terminators: list,
              model: transformers.AutoModelForCausalLM=None,
              tokenizer: transformers.AutoTokenizer=None,
              save_dir: str=None,
              save_name: str=None,
              sys_prompts: list=[],
              max_new_tokens: int=4096,
              temperature: float=1,
              top_p: float=1.0,
              batch_size: int=1,
              num_return_sequences: int=1,
              instruction_tuned: bool=True,
              **kwargs):
    '''
    Wrap up the inference process.
        Parameters:
            pipeline (transformers.Pipeline): The initialized pipeline.
            user_prompt (str): The user prompt to be used.
            terminators (list): The list of terminators for the model.
            max_length (int): The maximum length of the output.
            sys_prompt (str): The system prompt to be used.
            temperature (float): The temperature to be used.
            do_sample (bool): Whether to sample the output.
            top_p (float): The top-p value to be used.
            batch_size (int): The batch size to be used.
            num_return_sequences (int): The number of sequences to be returned.
        Returns:
            outputs (list): The list of outputs from the model.
    '''
    assert isinstance(user_prompts, list), "The user prompts must be a list"
    assert isinstance(sys_prompts, list), "The system prompts must be a list"
    if temperature:
        do_sample = True
    else:
        do_sample = False
    if sys_prompts[0]:
        assert len(user_prompts) == len(sys_prompts), "The number of user prompts must be equal to the number of system prompts"
    try:
        model_name = pipeline.model.config._name_or_path
    except:
        model_name = model.config._name_or_path
    if instruction_tuned:
        if not sys_prompts[0]:
            batched_queries = [[{'role': 'user', 'content': user_prompts[i]}] for i in range(len(user_prompts))]
        else:
            if 'llama' in model_name.lower():
                batched_queries = [[ {"role": "system", "content": sys_prompts[i]}, {'role': 'user', 'content': user_prompts[i]}] for i in range(len(user_prompts))]
            else:
                assert 'mistral' in model_name.lower() or 'gemma' in model_name.lower() or 'mixtral' in model_name.lower(), "Only Mistral/llama/mixtral models are supported ..."
                # Mistral models do not require system prompts so append the user prompts to the system prompts
                batched_queries = [[{'role': 'user', 'content': sys_prompts[i]+'\n\n'+user_prompts[i]}] for i in range(len(user_prompts))]
    else:
        batched_queries = apply_template([[p] for p in user_prompts],model_name=model_name, urial='inst_1k_v4.help')
    try:
        with open(os.path.join(save_dir, save_name), 'r') as f:
            outputs = json.load(f)
    except:
        outputs = []
    start_idx = len(outputs)
    if save_dir:
        with open(os.path.join(save_dir, save_name.replace('output.json', 'input.json').replace('outputs.json', 'inputs.json')), 'w') as f:
            json.dump(batched_queries, f, indent=4)
    for i in tqdm(range(start_idx, len(batched_queries), batch_size)):
        if instruction_tuned:
            curr_outputs = pipeline(batched_queries[i:i+batch_size],
                                    max_new_tokens=max_new_tokens,
                                    eos_token_id=terminators,
                                    do_sample=do_sample,
                                    temperature=temperature,
                                    top_p=top_p,
                                    return_full_text=False,
                                    batch_size=batch_size,
                                    num_return_sequences=num_return_sequences)
            outputs.extend([[oo["generated_text"] for oo in o] for o in curr_outputs])
        else:
            inputs = tokenizer(batched_queries[i:i+batch_size],
                               return_tensors="pt",
                               padding=True,
                               truncation=True,
                               max_length=max_new_tokens)
            inputs.to('cuda')
            input_len = inputs['input_ids'].shape[1]
            output_ids = model.generate(**inputs,
                                        max_new_tokens=200,
                                        do_sample=do_sample,
                                        temperature=temperature,
                                        top_p=top_p,
                                        num_return_sequences=num_return_sequences)
            curr_outputs = [tokenizer.decode(output_id[input_len:], skip_special_tokens=True) for output_id in output_ids]
            outputs.extend(curr_outputs)
        if save_dir:
            with open(os.path.join(save_dir, save_name), 'w') as f:
                json.dump(outputs, f, indent=4)
    return [o[0] for o in outputs]


# def pipe_and_infer(model_id: str,
#                    save_dir: str=None,
#                    save_name: str=None,
#                    use_api: bool=False,
#                    top_p: float=1.0,
#                    max_new_tokens: int=4096,
#                    **kwargs):
#     '''
#     Initialize and run inference on a model.
#         Parameters:
#             model_id (str): The model ID to be used.
#             save_dir (str): The directory to save the outputs.
#             save_name (str): The name of the file to save the outputs.
#             use_api (bool): Whether to use the API for inference
#             **kwargs: Additional keyword arguments for the inference and pipeline function.
#         Returns:
#             outputs (list): The list of outputs from the model.
#     '''
#     if save_dir:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#     if not use_api:
#         if 'it' in model_id.lower() or 'inst' in model_id.lower() or 'chat' in model_id.lower():
#             kwargs['instruction_tuned'] = True
#             pipeline, terminators = init_pipeline(model_id=model_id,
#                                             **kwargs)
#             outputs = inference(pipeline=pipeline,
#                                 terminators=terminators,
#                                 save_dir=save_dir,
#                                 save_name=save_name,
#                                 **kwargs)
#             # unload the model
#             pipeline.model = None
#         else:
#             kwargs['instruction_tuned'] = False
#             model = transformers.AutoModelForCausalLM.from_pretrained(model_id, token=kwargs['token'], device_map="auto")
#             tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, padding_side='left', **kwargs)
#             tokenizer.pad_token_id = model.config.eos_token_id
#             outputs = inference(model=model,
#                                 pipeline=None,
#                                 terminators=None,
#                                 tokenizer=tokenizer,
#                                 save_dir=save_dir,
#                                 save_name=save_name,
#                                 **kwargs)
#             # unload the model
#             model = None
            
#     else:
#         try:
#             with open(os.path.join(save_dir, save_name), 'r') as f:
#                 outputs = json.load(f)
#         except:
#             outputs = []
#         start_idx = len(outputs)
#         llama_client = LlamaWrapper(url=kwargs['url'],
#                                     api_key=kwargs['api_key'],
#                                     deployment=kwargs['deployment'])
#         if kwargs['sys_prompts']:
#             assert len(kwargs['user_prompts']) == len(kwargs['sys_prompts']), "The number of user prompts must be equal to the number of system prompts"
#         for i in tqdm(range(len(kwargs['user_prompts']))):
#             if i < start_idx:
#                 continue
#             user_prompt = kwargs['user_prompts'][i]
#             output = llama_client.run(system_prompt = kwargs['sys_prompts'][i],
#                                       user_prompt=user_prompt,
#                                       model_version=model_id,
#                                       temperature=kwargs['temperature'],
#                                       top_p=top_p,
#                                       max_new_tokens=max_new_tokens)
#             outputs.append(output)
#             with open(os.path.join(save_dir, save_name), 'w') as f:
#                 json.dump(outputs, f, indent=4)
#         # check no response
#         flag = True
#         count = 0
#         while flag and count < 3:
#             count+=1
#             print("Checking for API NO RESPONSE ...")
#             flag = False
#             for i in tqdm(range(len(outputs))):
#                 if outputs[i] == "API NO RESPONSE":
#                     # try to rerun the API
#                     user_prompt = kwargs['user_prompts'][i]
#                     output = llama_client.run(system_prompt = kwargs['sys_prompts'][i],
#                                             user_prompt=user_prompt,
#                                             model_version=model_id,
#                                             temperature=kwargs['temperature'],
#                                             top_p=top_p,
#                                             max_new_tokens=max_new_tokens)
#                     outputs[i] = output
#                     with open(os.path.join(save_dir, save_name), 'w') as f:
#                         json.dump(outputs, f, indent=4)
#                     flag = True
        
#     return outputs

def pipe_and_infer(model_id: str,
                   save_dir: str=None,
                   save_name: str=None,
                   use_api: bool=False,
                   use_modal: bool=False,
                   modal_endpoint: str=None,
                   top_p: float=1.0,
                   max_new_tokens: int=4096,
                   **kwargs):
    '''
    Initialize and run inference on a model.
        Parameters:
            model_id (str): The model ID to be used.
            save_dir (str): The directory to save the outputs.
            save_name (str): The name of the file to save the outputs.
            use_api (bool): Whether to use the API for inference.
            use_modal (bool): Whether to use a Modal endpoint for inference.
            modal_endpoint (str): The URL of the Modal endpoint.
            **kwargs: Additional keyword arguments for the inference and pipeline function.
        Returns:
            outputs (list): The list of outputs from the model.
    '''
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    # Option 3: Use Modal endpoint
    if use_modal:
        import requests
        import json
        
        try:
            with open(os.path.join(save_dir, save_name), 'r') as f:
                outputs = json.load(f)
        except:
            outputs = []
            
        start_idx = len(outputs)
        
        assert modal_endpoint is not None, "Modal endpoint URL must be provided when use_modal=True"
        
        # Prepare headers for the request
        headers = {
            'Content-Type': 'application/json'
        }
        
        # Add authentication header if provided
        if 'modal_api_key' in kwargs:
            headers['Authorization'] = f"Bearer {kwargs['modal_api_key']}"
            
        for i in tqdm(range(len(kwargs['user_prompts']))):
            if i < start_idx:
                continue
                
            user_prompt = kwargs['user_prompts'][i]
            sys_prompt = kwargs['sys_prompts'][i] if kwargs.get('sys_prompts') and i < len(kwargs['sys_prompts']) else ""
            
            # Prepare the payload for Modal endpoint
            payload = {
                'prompt': user_prompt,
                'temperature': kwargs.get('temperature', 0),
                'max_tokens': 2048,
            }
            
            try:
                # Make the request to the Modal endpoint
                response = requests.post(modal_endpoint, headers=headers, json=payload)
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                # Parse the response
                result = response.json()
                
                # Extract the model's response text from the result
                # Adjust this based on your Modal endpoint's response format
                output = result.get('response', "Modal API NO RESPONSE")
                
            except Exception as e:
                print(f"Error calling Modal endpoint: {str(e)}")
                output = "Modal API ERROR"
                
            outputs.append(output)
            
            # Save after each iteration to ensure progress is retained
            if save_dir:
                with open(os.path.join(save_dir, save_name), 'w') as f:
                    json.dump(outputs, f, indent=4)
        
        # Retry logic for failed responses
        flag = True
        count = 0
        while flag and count < 3:
            count += 1
            print("Checking for Modal API NO RESPONSE or ERROR...")
            flag = False
            for i in tqdm(range(len(outputs))):
                if outputs[i] in ["Modal API NO RESPONSE", "Modal API ERROR"]:
                    # try to rerun the API
                    user_prompt = kwargs['user_prompts'][i]
                    sys_prompt = kwargs['sys_prompts'][i] if kwargs.get('sys_prompts') and i < len(kwargs['sys_prompts']) else ""
                    
                    payload = {
                        'prompt': user_prompt,
                        'system_prompt': sys_prompt,
                        'temperature': kwargs.get('temperature', 1.0),
                        'top_p': top_p,
                        'max_tokens': max_new_tokens
                    }
                    
                    try:
                        response = requests.post(modal_endpoint, headers=headers, json=payload)
                        response.raise_for_status()
                        result = response.json()
                        output = result.get('response', "Modal API NO RESPONSE")
                    except Exception as e:
                        print(f"Error in retry {count} for prompt {i}: {str(e)}")
                        output = "Modal API ERROR after retry"
                        
                    outputs[i] = output
                    
                    if save_dir:
                        with open(os.path.join(save_dir, save_name), 'w') as f:
                            json.dump(outputs, f, indent=4)
                    flag = True
                    
    # Original Option 1: Local inference
    elif not use_api:
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
            
    # Original Option 2: API inference
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