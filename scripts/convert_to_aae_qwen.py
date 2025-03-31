import json
import os
from together import Together
from typing import List, Dict
from dataclasses import dataclass

# Initialize Together.ai API
client = Together()

def load_checkpoint(output_file: str) -> List[Dict]:
    """Load existing AAE dataset if it exists."""
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            return json.load(f)
    return []   

def convert_to_aae(text: str) -> str:
    """Convert SAE text to AAE using Qwen."""
    prompt = f"""Convert this SAE text to AAE:

SAE: {text}

AAE:"""
    
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2-VL-72B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
        )
        text = response.choices[0].message.content.strip()
        
        # Parse the response to get just the translation
        lines = text.split('\n')
        for line in lines:
            if line.startswith('AAE:'):
                return line[4:].strip()
            elif not line.startswith('SAE:'):
                return line.strip()
        
        return text.strip()
    except Exception as e:
        print(f"Error in convert_to_aae: {str(e)}")
        return None

def process_dataset(input_file: str, output_file: str):
    """Process the dataset and create AAE translations."""
    # Load dataset
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Load existing AAE dataset to resume from last processed example
    aae_dataset = load_checkpoint(output_file)
    start_idx = len(aae_dataset)
    
    print(f"Found {len(aae_dataset)} existing translations")
    print(f"Resuming from example {start_idx + 1}/{len(data)}")
    
    # Process remaining examples
    for i in range(start_idx, len(data)):
        example = data[i]
        print(f"\nProcessing example {i+1}/{len(data)}")
        
        try:
            # Translate instruction
            aae_instruction = convert_to_aae(example['instruction'])
            if not aae_instruction:
                print(f"Failed to translate instruction for example {i+1}")
                continue
            
            # Translate input if it exists
            aae_input = ""
            if example.get('input'):
                aae_input = convert_to_aae(example['input'])
                if not aae_input:
                    print(f"Failed to translate input for example {i+1}")
                    continue
            
            # Create AAE example
            aae_example = {
                'instruction': aae_instruction,
                'input': aae_input,
                'output': example['output']
            }
            
            # Add to dataset
            aae_dataset.append(aae_example)
            
            # Save intermediate results
            with open(output_file, 'w') as f:
                json.dump(aae_dataset, f, indent=2)
            
            print(f"Successfully translated example {i+1}")
            
        except Exception as e:
            print(f"Error processing example {i+1}: {str(e)}")
            print("Saving checkpoint before exiting...")
            with open(output_file, 'w') as f:
                json.dump(aae_dataset, f, indent=2)
            raise
    
    print(f"\nSuccessfully processed {len(aae_dataset)} examples")
    print(f"AAE dataset saved to {output_file}")

if __name__ == "__main__":
    input_file = 'data/alpaca_data.json'
    output_file = 'data/aae_data.json'
    process_dataset(input_file, output_file) 