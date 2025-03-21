import json
import time
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from together import Together

# Initialize Together.ai API
client = Together()

@dataclass
class TranslationScore:
    translation: str
    model: str
    field: str
    meaning_score: float
    aae_score: float
    total_score: float

def parse_scores(text: str) -> Optional[tuple[float, float, float]]:
    """Parse scores from the model's response text."""
    try:
        # Split into lines and clean
        lines = [line.strip().lower() for line in text.split('\n')]
        
        # Look for score lines
        meaning_score = None
        aae_score = None
        total_score = None
        
        for line in lines:
            if 'meaning:' in line:
                try:
                    meaning_score = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    continue
            elif 'aae:' in line:
                try:
                    aae_score = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    continue
            elif 'total:' in line:
                try:
                    total_score = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    continue
        
        # Validate scores
        if meaning_score is None or aae_score is None or total_score is None:
            return None
            
        # Ensure scores are between 0 and 1
        meaning_score = max(0.0, min(1.0, meaning_score))
        aae_score = max(0.0, min(1.0, aae_score))
        total_score = max(0.0, min(1.0, total_score))
        
        return meaning_score, aae_score, total_score
    except Exception as e:
        print(f"Error parsing scores: {str(e)}")
        return None

def score_translation(original: str, translation: str, model: str, field: str) -> TranslationScore:
    """Use DeepSeek-V3 to score a translation."""
    prompt = f"""Compare these two texts and give scores (0-1):

SAE: {original}
AAE: {translation}

Score the AAE translation on:
1. Meaning preservation (0-1)
2. AAE authenticity (0-1)

Format:
Meaning: X
AAE: Y
Total: Z"""
    
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50,
        )
        text = response.choices[0].message.content.strip()
        
        # Parse scores
        scores = parse_scores(text)
        if scores is None:
            print(f"Failed to parse scores for translation")
            return TranslationScore(
                translation=translation,
                model=model,
                field=field,
                meaning_score=0.0,
                aae_score=0.0,
                total_score=0.0
            )
            
        meaning_score, aae_score, total_score = scores
        
        return TranslationScore(
            translation=translation,
            model=model,
            field=field,
            meaning_score=meaning_score,
            aae_score=aae_score,
            total_score=total_score
        )
            
    except Exception as e:
        print(f"Error in score_translation: {str(e)}")
        return TranslationScore(
            translation=translation,
            model=model,
            field=field,
            meaning_score=0.0,
            aae_score=0.0,
            total_score=0.0
        )

def save_checkpoint(checkpoint_file: str, data: Dict):
    """Save checkpoint data to file."""
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f, indent=2)

def load_checkpoint(checkpoint_file: str) -> Optional[Dict]:
    """Load checkpoint data from file if it exists."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def select_best_translations(example: Dict, translations: List[Dict]) -> Dict:
    """Select the best translations for instruction and input based on scoring."""
    # Group translations by field
    instruction_translations = [t for t in translations if t['field'] == 'instruction']
    input_translations = [t for t in translations if t['field'] == 'input']
    
    # Score all translations
    scored_translations = []
    
    # Score instruction translations
    for trans in instruction_translations:
        score = score_translation(
            example['instruction'],
            trans['text'],
            trans['model'],
            trans['field']
        )
        scored_translations.append(score)
        print(f"\nInstruction translation from {trans['model']}:")
        print(f"Text: {trans['text']}")
        print(f"Meaning Score: {score.meaning_score:.2f}")
        print(f"AAE Score: {score.aae_score:.2f}")
        print(f"Total Score: {score.total_score:.2f}")
        
        # Add delay between scoring calls to respect rate limits
        time.sleep(2)
    
    # Score input translations
    for trans in input_translations:
        score = score_translation(
            example['input'],
            trans['text'],
            trans['model'],
            trans['field']
        )
        scored_translations.append(score)
        print(f"\nInput translation from {trans['model']}:")
        print(f"Text: {trans['text']}")
        print(f"Meaning Score: {score.meaning_score:.2f}")
        print(f"AAE Score: {score.aae_score:.2f}")
        print(f"Total Score: {score.total_score:.2f}")
        
        # Add delay between scoring calls to respect rate limits
        time.sleep(2)
    
    # Select best translations for each field
    best_instruction = max(
        [t for t in scored_translations if t.field == 'instruction'],
        key=lambda x: x.total_score,
        default=None
    )
    best_input = max(
        [t for t in scored_translations if t.field == 'input'],
        key=lambda x: x.total_score,
        default=None
    )
    
    # Create AAE example
    aae_example = {
        'instruction': best_instruction.translation if best_instruction else example['instruction'],
        'input': best_input.translation if best_input else example.get('input', ''),
        'output': example['output']
    }
    
    return aae_example

def process_translations(input_file: str, output_file: str):
    """Process translation candidates, score them, and create the final AAE dataset."""
    checkpoint_file = 'data/scoring_checkpoint.json'
    
    # Load translation candidates
    with open(input_file, 'r') as f:
        translation_data = json.load(f)
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        print("Resuming from checkpoint...")
        aae_dataset = checkpoint.get('aae_dataset', [])
        processed_indices = set(checkpoint.get('processed_indices', []))
        print(f"Resumed {len(aae_dataset)} processed examples")
    else:
        aae_dataset = []
        processed_indices = set()
    
    # Process each example
    for i, example_data in enumerate(translation_data):
        if i in processed_indices:
            print(f"Skipping already processed example {i+1}")
            continue
            
        print(f"\nProcessing example {i+1}/{len(translation_data)}")
        
        try:
            # Select best translations
            aae_example = select_best_translations(
                example_data['original'],
                example_data['translations']
            )
            
            # Add to dataset
            aae_dataset.append(aae_example)
            processed_indices.add(i)
            
            # Save checkpoint after each example
            checkpoint_data = {
                'aae_dataset': aae_dataset,
                'processed_indices': list(processed_indices)
            }
            save_checkpoint(checkpoint_file, checkpoint_data)
            
            # Save final output
            with open(output_file, 'w') as f:
                json.dump(aae_dataset, f, indent=2)
            
            print(f"Saved checkpoint for example {i+1}")
            
        except Exception as e:
            print(f"Error processing example {i+1}: {str(e)}")
            print("Saving checkpoint before exiting...")
            checkpoint_data = {
                'aae_dataset': aae_dataset,
                'processed_indices': list(processed_indices)
            }
            save_checkpoint(checkpoint_file, checkpoint_data)
            raise
    
    print(f"\nSuccessfully processed {len(aae_dataset)} examples")
    print(f"Final AAE dataset saved to {output_file}")
    
    # Clean up checkpoint file after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

if __name__ == "__main__":
    input_file = 'data/translation_candidates.json'
    output_file = 'data/aae_data.json'
    process_translations(input_file, output_file) 