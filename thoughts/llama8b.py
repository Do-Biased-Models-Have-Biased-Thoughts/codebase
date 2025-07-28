import json
import os
from pathlib import Path
from datasets import Dataset
import torch
import transformers
from huggingface_hub import login
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
login(token="your token")
model_id = "meta-llama/Llama-3.1-8B-Instruct"
pipe = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
def extract_answer(text):
    """Extract the answer from the model response."""
    if text is None or text.strip() == '':
        return ''  # Return empty string instead of None
    
    match = re.search(r'{\s*"answer"\s*:\s*"([^"]+)"\s*}', text)
    if match:
        try:
            return json.loads(match.group(0))["answer"]
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from: {match.group(0)}")
    
    # If no JSON found, try to extract just the answer text
    # Look for common patterns in responses
    patterns = [
        r'"answer"\s*:\s*"([^"]+)"',  # Try to extract just from the key-value pair
        r'answer is (.+?)[\.\n]',      # Look for "answer is X" pattern
        r'I choose (.+?)[\.\n]',       # Look for "I choose X" pattern
        r'The answer is (.+?)[\.\n]',  # Another common pattern
        r'select (.+?)[\.\n]',         # "I select X" pattern
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If all else fails, just return an empty string to avoid None errors
    logger.warning(f"Could not extract answer from: {text[:100]}...")
    return ''

def prepare_prompts(dataset):
    prompt_template = '''Given context: {context}\nQuestion: {question}\nOptions: {options}\nGiven the context, question, and options, your job is to answer the question by selecting only one correct option from the given three options. You must provide your selected answer and explanation for selecting the answer in JSON format with "answer" and "explanation" as the only keys. Make sure there is no other text than just one json in your response'''
    def format_prompt(example):
        options = [example["ans0"], example["ans1"], example["ans2"]]
        options_str = ", ".join(options)
        return prompt_template.format(
            context=example["context"],
            question=example["question"],
            options=options_str
        )
    
    dataset = dataset.map(lambda x: {"prompt": format_prompt(x)})
    return dataset

def process_batch(batch, pipe):
    """Process a batch of prompts and return responses."""
    prompts = batch["prompt"]
    responses = []
    
    try:
        # Process each prompt individually for more reliable results
        for i, prompt in enumerate(prompts):
            try:
                messages = [
                    {"role": "user", "content": prompt}
                ]
                
                formatted_prompt = pipe.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )

                terminators = [
                    pipe.tokenizer.eos_token_id,
                    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                result = pipe(
                    formatted_prompt,
                    max_new_tokens=256,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.01,
                    top_p=0.95,
                )
                
                # Extract generated text
                generated_text = result[0]["generated_text"].strip()
                responses.append(generated_text)
                
            except Exception as e:
                logger.error(f"Error processing prompt {i}: {str(e)}")
                responses.append("")  # Add empty string for failed generation
                
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        # Return empty strings for any remaining prompts
        while len(responses) < len(prompts):
            responses.append("")
    
    return responses

def match_answer_to_options(predicted_answer, options):
    """Match the predicted answer to one of the options."""
    if predicted_answer is None or predicted_answer == '':
        return -1
        
    predicted_answer = predicted_answer.lower()
    options = [option.lower() for option in options]
    
    # First try exact matching
    for index, option in enumerate(options):
        if predicted_answer == option:
            return index
            
    # Then try substring matching
    for index, option in enumerate(options):
        if predicted_answer in option or option in predicted_answer:
            return index
    
    # If still no match, try to find the option with highest word overlap
    pred_words = set(predicted_answer.split())
    max_overlap = 0
    best_index = -1
    
    for index, option in enumerate(options):
        option_words = set(option.split())
        overlap = len(pred_words.intersection(option_words))
        if overlap > max_overlap:
            max_overlap = overlap
            best_index = index
    
    # Only return the best index if there was some overlap
    if max_overlap > 0:
        return best_index
        
    return -1

def process_dataset(input_file, output_file, pipe, batch_size=32, save_interval=32):
    """Process an input JSONL file and save the processed data."""
    # Ensure output file directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Remove output file if it exists to avoid appending to previous results
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Load dataset
    dataset = Dataset.from_json(str(input_file))
    dataset = prepare_prompts(dataset)
    
    processed_rows = []
    total = len(dataset)
    
    logger.info(f"Processing {total} examples from {input_file}")
    
    for i in range(0, total, batch_size):
        # Select a batch of examples
        end_idx = min(i + batch_size, total)
        batch = dataset.select(range(i, end_idx))
        
        logger.info(f"Processing batch {i}-{end_idx} of {total}")
        
        # Process the batch
        batch_responses = process_batch({"prompt": batch["prompt"]}, pipe)
        
        for idx, response in enumerate(batch_responses):
            example = batch[idx].copy()  # Make a copy to avoid modifying the original
            example["response"] = response
            
            # Save only the raw response without extracting or matching answers
            processed_rows.append(example)
        
        # Save intermediate results
        if len(processed_rows) >= save_interval:
            with open(output_file, "a", encoding="utf-8") as outfile:
                for row in processed_rows:
                    outfile.write(json.dumps(row) + "\n")
            
            logger.info(f"Saved {len(processed_rows)} rows to {output_file}")
            processed_rows = []  # Clear the buffer
    
    # Save any remaining rows
    if processed_rows:
        with open(output_file, "a", encoding="utf-8") as outfile:
            for row in processed_rows:
                outfile.write(json.dumps(row) + "\n")
        
        logger.info(f"Saved remaining {len(processed_rows)} rows to {output_file}")

def process_jsonl_files(input_dir, output_dir):
    """Process all JSONL files in the input directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all JSONL files
    input_files = list(Path(input_dir).glob("*.jsonl"))
    logger.info(f"Found {len(input_files)} JSONL files to process")
    
    for input_file in input_files:
        output_file = Path(output_dir) / input_file.name
        logger.info(f"Processing {input_file}...")
        
        try:
            process_dataset(input_file, output_file, pipe, batch_size=4)
            logger.info(f"Completed processing {input_file} -> {output_file}")
        except Exception as e:
            logger.error(f"Error processing {input_file}: {str(e)}")

if __name__ == "__main__":
    input_dir = "data"
    output_dir = "output"
    process_jsonl_files(input_dir, output_dir)