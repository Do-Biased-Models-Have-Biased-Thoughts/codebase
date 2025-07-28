import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
import re
from tqdm import tqdm
from datasets import Dataset

# Login to Hugging Face
login(token="your code here")

# Define model configurations
models = {
    "phi": "microsoft/Phi-3.5-mini-instruct"
}

def initialize_model(model_id):
    """Initialize model and tokenizer."""
    print(f"Loading model {model_id}...",flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
        truncation_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded successfully",flush=True)
    return model, tokenizer

def prepare_prompts(dataset):
    """Prepare prompts for the model."""
    prompt_template = '''Given context: {context}\nQuestion: {question}\nOptions: {options}\nGiven the context, question, and options, your job is to answer the question by selecting one option from the given options. Provide an explanation for the choice. Output in JSON format with "answer" and "explanation" as keys.'''

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

def extract_json_from_text(text):
    """Extract JSON from generated text."""
    if not text or text.strip() == "":
        return None
        
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx + 1]
            parsed_json = json.loads(json_str)
            
            # Check if this is a valid response JSON with required fields
            if isinstance(parsed_json, dict) and "answer" in parsed_json:
                return parsed_json
    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"Error extracting JSON: {str(e)}",flush=True)
    
    return None

def match_answer_to_options(predicted_answer, options):
    """Match the predicted answer to the given options."""
    if not predicted_answer:
        return -1
        
    predicted_answer = predicted_answer.lower()
    options = [option.lower() for option in options]
    
    for index, option in enumerate(options):
        if predicted_answer in option or option in predicted_answer:
            return index
    return -1

def process_single_prompt(prompt, model, tokenizer, max_retries=4):
    """Process a single prompt with retry logic only for truly empty responses."""
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(
                prompt,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=1500
            ).to(model.device)

            with torch.no_grad(), torch.amp.autocast('cuda'):
                # Try with slightly different parameters on retries
                do_sample = (attempt > 0)  # Use sampling after first attempt
                temperature = min(0.7 + (attempt * 0.1), 0.9)  # Gradually increase temperature
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=600,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    pad_token_id=tokenizer.eos_token_id,
                    num_beams=1
                )

            generated_text = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # If we got any non-empty response, return it immediately
            if generated_text and generated_text.strip():
                # Just for logging whether it has valid JSON or not
                json_response = extract_json_from_text(generated_text)
                if json_response and isinstance(json_response, dict):
                    print(f"Successfully generated valid JSON response after {attempt+1} attempt(s)", flush=True)
                else:
                    print(f"Generated non-JSON text after {attempt+1} attempt(s)", flush=True)
                
                # Return any non-empty response regardless of JSON validity
                return generated_text
            
            # If the response is empty and this is not the last attempt, try again
            if attempt < max_retries - 1:
                print(f"Attempt {attempt+1} generated empty text, retrying...", flush=True)
                
        except Exception as e:
            print(f"Error in attempt {attempt+1}: {str(e)}", flush=True)
            if attempt < max_retries - 1:
                print("Retrying due to error...", flush=True)

    # If we reach here, all attempts failed to produce any text
    print(f"All {max_retries} attempts failed to generate any text", flush=True)
    return ""

def process_batch(batch_prompts, model, tokenizer, max_retries=4):
    """Process a batch of prompts with retry logic for empty responses."""
    generated_texts = []
    
    for prompt in batch_prompts:
        generated_text = process_single_prompt(prompt, model, tokenizer, max_retries)
        generated_texts.append(generated_text)
        
    return generated_texts

def process_dataset(input_file, output_file, model, tokenizer, save_interval=100, batch_size=32):
    """Process the entire dataset."""
    print(f"Processing dataset: {input_file}",flush=True)
    input_file = str(input_file)
    
    # Step 1: Load the entire dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]
    
    # Step 2: Split into rows to process and rows to keep as-is
    rows_to_process = []
    rows_to_keep = []
    
    for idx, row in enumerate(all_data):
        if "response" not in row or not row["response"]:
            # This row needs processing
            rows_to_process.append((idx, row))
        else:
            # This row already has a response, keep as-is
            rows_to_keep.append(row)
    
    print(f"Found {len(rows_to_process)} rows to process and {len(rows_to_keep)} rows to keep as-is",flush=True)
    
    # If there are no rows to process, just write out the original data
    if not rows_to_process:
        print("No empty responses found, all data will be kept as-is",flush=True)
        with open(output_file, "w", encoding="utf-8") as outfile:
            for row in all_data:
                outfile.write(json.dumps(row) + "\n")
        return
    
    # Step 3: Create a dataset for rows that need processing
    rows_to_process_data = [item[1] for item in rows_to_process]
    dataset = Dataset.from_list(rows_to_process_data)
    dataset = prepare_prompts(dataset)
    
    # Step 4: Process these rows in batches
    all_processed_rows = []  # Store all processed rows here
    original_indices = [item[0] for item in rows_to_process]
    total_batches = (len(dataset) + batch_size - 1) // batch_size

    # Clear CUDA cache before starting
    torch.cuda.empty_cache()
    
    # Create temporary directory for intermediate saves
    temp_dir = Path(output_file).parent / "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = temp_dir / f"{Path(output_file).name}.temp"
    
    for i in tqdm(range(0, len(dataset), batch_size), total=total_batches, desc="Processing empty responses"):
        try:
            batch_indices = range(i, min(i + batch_size, len(dataset)))
            batch = dataset.select(batch_indices)
            prompts = batch["prompt"]
            # Use retry logic with max 4 attempts for empty responses
            batch_responses = process_batch(prompts, model, tokenizer, max_retries=4)

            processed_rows = []
            for j, response in enumerate(batch_responses):
                idx = i + j
                if idx >= len(dataset):
                    break
                    
                example = dict(batch[j])  # Create a copy to avoid modifying the original
                example["response"] = response
                response_json = extract_json_from_text(response)

                if response_json and isinstance(response_json, dict):
                    predicted_answer = response_json.get("answer", "").strip()
                    options = [example["ans0"], example["ans1"], example["ans2"]]
                    predicted_label = match_answer_to_options(predicted_answer, options)
                    example["predicted_label"] = predicted_label
                    example["explanation"] = response_json.get("explanation", "")
                else:
                    example["predicted_label"] = -1
                    example["explanation"] = "Invalid response format"

                processed_rows.append((original_indices[idx], example))
                all_processed_rows.append((original_indices[idx], example))  # Also store in complete list

            # Save the current batch to a temporary file
            if processed_rows:
                with open(temp_file, "a", encoding="utf-8") as tempfile:
                    for _, row in processed_rows:
                        tempfile.write(json.dumps(row) + "\n")
                
                # Periodically log progress
                if len(all_processed_rows) % save_interval == 0:
                    print(f"Processed {len(all_processed_rows)} out of {len(rows_to_process)} rows",flush=True)
                    # Periodically clear CUDA cache to prevent OOM
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}",flush=True)
            continue
    
    # Step 5: Now properly combine all data and write the final output file
    final_data = all_data.copy()  # Start with a copy of the original data
    
    # Update the processed rows
    for original_idx, processed_row in all_processed_rows:
        final_data[original_idx] = processed_row
    
    # Write out all rows in the original order to the final output file
    with open(output_file, "w", encoding="utf-8") as outfile:
        for row in final_data:
            outfile.write(json.dumps(row) + "\n")
            
    # Clean up temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print(f"Successfully processed and saved all data to {output_file}",flush=True)
    print(f"Total processed rows: {len(all_processed_rows)} out of {len(rows_to_process)}",flush=True)

def process_jsonl_files(input_dir, output_dir, model_id):
    """Process all JSONL files in the input directory."""
    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer = initialize_model(model_id)
    
    for input_file in Path(input_dir).glob("*.jsonl"):
        output_file = Path(output_dir) / input_file.name
        print(f"\nProcessing {input_file} with model {model_id}...", flush=True)
        process_dataset(input_file, output_file, model, tokenizer)
        print(f"Completed processing {input_file} -> {output_file}", flush=True)

if __name__ == "__main__":
    # Enable CUDA optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    input_dir = "data"
    for model_name, model_id in models.items():
        output_dir = model_name
        process_jsonl_files(input_dir, output_dir, model_id)
