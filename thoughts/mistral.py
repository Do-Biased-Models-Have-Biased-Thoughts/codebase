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
login(token="your token here")

# Define model configurations
models = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3"
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
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx + 1]
            return json.loads(json_str)
    except:
        pass
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

def process_batch(batch_prompts, model, tokenizer):
    """Process a batch of prompts."""
    try:
        inputs = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1500
        ).to(model.device)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            outputs = model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1
            )

        generated_texts = tokenizer.batch_decode(
            [output[inputs.input_ids.shape[1]:] for output in outputs],
            skip_special_tokens=True
        )
        
        return generated_texts

    except Exception as e:
        print(f"Error generating batch responses: {str(e)}", flush=True)
        return [""] * len(batch_prompts)

def process_dataset(input_file, output_file, model, tokenizer, save_interval=1000, batch_size=32):
    """Process the entire dataset."""
    print(f"Processing dataset: {input_file}",flush=True)
    input_file = str(input_file)
    dataset = Dataset.from_json(input_file)
    dataset = prepare_prompts(dataset)

    if os.path.exists(output_file):
        os.remove(output_file)

    processed_rows = []
    total_batches = (len(dataset) + batch_size - 1) // batch_size

    # Clear CUDA cache before starting
    torch.cuda.empty_cache()
    
    for i in tqdm(range(0, len(dataset), batch_size), total=total_batches, desc="Processing batches"):
        try:
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            prompts = batch["prompt"]
            batch_responses = process_batch(prompts, model, tokenizer)

            for idx, response in enumerate(batch_responses):
                example = dict(batch[idx])  # Create a copy to avoid modifying the original
                example["response"]=response
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

                processed_rows.append(example)

            if len(processed_rows) >= save_interval:
                with open(output_file, "a", encoding="utf-8") as outfile:
                    for row in processed_rows:
                        outfile.write(json.dumps(row) + "\n")
                processed_rows = []
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}",flush=True)
            continue

    if processed_rows:
        with open(output_file, "a", encoding="utf-8") as outfile:
            for row in processed_rows:
                outfile.write(json.dumps(row) + "\n")

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
    
    input_dir = "Data" # BBQ data
    for model_name, model_id in models.items():
        output_dir = model_name
        process_jsonl_files(input_dir, output_dir, model_id)