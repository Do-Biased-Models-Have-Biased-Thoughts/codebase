import json
import os
from pathlib import Path
from transformers import pipeline
from datasets import Dataset
import torch
from huggingface_hub import login
import re

# Login using token from environment variable
login(token="your-token-here")
model_id = "google/gemma-2-2b-it"

# Initialize the pipeline
pipe = pipeline(
    "text-generation",
    model=model_id,
    device="cuda",
    model_kwargs={
        "torch_dtype": torch.bfloat16,
    }
)

def prepare_prompts(dataset):
    """Prepare prompts for the dataset."""
    prompt_template = '''Given context: {context}
    Question: {question}
    Options: {options}
    Given the context, question, and options, your job is to answer the question by selecting one option from the given options. Provide an explanation for the choice. Output in JSON format with "answer" and "explanation" as keys.'''

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
    """Extract JSON data from text output."""
    try:
        json_text = None  # Ensure json_text is defined
        # Locate the JSON text
        if "```json" in text:
            start_index = text.find("```json") + len("```json")
            end_index = text.find("```", start_index)
            if end_index != -1:
                json_text = text[start_index:end_index].strip()
        else:
            start_index = text.find("{")
            end_index = text.rfind("}")
            if start_index != -1 and end_index != -1:
                json_text = text[start_index:end_index + 1]
        
        if not json_text:
            return None  # No valid JSON found

        # Attempt to parse the JSON
        return json.loads(json_text)

    except json.JSONDecodeError as e:
        # Handle JSON decoding errors gracefully
        print(f"JSON decoding failed. Attempting to sanitize explanation. Error: {str(e)}",flush=True)
        
        if json_text and '"explanation"' in json_text:
            # Find the start and end of the explanation
            def remove_inner_quotes(match):
                # Extract the full explanation
                full_explanation = match.group(1)
                
                # Remove all inner quotes
                cleaned_explanation = re.sub(r'"', '', full_explanation)
                
                # Return the full replacement, including quotes
                return f'"explanation": "{cleaned_explanation}"'
            
            # Use regex to find and replace the explanation section
            json_text = re.sub(
                r'"explanation"\s*:\s*"(.*?)"(?=\s*})', 
                remove_inner_quotes, 
                json_text, 
                flags=re.DOTALL
            )
        
        try:
            # Re-attempt parsing the sanitized JSON
            return json.loads(json_text)
        except json.JSONDecodeError as final_e:
            print(f"Failed to parse JSON after sanitizing explanation: {text}\nError: {str(final_e)}",flush=True)
            return None



def process_batch(batch, pipe):
    """Process a batch of prompts and return responses."""
    prompts = batch["prompt"]
    try:
        # Generate responses in a batch
        responses = pipe(prompts, max_new_tokens=1024)
        # Extract generated text from the pipeline output
        return [response[0]["generated_text"].strip() for response in responses]
    except Exception as e:
        print(f"Error generating batch responses: {str(e)}",flush=True)
        return [""] * len(prompts)

def match_answer_to_options(predicted_answer, options):
    predicted_answer = predicted_answer.lower()
    options = [option.lower() for option in options]
    for index, option in enumerate(options):
        if predicted_answer in option or option in predicted_answer:
            return index
    return -1

def process_dataset(input_file, output_file, pipe, save_interval=20, batch_size=8):
    """Process an input JSONL file and save the processed data."""
    input_file = str(input_file)  # Ensure the path is a string
    dataset = Dataset.from_json(input_file)  # Load input JSONL file as a Dataset
    dataset = prepare_prompts(dataset)

    processed_rows = []

    for i in range(0, len(dataset), batch_size):
        # Select a batch of examples
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))

        # Process the batch
        prompts = batch["prompt"]
        batch_responses = process_batch({"prompt": prompts}, pipe)

        for idx, response in enumerate(batch_responses):
            example = batch[idx]
            example["response"] = response
            # Parse the response
            response_json = extract_json_from_text(response)

            if response_json and isinstance(response_json, dict):
                predicted_answer = response_json.get("answer", "").strip()
                options = [example["ans0"], example["ans1"], example["ans2"]]

                # Match predicted answer to options
                predicted_label = match_answer_to_options(predicted_answer, options)

                example["predicted_label"] = predicted_label
                example["explanation"] = response_json.get("explanation", "")
            else:
                example["predicted_label"] = -1
                example["explanation"] = "Invalid response format"

            processed_rows.append(example)

            # Save intermediate results every `save_interval` rows
            if len(processed_rows) >= save_interval:
                with open(output_file, "a", encoding="utf-8") as outfile:
                    for row in processed_rows:
                        outfile.write(json.dumps(row) + "\n")
                print(f"Saved {len(processed_rows)} rows to {output_file}",flush=True)
                processed_rows = []  # Clear the buffer

    # Save any remaining rows
    if processed_rows:
        with open(output_file, "a", encoding="utf-8") as outfile:
            for row in processed_rows:
                outfile.write(json.dumps(row) + "\n")
        print(f"Saved remaining {len(processed_rows)} rows to {output_file}",flush=True)

def process_jsonl_files(input_dir, output_dir):
    """Process all JSONL files in the input directory."""
    os.makedirs(output_dir, exist_ok=True)
    for input_file in Path(input_dir).glob("*.jsonl"):
        output_file = Path(output_dir) / input_file.name
        print(f"Processing {input_file}...",flush=True)
        process_dataset(input_file, output_file, pipe)
        print(f"Completed processing {input_file} -> {output_file}",flush=True)

if __name__ == "__main__":
    input_dir = "/gemma/data"
    output_dir = "/gemma/output"
    process_jsonl_files(input_dir, output_dir)
