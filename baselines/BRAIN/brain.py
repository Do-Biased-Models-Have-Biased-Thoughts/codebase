import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_jsonl(file_path):
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def format_options(entry):
    if "ans0" in entry and "ans1" in entry and "ans2" in entry:
        options = [entry["ans0"], entry["ans1"], entry["ans2"]]
    elif "answer_info" in entry:
        options = []
        for i in range(3):
            key = f"ans{i}"
            if key in entry["answer_info"]:
                if isinstance(entry["answer_info"][key], list):
                    options.append(entry["answer_info"][key][0])
                else:
                    options.append(entry["answer_info"][key])
    else:
        raise ValueError("Could not find answer options in the entry")

    formatted = ""
    for i, option in enumerate(options):
        letter = chr(65 + i)  # A, B, C, ...
        formatted += f"{letter} {option}\n"

    return formatted.strip()

def prepare_case1(entry):
    options = format_options(entry)
    return f"Question: {entry['question']}\n{options}\nContext: {entry['context']}"

def prepare_case2(entry):
    options = format_options(entry)
    explanation = entry["context"]
    return f"Question: {entry['question']}\n{options}\nExplanation: {explanation}"

def get_model_probabilities(model, tokenizer, input_text, correct_answer_index):
    torch.manual_seed(args.seed)
    prompt = input_text + "\nReturn just the correct option choice (A/B or C) and nothing else"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    last_token_logits = logits[0, -1, :]
    option_letters = ['A', 'B', 'C', 'D']
    probs = []
    for letter in option_letters[:3]:
        token_id = tokenizer.encode(" " + letter, add_special_tokens=False)[0]  
        prob = F.softmax(last_token_logits, dim=0)[token_id].item()
        probs.append(prob)

    # Normalize probabilities
    probs = np.array(probs)
    probs = probs / (probs.sum() + 1e-10)  # Add small epsilon to avoid division by zero

    return probs

def compute_kl_divergence(p, q):
    """Compute KL divergence between two probability distributions."""
    epsilon = 1e-10  # Small constant to avoid log(0)
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon

    # Normalize to ensure they sum to 1
    p = p / p.sum()
    q = q / q.sum()

    # Compute KL divergence
    kl = np.sum(p * np.log(p / q))

    return kl

def compute_js_divergence(p, q):
    torch.manual_seed(args.seed)
    """Compute Jensen-Shannon divergence between two probability distributions."""
    epsilon = 1e-10  # Small constant to avoid log(0)
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon

    # Normalize to ensure they sum to 1
    p = p / p.sum()
    q = q / q.sum()

    # Calculate the mixture distribution M = (P+Q)/2
    m = (p + q) / 2.0
    js_div = 0.5 * compute_kl_divergence(p, m) + 0.5 * compute_kl_divergence(q, m)

    return js_div

def process_jsonl_entry(entry, model, tokenizer):
    torch.manual_seed(args.seed)
    """Process a single JSONL entry and compute JS divergence."""
    correct_answer_index = entry.get("label", 0)
    ex_id = entry.get("example_id", "")
    q_index = entry.get("question_index","")
    q_polarity = entry.get("question_polarity","")
    context_condition = entry.get("context_condition","")
    context = entry.get("context","")
    question = entry.get("question","")
    explanation = entry.get("explanation","")

    # Prepare inputs for both cases
    case1_input = prepare_case1(entry)
    case2_input = prepare_case2(entry)

    # Get probability distributions
    case1_probs = get_model_probabilities(model, tokenizer, case1_input, correct_answer_index)
    case2_probs = get_model_probabilities(model, tokenizer, case2_input, correct_answer_index)

    # Compute JS divergence
    js_div = compute_js_divergence(case1_probs, case2_probs)

    return {
        "example_id": ex_id,
        "question_index": q_index,
        "question_polarity": q_polarity,
        "context_condition": context_condition,
        "question": entry["question"],
        "context": context,
        "explanation": explanation,
        "options": format_options(entry.get("answer_info", {})),
        "label": entry.get("label", -100),
        "prediction": entry.get("predicted_label"),
        "correct_answer_index": correct_answer_index,
        "case1_probs": case1_probs.tolist(),
        "case2_probs": case2_probs.tolist(),
        "js_divergence": js_div
    }

def calculate_validation_stats(validation_file, model_name, max_examples=None):
    validation_examples = load_jsonl(validation_file)
    if max_examples:
        validation_examples = validation_examples[:max_examples]
    model = AutoModelForCausalLM.from_pretrained(model_name,local_files_only=True,torch_dtype="auto",device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)
    js_divergences = []
    for i, entry in enumerate(validation_examples):
        result = process_jsonl_entry(entry, model, tokenizer)
        js_divergences.append(result["js_divergence"])
    mean_js_div = np.mean(js_divergences)
    std_js_div = np.std(js_divergences)

    return mean_js_div, std_js_div, model, tokenizer

def process_test_set(test_file, mean_js_div, std_js_div, model, tokenizer, max_examples=None):
    test_examples = load_jsonl(test_file)
    if max_examples:
        test_examples = test_examples[:max_examples]
    results = []
    threshold = mean_js_div + 2.5 * std_js_div
    correct_count = 0
    for i, entry in enumerate(test_examples):
        result = process_jsonl_entry(entry, model, tokenizer)
        result["agreement"] = 1 if result["js_divergence"] <= threshold else 0
        if result["agreement"] == 0:
            correct_count += 1

        results.append(result)
    accuracy = correct_count / len(results) if results else 0

    return {
        "results": results,
        "avg_js_divergence": np.mean([r["js_divergence"] for r in results]) if results else 0,
        "js_threshold": threshold,
        "agreement_count": len(results) - correct_count,
        "disagreement_count": correct_count,
        "accuracy": accuracy
    }

def main(validation_file, test_file, model_name, max_validation=None, max_test=None, output_file=None):
    """Main function to process validation and test files and compute results."""
    print(f"Calculating validation statistics from {validation_file}")
    mean_js_div, std_js_div, model, tokenizer = calculate_validation_stats(
        validation_file, model_name, max_validation
    )

    print(f"Validation statistics: Mean JS divergence = {mean_js_div}, Std deviation = {std_js_div}")
    print(f"Threshold (mean + 2.5*std): {mean_js_div + 2.5*std_js_div}")

    print(f"Processing test set from {test_file}")
    results = process_test_set(
        test_file, mean_js_div, std_js_div, model, tokenizer, max_test
    )

    print(f"Test set results:")
    print(f"- Average JS divergence: {results['avg_js_divergence']}")
    print(f"- Examples with agreement (JS div ≤ threshold): {results['agreement_count']}")
    print(f"- Examples with disagreement (JS div > threshold): {results['disagreement_count']}")
    print(f"- Accuracy (disagreement count / total): {results['accuracy']}")

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate JS divergence and agreement metrics")
    parser.add_argument("--validation", required=True, help="Path to validation JSONL file")
    parser.add_argument("--test", required=True, help="Path to test JSONL file")
    parser.add_argument("--model_name", required=True, help="Path to the model to be loaded")
    parser.add_argument("--max_validation", type=int, default=50, help="Maximum validation examples to process")
    parser.add_argument("--max_test", type=int, default=250, help="Maximum test examples to process")
    parser.add_argument("--output", required=True, help="Output file path(.jsonl)")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--hf_token", default=None, help="HuggingFace access token (optional, can also use HF_TOKEN env variable)")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    hf_token = args.hf_token if args.hf_token else os.getenv("HF_TOKEN")
    if hf_token:
        login(hf_token)
    else:
        print("Warning: HuggingFace token not provided. Some models may not load.")

    results = main(
        args.validation,
        args.test,
        args.model_name,
        args.max_validation,
        args.max_test,
        args.output
    )
