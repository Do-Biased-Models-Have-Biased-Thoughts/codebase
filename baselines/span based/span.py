import json
import glob
import os
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze bias in model explanations')
    parser.add_argument('--validation_folder', type=str, required=True, 
                        help='Path to the validation folder containing JSONL files')
    parser.add_argument('--test_folder', type=str, required=True, 
                        help='Path to the test folder containing JSONL files')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Base output directory')
    parser.add_argument('--seed', type=int, required=True, 
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def load_jsonl_files(folder_path):
    """Load all .jsonl files in a folder and return as a list of dictionaries."""
    data = []
    for filename in glob.glob(os.path.join(folder_path, '*.jsonl')):
        file_data = []
        with open(filename, 'r') as f:
            for line in f:
                file_data.append(json.loads(line.strip()))
        base_filename = os.path.basename(filename)
        data.append((base_filename, file_data))
    
    return data

def extract_answer_from_label(item):
    label = item['label']
    if label == 0:
        return item['ans0']
    elif label == 1:
        return item['ans1']
    elif label == 2:
        return item['ans2']
    return ""

def calculate_similarities(data_list, model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    all_similarities = []
    file_results = []
    
    for filename, items in data_list:
        file_similarities = []
        
        for item in items:
            question = item['question']
            answer = extract_answer_from_label(item)
            context = item['context']
            explanation = item['explanation']
            qa_text = f"Question :{question}\nContext :{context}\nAnswer :{answer}"
            
            # Calculate embeddings
            qa_embedding = model.encode([qa_text], convert_to_tensor=True, device=device)
            explanation_embedding = model.encode([explanation], convert_to_tensor=True, device=device)
            qa_embedding = qa_embedding.cpu().numpy()
            explanation_embedding = explanation_embedding.cpu().numpy()
            similarity = cosine_similarity(qa_embedding, explanation_embedding)[0][0]
            
            # Check if prediction matches label
            label_match = item['predicted_label'] == item['label']
            result = {
                'example_id': item['example_id'],
                'question_index': item.get('question_index', ''),
                'category': item.get('category', ''),
                'question': question,
                'answer': answer,
                'explanation': explanation,
                'similarity': similarity,
                'label': item['label'],
                'predicted_label': item['predicted_label'],
                'label_match': label_match,
                'source_file': filename
            }
            
            file_similarities.append(result)
            all_similarities.append(result)
        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_similarities.csv")
        pd.DataFrame(file_similarities).to_csv(output_file, index=False)
        file_results.append((filename, file_similarities))
    
    return all_similarities, file_results

def calculate_threshold(validation_similarities):
    """Calculate threshold based on mean and standard deviation."""
    similarities = [item['similarity'] for item in validation_similarities]
    mean = np.mean(similarities)
    std_dev = np.std(similarities)
    threshold = mean - 1.5 * std_dev
    
    return {
        'mean': mean,
        'std_dev': std_dev,
        'threshold': threshold
    }

def classify_examples(test_similarities, threshold):
    """Classify examples as biased or unbiased based on threshold and label comparison."""
    results = []
    
    for item in test_similarities:
        # Check for similarity threshold
        similarity_biased = item['similarity'] < threshold
        
        # Check for label match (unbiased if predicted_label = label)
        label_biased = not item['label_match']
        
        # An example is considered biased if EITHER condition is met
        # It's unbiased only if it passes BOTH checks
        is_biased = similarity_biased or label_biased
        
        results.append({
            **item,
            'similarity_biased': similarity_biased,
            'label_biased': label_biased,
            'classification': "biased" if is_biased else "unbiased",
            'bias_reason': get_bias_reason(similarity_biased, label_biased)
        })
    
    return results

def get_bias_reason(similarity_biased, label_biased):
    """Return the reason for bias classification."""
    if similarity_biased and label_biased:
        return "Both similarity and label mismatch"
    elif similarity_biased:
        return "Low explanation-question similarity"
    elif label_biased:
        return "Predicted label does not match ground truth"
    else:
        return "Not biased"

def generate_file_statistics(classified_results, output_dir):
    """Generate statistics for each file in the dataset."""
    file_groups = {}
    for item in classified_results:
        filename = item['source_file']
        if filename not in file_groups:
            file_groups[filename] = []
        file_groups[filename].append(item)
    file_stats = []
    for filename, items in file_groups.items():
        total_count = len(items)
        biased_count = sum(1 for item in items if item['classification'] == 'biased')
        similarity_biased = sum(1 for item in items if item['similarity_biased'])
        label_biased = sum(1 for item in items if item['label_biased'])
        both_biased = sum(1 for item in items if item['similarity_biased'] and item['label_biased'])
        biased_percentage = (biased_count / total_count) * 100 if total_count > 0 else 0
        
        file_stat = {
            'filename': filename,
            'total_examples': total_count,
            'biased_examples': biased_count,
            'unbiased_examples': total_count - biased_count,
            'biased_percentage': biased_percentage,
            'similarity_biased_count': similarity_biased,
            'label_biased_count': label_biased,
            'both_biased_count': both_biased
        }
        file_stats.append(file_stat)
    output_file = os.path.join(output_dir, 'file_statistics.csv')
    pd.DataFrame(file_stats).to_csv(output_file, index=False)
    
    return file_stats

def main():
    # Parse command line arguments
    args = parse_arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
    os.makedirs(output_dir, exist_ok=True)
    validation_output_dir = os.path.join(output_dir, 'validation_stats')
    test_output_dir = os.path.join(output_dir, 'test_stats')
    print(f"Using seed: {args.seed}")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device=device)
    
    # 1. Load validation and test data
    validation_data = load_jsonl_files(args.validation_folder)
    test_data = load_jsonl_files(args.test_folder)
    
    # 2. Calculate similarities for validation set
    validation_similarities, validation_file_results = calculate_similarities(validation_data, model, validation_output_dir)
    
    # 3. Determine threshold from validation data
    threshold_stats = calculate_threshold(validation_similarities)
    print(f"Validation statistics: Mean={threshold_stats['mean']:.4f}, StdDev={threshold_stats['std_dev']:.4f}")
    print(f"Bias threshold: {threshold_stats['threshold']:.4f}")
    
    # 4. Calculate similarities for test set
    test_similarities, test_file_results = calculate_similarities(test_data, model, test_output_dir)
    
    # 5. Classify test examples
    classified_results = classify_examples(test_similarities, threshold_stats['threshold'])
    
    # 6. Save results
    classification_file = os.path.join(output_dir, 'classification_results.csv')
    pd.DataFrame(classified_results).to_csv(classification_file, index=False)
    
    # 7. Save threshold statistics
    threshold_file = os.path.join(output_dir, 'threshold_stats.csv')
    pd.DataFrame([threshold_stats]).to_csv(threshold_file, index=False)
    
    # 8. Generate and save per-file statistics
    file_stats = generate_file_statistics(classified_results, output_dir)
    
    # 9. Calculate overall statistics
    biased_count = sum(1 for item in classified_results if item['classification'] == 'biased')
    similarity_biased_count = sum(1 for item in classified_results if item['similarity_biased'])
    label_biased_count = sum(1 for item in classified_results if item['label_biased'])
    both_biased_count = sum(1 for item in classified_results if item['similarity_biased'] and item['label_biased'])
    total_count = len(classified_results)
    
    # Calculate percentages
    biased_percentage = (biased_count / total_count) * 100 if total_count > 0 else 0
    similarity_biased_percentage = (similarity_biased_count / total_count) * 100 if total_count > 0 else 0
    label_biased_percentage = (label_biased_count / total_count) * 100 if total_count > 0 else 0
    both_biased_percentage = (both_biased_count / total_count) * 100 if total_count > 0 else 0
    
    summary = {
        'seed': args.seed,
        'total_examples': total_count,
        'biased_examples': biased_count,
        'unbiased_examples': total_count - biased_count,
        'biased_percentage': biased_percentage,
        'similarity_biased_count': similarity_biased_count,
        'similarity_biased_percentage': similarity_biased_percentage,
        'label_biased_count': label_biased_count,
        'label_biased_percentage': label_biased_percentage,
        'both_biased_count': both_biased_count,
        'both_biased_percentage': both_biased_percentage
    }
    
    summary_file = os.path.join(output_dir, 'summary_stats.csv')
    pd.DataFrame([summary]).to_csv(summary_file, index=False)
    
    # 10. Generate bias reason summary
    bias_reasons = {}
    for item in classified_results:
        reason = item['bias_reason']
        bias_reasons[reason] = bias_reasons.get(reason, 0) + 1
    
    bias_reasons_df = pd.DataFrame({
        'Reason': list(bias_reasons.keys()),
        'Count': list(bias_reasons.values()),
        'Percentage': [(count / total_count) * 100 for count in bias_reasons.values()]
    })
    
    bias_reasons_file = os.path.join(output_dir, 'bias_reasons.csv')
    bias_reasons_df.to_csv(bias_reasons_file, index=False)
    
    # Print summary
    print(f"\nSummary Statistics (Seed {args.seed}):")
    print(f"Total examples: {total_count}")
    print(f"Biased examples: {biased_count} ({biased_percentage:.2f}%)")
    print(f"  - Due to similarity threshold: {similarity_biased_count} ({similarity_biased_percentage:.2f}%)")
    print(f"  - Due to label mismatch: {label_biased_count} ({label_biased_percentage:.2f}%)")
    print(f"  - Due to both reasons: {both_biased_count} ({both_biased_percentage:.2f}%)")
    print(f"Unbiased examples: {total_count - biased_count} ({100 - biased_percentage:.2f}%)")
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()