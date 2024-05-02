import json
import warnings
from codebleu import calc_codebleu

warnings.filterwarnings('ignore')

def load_jsonl_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            yield json.loads(line)

def average_metrics(scores):
    avg_scores = {
        'codebleu': 0,
        'ngram_match_score': 0,
        'weighted_ngram_match_score': 0,
        'syntax_match_score': 0,
        'dataflow_match_score': 0
    }
    if scores:
        for score in scores:
            for key in avg_scores:
                avg_scores[key] += score[key]
        for key in avg_scores:
            avg_scores[key] /= len(scores)
    return avg_scores

def evaluate_codebleu(data_file, evaluate_new_desc=True):
    src_desc_scores = []
    dst_desc_scores = []
    new_desc_scores = []

    for row in load_jsonl_file(data_file):
        dst_method = row['dst_method']
        if not src_desc_scores: 
            src_score = calc_codebleu([row['src_desc']], [dst_method], lang="java", weights=(0.25, 0.25, 0.25, 0.25))
            dst_score = calc_codebleu([row['dst_desc']], [dst_method], lang="java", weights=(0.25, 0.25, 0.25, 0.25))
            src_desc_scores.append(src_score)
            dst_desc_scores.append(dst_score)
        if evaluate_new_desc:
            new_score = calc_codebleu([row['new_desc']], [dst_method], lang="java", weights=(0.25, 0.25, 0.25, 0.25))
            new_desc_scores.append(new_score)

    avg_src_desc = average_metrics(src_desc_scores)
    avg_dst_desc = average_metrics(dst_desc_scores)
    avg_new_desc = average_metrics(new_desc_scores) if evaluate_new_desc else None

    return avg_src_desc, avg_dst_desc, avg_new_desc

def run_evaluations():
    base_path = './dataset/cup2_dataset/updated_descriptions_gpt-3.5-turbo'
    versions = [f"{base_path}_v{i}.jsonl" for i in range(1, 11)] 
    overall_new_desc_scores = []
    
    avg_src_desc, avg_dst_desc, _ = evaluate_codebleu(versions[0], evaluate_new_desc=False)
    print("Average CodeBLEU Scores for src_desc and dst_desc (common across all files):")
    print(f"  src_desc CodeBLEU: {avg_src_desc['codebleu']:.4f}")
    print(f"  dst_desc CodeBLEU: {avg_dst_desc['codebleu']:.4f}")
    print("----------------------------------------------------")

    for version_path in versions:
        _, _, avg_new_desc = evaluate_codebleu(version_path)
        overall_new_desc_scores.append(avg_new_desc)
        print(f"Average CodeBLEU Scores for new_desc in {version_path.split('_')[-1].split('.')[0]}:")
        print(f"  CodeBLEU: {avg_new_desc['codebleu']:.4f}")
        print("----------------------------------------------------")

    
    overall_avg_new_desc = average_metrics(overall_new_desc_scores)
    avg_src_desc, avg_dst_desc, _ = evaluate_codebleu(versions[0], evaluate_new_desc=False)
    print("Average CodeBLEU Scores for src_desc and dst_desc (common across all files):")
    print(f"  src_desc CodeBLEU: {avg_src_desc['codebleu']:.4f}")
    print(f"  dst_desc CodeBLEU: {avg_dst_desc['codebleu']:.4f}")
    print("----------------------------------------------------")
    print("Overall Average CodeBLEU Scores for new_desc across all versions:")
    print(f"  CodeBLEU: {overall_avg_new_desc['codebleu']:.4f}")

run_evaluations()
