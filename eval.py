import json
import warnings
from codebleu import calc_codebleu

warnings.filterwarnings('ignore')

def load_jsonl_file(filepath):
    """
    Loads a JSONL file and yields each line as a JSON object.

    Args:
        filepath (str): The path to the JSONL file.

    Yields:
        dict: A JSON object representing each line in the file.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            yield json.loads(line)

def average_metrics(scores):
    """
    Calculate the average scores for different metrics.

    Args:
        scores (list): A list of dictionaries containing scores for different metrics.

    Returns:
        dict: A dictionary containing the average scores for each metric.
    """
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
    """
    Evaluate the CodeBLEU scores for the given data file.

    Args:
        data_file (str): The path to the data file.
        evaluate_new_desc (bool, optional): Whether to evaluate the CodeBLEU score for the new description. 
        Defaults to True.

    Returns:
        tuple: A tuple containing the average CodeBLEU scores for the source description, destination description,
        and new description (if evaluate_new_desc is True). If evaluate_new_desc is False, the third element
        of the tuple will be None.
    """
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
    """
    Runs evaluations for the cup-llm project.

    This function evaluates the CodeBLEU scores for different versions of descriptions in the cup-llm project.
    It prints the average CodeBLEU scores for src_desc and dst_desc (common across all files),
    as well as the average CodeBLEU scores for new_desc in each version.

    Parameters:
        None

    Returns:
        None
    """
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
