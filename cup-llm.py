import pandas as pd
import openai
import time 

def load_and_filter_data(file_path):
    data = pd.read_json(file_path, lines=True)
    filtered_data = data[data['label'] == True]
    return filtered_data

def generate_description(code_change_seq, src_desc, dst_desc, model_name, index, feedback_loop=False, previous_feedback=None):
    if feedback_loop and previous_feedback:
        src_desc = previous_feedback  # Use feedback to adjust the source description

    prompt = f"The original method description was: '{src_desc}'. The updated method description is '{dst_desc}'. Given the following code changes {code_change_seq}, give the output description of the method in a crisp and clear manner, also targeting a high CODEBLEU metric."
    
    attempt = 0
    while attempt < 10: 
        try:
            if "turbo" in model_name:  
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are helping to generate code descriptions based on changes. If there is anything important in the src_desc, please keep it."},
                        {"role": "user", "content": prompt}
                    ]
                )
                generated_desc = response['choices'][0]['message']['content'].strip()
            else:
                response = openai.Completion.create(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=150,
                    stop=None
                )
                generated_desc = response.choices[0].text.strip()
            return generated_desc
        except openai.error.RateLimitError as e:
            print(f"Rate limit reached at index {index}. Error: {str(e)} Waiting for 20 seconds...")
            time.sleep(20)  
            attempt += 1
            print("Trying again...")
        except openai.error.OpenAIError as e:
            print(f"An OpenAI API error occurred: {str(e)}")
            break
    return "Failed to generate description after several attempts."

def write_to_jsonl(new_data, file_path):
    new_data.to_json(file_path, orient='records', lines=True)

def main():
    start_time = time.time()
    input_file_path = './dataset/cup2_dataset/trunc_valid.jsonl'
    data = load_and_filter_data(input_file_path)
    
    models = ['gpt-3.5-turbo']
    previous_feedback = None
    feedback_enabled = True  # Toggle to enable/disable feedback loop

    for model in models:
        for version in range(1, 11):  
            loop_start_time = time.time()
            output_file_path = f'./dataset/cup2_dataset/updated_descriptions_{model}_v{version}.jsonl'
            print(f"Processing with model: {model}, version {version}")
            updated_records = []
            
            for index, row in data.iterrows():
                new_desc = generate_description(row['code_change_seq'], row['src_desc'], row['dst_desc'], model, index, feedback_enabled, previous_feedback)
                previous_feedback = new_desc  # Store the last generated description as feedback
                row['new_desc'] = new_desc  
                updated_records.append(row)
            
            updated_df = pd.DataFrame(updated_records)
            write_to_jsonl(updated_df, output_file_path)
            print(f"Updated data for {model}, version {version} has been written to {output_file_path}")

            loop_end_time = time.time()
            loop_duration_seconds = loop_end_time - loop_start_time
            print(f"Time taken for version {version}: {loop_duration_seconds:.2f} seconds ({loop_duration_seconds / 60:.2f} minutes)")

    end_time = time.time()
    total_duration_seconds = end_time - start_time
    print(f"Total program execution time: {total_duration_seconds:.2f} seconds ({total_duration_seconds / 60:.2f} minutes)")

if __name__ == "__main__":
    main()
