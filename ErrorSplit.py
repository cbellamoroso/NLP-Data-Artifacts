import json
import random
import pandas as pd

# File paths
input_file_1 = r".\ModelEvals\SNLIEval\BaseNLI_eval.jsonl"
input_file_2 = r".\ModelEvals\SNLI_MLNI_ANLI\eval_predictions.jsonl"
output_file_correct_second = "Correct_Second_Mistake_First.jsonl"
output_file_mistake_second = "Mistake_Second_Correct_First.jsonl"
output_file_both_incorrect = "Both_Incorrect.jsonl"
output_file_incorrect_only = "Incorrect_Only.jsonl"  #
final_excel_correct_second = "Correct_Second_Mistake_First.xlsx"
final_excel_mistake_second = "Mistake_Second_Correct_First.xlsx"
final_excel_both_incorrect = "Both_Incorrect.xlsx"
final_excel_incorrect_only = "Incorrect_Only.xlsx"


#read a JSONL file
def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


# write a JSONL file
def write_jsonl(file_path, data):
    with open(file_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')


#Read the JSONL files
file_1_data = read_jsonl(input_file_1)
file_2_data = read_jsonl(input_file_2)

#Compare the files and filter differences
correct_second_mistake_first = []
mistake_second_correct_first = []
both_incorrect = []
both_correct = []

for entry_1, entry_2 in zip(file_1_data, file_2_data):
    true_label = entry_1["label"]
    pred_1 = entry_1["predicted_label"]
    pred_2 = entry_2["predicted_label"]

    # Check conditions
    if pred_2 == true_label and pred_1 != true_label:
        correct_second_mistake_first.append(entry_2)
    elif pred_2 != true_label and pred_1 == true_label:
        mistake_second_correct_first.append(entry_2)
    elif pred_2 != true_label and pred_1 != true_label:
        both_incorrect.append(entry_2)

    #Add to incorrect predictions if pred_1 is incorrect
    if pred_1 == true_label & pred_2 == true_label:
        both_correct.append(entry_1)

#Randomly sample 100 entries from each
random.seed(42)  #For reproducibility
correct_second_mistake_first_sample = random.sample(correct_second_mistake_first,
                                                    min(len(correct_second_mistake_first), len(correct_second_mistake_first)))
mistake_second_correct_first_sample = random.sample(mistake_second_correct_first,
                                                    min(len(mistake_second_correct_first), len(mistake_second_correct_first)))
both_incorrect_sample = random.sample(both_incorrect, min(100, len(both_incorrect)))
both_correct_sample = random.sample(incorrect_only, min(100, len(both_correct)))

# Step 4: Write JSONL files
write_jsonl(output_file_correct_second, correct_second_mistake_first_sample)
write_jsonl(output_file_mistake_second, mistake_second_correct_first_sample)
write_jsonl(output_file_both_incorrect, both_incorrect_sample)
write_jsonl(output_file_both_correct, incorrect_only_sample)  # New file


#Convert JSONL to Excel
def jsonl_to_excel(jsonl_file, excel_file):
    with open(jsonl_file, 'r') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    df.to_excel(excel_file, index=False)


jsonl_to_excel(output_file_correct_second, final_excel_correct_second)
jsonl_to_excel(output_file_mistake_second, final_excel_mistake_second)
jsonl_to_excel(output_file_both_incorrect, final_excel_both_incorrect)
jsonl_to_excel(output_file_incorrect_only, final_excel_incorrect_only)

print("Processing complete! Excel files saved:")
print(f"- {final_excel_correct_second}")
print(f"- {final_excel_mistake_second}")
print(f"- {final_excel_both_incorrect}")
print(f"- {final_excel_incorrect_only}")
