import datasets
import json

#Load SNLI and ANLI datasets

snli = datasets.load_dataset('snli')
#anli = datasets.load_dataset('anli')


# Remove SNLI examples with no label
snli_train = snli['train'].filter(lambda ex: ex['label'] != -1)

# Remove 'uid' and 'reason' columns from ANLI test datasets
def remove_columns(dataset, columns):
    return dataset.remove_columns(columns)


combined_dataset = snli_train

# Convert combined dataset to JSONL format
output_file = "SNLI_train.jsonl"
with open(output_file, "w") as f:
    for example in combined_dataset:
        json.dump(example, f)
        f.write("\n")

print(f"Combined dataset saved to {output_file}")