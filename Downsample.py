import random

#Define input and output file names
input_file = "MNLI_train.jsonl"
output_file = "MNLIstar.jsonl"

#Specify the number of lines to sample
sample_size = 162865

#Count the number of lines in the file
with open(input_file, 'r') as f:
    total_lines = sum(1 for _ in f)

# Randomly select line numbers to include
selected_lines = set(random.sample(range(total_lines), sample_size))

#Write the sampled lines to the output file
with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for idx, line in enumerate(f_in):
        if idx in selected_lines:
            f_out.write(line)

print(f"Randomly sampled {sample_size} lines and saved to {output_file}")