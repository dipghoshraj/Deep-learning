# output_file = "nfs/formatted_text_to_sql_data.dev.txt"
output_file = "nfs/mental_health_data.dev.txt"


# with open(output_file, "r", encoding="utf-8") as f:
#     text = f.read()

# # Split text by whitespace to get words
# words = text.split()

# # Count the words
# total_word_count = len(words)

# print(f"Total word count in the file: {total_word_count}")


from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("YvvonM/mental_health_data", split="train[:20%]")
dataset = dataset.shuffle(seed=42) 


with open(output_file, "w", encoding="utf-8") as f:
    pass  # Just clears the file

# def format_for_token(obj):
#     return f"{obj['instruction']} schema is '{obj['input']}' query is `{obj['response']}` \n \n"

def mental_health_format_for_token(obj):
    return f" {obj['Input']} {obj['Response']} \n \n"

print(len(dataset))

with open(output_file, "a", encoding="utf-8") as f:
    for item in tqdm(dataset, desc="Writing formatted entries"):
        f.write(mental_health_format_for_token(item))

print(f"Done! {len(dataset)} entries written to {output_file}")