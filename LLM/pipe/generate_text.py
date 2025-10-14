filename = "mental_health_data.txt"

with open(filename, "r", encoding="utf-8") as f:
    text = f.read()

# Split text by whitespace to get words
words = text.split()

# Count the words
total_word_count = len(words)

print(f"Total word count in the file: {total_word_count}")
