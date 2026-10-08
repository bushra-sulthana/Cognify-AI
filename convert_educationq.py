import json

input_file = "knowledge_base/education/educationq_full.json"
output_file = "knowledge_base/education/educationq.txt"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "w", encoding="utf-8") as f:
    for i, item in enumerate(data, 1):
        f.write(f"Question {i}\n")
        f.write(f"Category: {item.get('category', '')}\n")
        f.write(f"Difficulty: {item.get('difficulty', '')}\n")
        f.write(f"Question: {item.get('question', '')}\n")
        f.write(f"Answer: {item.get('answer', '')}\n")
        f.write(f"Explanation: {item.get('explanation', '')}\n")
        f.write("\n" + "=" * 80 + "\n\n")

print(f"Converted {len(data)} records.")
print(f"Created: {output_file}")