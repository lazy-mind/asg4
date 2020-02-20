import json
json_path = 'eval.json'
with open(json_path, "r") as file:
    for line in file:
        entry = json.loads(line)
        print(entry["sentiment"])