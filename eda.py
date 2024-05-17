import json

with open('./Data/dataset1/text.data.jsonl/text.data.jsonl', "r") as file:
    content = list(file)
    content = json.loads(content[0])
    print(content["casebody"]["data"]["opinions"][0]["text"])