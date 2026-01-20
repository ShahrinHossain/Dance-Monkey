import json

def count(path):
    n = 0
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            n += 1
            obj = json.loads(line)
            required = ["title","body","url","date","language"]
            if any(k not in obj or obj[k] is None for k in required):
                bad += 1
    return n, bad

print("BN:", count("data/processed/bn.jsonl"))
print("EN:", count("data/processed/en.jsonl"))
