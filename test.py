import json
from collections import Counter

def accuracy(pred_path, truth_path):
    truth = {}
    with open(truth_path) as f:
        for line in f:
            r = json.loads(line)
            truth[r["image_path"]] = r["emotion_label"]
    correct = total = 0
    per_class_correct = Counter()
    per_class_total = Counter()
    with open(pred_path) as f:
        for line in f:
            r = json.loads(line)
            if r["image_path"] not in truth:
                continue
            t = truth[r["image_path"]]
            p = r["emotion_label"]
            per_class_total[t] += 1
            if p == t:
                correct += 1
                per_class_correct[t] += 1
            total += 1
    print(f"Overall: {correct}/{total} = {correct/total:.3f}")
    for c in ["neutral","happy","angry","sad","surprise"]:
        n = per_class_total[c]
        k = per_class_correct[c]
        print(f"  {c}: {k}/{n} = {k/n:.3f}" if n else f"  {c}: N/A (0 samples)")

accuracy("data/predictions_train.jsonl", "data/phase1_vist_train_new.jsonl")
accuracy("data/predictions_test.jsonl",  "data/phase1_vist_test_new.jsonl")




# class imbalance ratio
neutral, happy, angry, sad, surprise = 1172, 656, 16, 52, 39
ratio = neutral / angry   # ~73x
print(f"Class imbalance: majority/minority = {ratio:.1f}×")



