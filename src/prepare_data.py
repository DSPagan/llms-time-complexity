"""
Prepare the CodeComplex dataset for the experiments.

Pipeline: raw CodeComplex (data/data.jsonl) -> deduplicate by source code ->
normalize the label of the hardest class -> stratified 90/10 train/test split.

Starting from the original, publicly available CodeComplex snapshot (rather than a
pre-processed file) keeps the whole pipeline reproducible. Run as a script to
regenerate data/train_data.jsonl and data/test_data.jsonl:

    python src/prepare_data.py
"""

import json
import random
from collections import defaultdict

# CodeComplex labels the hardest class "np"; the thesis and prompts call it
# "exponential". Normalize so every downstream component uses the same name.
LABEL_ALIASES = {"np": "exponential"}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def deduplicate(records, field="src"):
    """Keep the first record for each distinct value of ``field``."""
    seen = set()
    unique = []
    for obj in records:
        value = obj.get(field)
        if value not in seen:
            seen.add(value)
            unique.append(obj)
    return unique


def normalize_labels(records, field="complexity", aliases=LABEL_ALIASES):
    for obj in records:
        obj[field] = aliases.get(obj.get(field), obj.get(field))
    return records


def stratified_split(records, field="complexity", test_size=0.1, seed=42,
                     min_test_per_class=1):
    """Split into train/test keeping the class proportions in both sets."""
    groups = defaultdict(list)
    for obj in records:
        groups[obj.get(field)].append(obj)

    rng = random.Random(seed)
    train, test = [], []
    for label, items in sorted(groups.items()):
        rng.shuffle(items)
        n = len(items)
        n_test = int(n * test_size)
        if n >= 2:
            n_test = min(max(min_test_per_class, n_test), n - 1)
        else:
            n_test = 0
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def load_clean(data_path="data/data.jsonl"):
    """Load the raw dataset, de-duplicated and with labels normalized."""
    records = read_jsonl(data_path)
    records = deduplicate(records, field="src")
    return normalize_labels(records)


def stratified_folds(records, field="complexity", k=5, seed=42):
    """Split records into k folds with balanced class proportions (for cross-validation)."""
    groups = defaultdict(list)
    for obj in records:
        groups[obj.get(field)].append(obj)
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for label, items in sorted(groups.items()):
        items = list(items)
        rng.shuffle(items)
        for i, obj in enumerate(items):
            folds[i % k].append(obj)
    for f in folds:
        rng.shuffle(f)
    return folds


def prepare(
    data_path="data/data.jsonl",
    train_path="data/train_data.jsonl",
    test_path="data/test_data.jsonl",
    test_size=0.1,
    seed=42,
):
    records = read_jsonl(data_path)
    print(f"Loaded {len(records)} records from {data_path}")

    records = deduplicate(records, field="src")
    print(f"After de-duplication by 'src': {len(records)} records")

    records = normalize_labels(records)

    train, test = stratified_split(records, test_size=test_size, seed=seed)

    counts = defaultdict(lambda: [0, 0])
    for obj in train:
        counts[obj["complexity"]][0] += 1
    for obj in test:
        counts[obj["complexity"]][1] += 1
    for label in sorted(counts):
        tr, te = counts[label]
        print(f"  {label:<12} train={tr:<5} test={te}")

    write_jsonl(train, train_path)
    write_jsonl(test, test_path)
    print(f"\nTrain: {len(train)}  ->  {train_path}")
    print(f"Test:  {len(test)}  ->  {test_path}")
    print(f"Total: {len(train) + len(test)}")


if __name__ == "__main__":
    prepare()
