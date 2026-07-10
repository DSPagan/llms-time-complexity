"""
Prepare the CodeComplex dataset for the cross-validation experiments.

Pipeline: raw CodeComplex (data/data.jsonl) -> deduplicate by source code ->
normalize the label of the hardest class -> stratified k folds.

Starting from the original, publicly available CodeComplex snapshot (rather than a
pre-processed file) keeps the whole pipeline reproducible. The notebooks call
``load_clean()`` and ``stratified_folds()`` directly.
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
