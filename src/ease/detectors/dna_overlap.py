"""DNA-GPT's released n-gram overlap statistic."""

import re
from collections import Counter

import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


STEMMER = PorterStemmer()
STOPWORDS = set(ENGLISH_STOP_WORDS)


def _tokens(text):
    raw = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return [
        STEMMER.stem(token) if len(token) > 3 else token
        for token in raw if token and token not in STOPWORDS
    ]


def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def official_overlap_score(target_suffix, regenerated_suffix):
    target = _tokens(target_suffix)
    generated = _tokens(regenerated_suffix)
    if not target or not generated:
        return 0.0
    weighted = 0.0
    non_zero_orders = []
    for n in range(4, 25):
        target_ngrams = _ngram_counts(target, n)
        generated_ngrams = _ngram_counts(generated, n)
        denominator = max(sum(target_ngrams.values()), 1)
        overlap = sum(
            min(count, generated_ngrams[item])
            for item, count in target_ngrams.items()
        )
        ratio = (overlap / denominator) / len(generated)
        weighted += n * np.log(n) * ratio
        if ratio != 0.0:
            non_zero_orders.append(n)
    return float(weighted / (sum(non_zero_orders) + 1e-8))
