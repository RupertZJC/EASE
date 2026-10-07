"""T5 infilling perturbations used by DetectGPT and NPR.

This module keeps the released evaluation settings in a small, reusable form.
"""

import re

import numpy as np


_MASK_PATTERN = re.compile(r"<extra_id_\d+>")


def _count_masks(texts):
    return [sum(token.startswith("<extra_id_") for token in text.split()) for text in texts]


def _tokenize_and_mask(text, span_length, pct, buffer_size=1):
    tokens = text.split(" ")
    if len(tokens) <= span_length:
        return text
    marker = "<<<mask>>>"
    n_spans = int(pct * len(tokens) / (span_length + 2 * buffer_size))
    placed = 0
    while placed < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        lo, hi = max(0, start - buffer_size), min(len(tokens), end + buffer_size)
        if marker not in tokens[lo:hi]:
            tokens[start:end] = [marker]
            placed += 1
    mask_id = 0
    for index, token in enumerate(tokens):
        if token == marker:
            tokens[index] = f"<extra_id_{mask_id}>"
            mask_id += 1
    return " ".join(tokens)


def _replace_masks(masked, tokenizer, model, device, top_p=1.0, max_length=150):
    expected = _count_masks(masked)
    stop_id = tokenizer.encode(f"<extra_id_{max(expected)}>")[0]
    encoded = tokenizer(masked, return_tensors="pt", padding=True).to(device)
    output = model.generate(
        **encoded, max_length=max_length, do_sample=True, top_p=top_p,
        num_return_sequences=1, eos_token_id=stop_id,
    )
    decoded = tokenizer.batch_decode(output, skip_special_tokens=False)
    fills = [
        [piece.strip() for piece in _MASK_PATTERN.split(text.replace("<pad>", "").replace("</s>", "").strip())[1:-1]]
        for text in decoded
    ]
    results = []
    for text, pieces, count in zip(masked, fills, expected):
        tokens = text.split(" ")
        if len(pieces) < count:
            results.append("")
            continue
        for fill_id in range(count):
            marker = f"<extra_id_{fill_id}>"
            if marker in tokens:
                tokens[tokens.index(marker)] = pieces[fill_id]
        results.append(" ".join(tokens))
    return results


def perturb_texts(texts, tokenizer, model, device, span_length=2,
                  pct_words_masked=0.3, buffer_size=1, mask_top_p=1.0,
                  max_length=150, n_perturbations=1):
    expanded = [text for text in texts for _ in range(n_perturbations)]
    perturbed = [""] * len(expanded)
    pending = list(range(len(expanded)))
    for _ in range(10):
        if not pending:
            break
        masked = [
            _tokenize_and_mask(expanded[index], span_length, pct_words_masked, buffer_size)
            for index in pending
        ]
        replacements = _replace_masks(masked, tokenizer, model, device, mask_top_p, max_length)
        next_pending = []
        for index, value in zip(pending, replacements):
            perturbed[index] = value
            if not value:
                next_pending.append(index)
        pending = next_pending
    if pending:
        raise RuntimeError(f"T5 failed to fill {len(pending)} perturbations")
    return perturbed
