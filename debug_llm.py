"""Quick debug script to test litellm call and inspect response."""

import sys
import litellm

sys.path.insert(0, "src")

from config import LLMConfig

cfg = LLMConfig()
print(f"Model:    {cfg.model}")
print(f"API Base: {cfg.api_base_url}")
print(
    f"API Key:  {cfg.api_key[:8]}..."
    if len(cfg.api_key) > 8
    else f"API Key: {cfg.api_key}"
)
print(f"Temp:     {cfg.temperature}")
print(f"Max Tok:  {cfg.max_tokens}")
print(f"RPM:      {cfg.rpm}")

litellm.drop_params = True

COMPLEX_PROMPT = """You are a legal text augmentation system. Given a license text fragment with placeholder variables, replace them with realistic, contextually appropriate values.

Original license: Apache-2.0
Source: scancode

Text fragment:
```
Copyright [yyyy] [name of copyright owner]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
```

Placeholders to replace:
- [yyyy] (type: year)
- [name of copyright owner] (type: owner)

Requirements:
1. Replace each placeholder with a realistic value appropriate for a software license
2. Use varied but realistic names (companies, authors, years, project names)
3. Maintain the legal tone and structure of the text
4. Ensure replaced values look natural and not obviously synthetic
5. Keep the same format as the original placeholder

Output ONLY the augmented text, nothing else. Do not add explanations or comments."""

print("\n--- Test 1: Simple prompt (max_tokens=1024) ---")
try:
    r1 = litellm.completion(
        model=cfg.model,
        messages=[{"role": "user", "content": "Say hello in 5 words."}],
        temperature=cfg.temperature,
        max_tokens=1024,
        api_base=cfg.api_base_url,
        api_key=cfg.api_key,
        custom_llm_provider="openai",
    )
    c1 = r1["choices"][0]["message"]
    print(f"  Content: {repr(c1['content'])}")
    print(f"  Usage: {r1['usage']}")
    print(f"  Finish reason: {r1['choices'][0]['finish_reason']}")
except Exception as e:
    print(f"  Error: {e}")

print("\n--- Test 2: Complex license prompt (max_tokens=1024) ---")
try:
    r2 = litellm.completion(
        model=cfg.model,
        messages=[{"role": "user", "content": COMPLEX_PROMPT}],
        temperature=cfg.temperature,
        max_tokens=1024,
        api_base=cfg.api_base_url,
        api_key=cfg.api_key,
        custom_llm_provider="openai",
    )
    c2 = r2["choices"][0]["message"]
    print(f"  Content: {repr(c2['content'])}")
    print(f"  Usage: {r2['usage']}")
    print(f"  Finish reason: {r2['choices'][0]['finish_reason']}")
except Exception as e:
    print(f"  Error: {e}")

print("\n--- Test 3: Complex license prompt (max_tokens=4096) ---")
try:
    r3 = litellm.completion(
        model=cfg.model,
        messages=[{"role": "user", "content": COMPLEX_PROMPT}],
        temperature=cfg.temperature,
        max_tokens=4096,
        api_base=cfg.api_base_url,
        api_key=cfg.api_key,
        custom_llm_provider="openai",
    )
    c3 = r3["choices"][0]["message"]
    print(f"  Content: {repr(c3['content'])}")
    print(f"  Usage: {r3['usage']}")
    print(f"  Finish reason: {r3['choices'][0]['finish_reason']}")
except Exception as e:
    print(f"  Error: {e}")


# Test 4: RareLicenseAugmenter — end-to-end verification
#
# Checks:
#   1. Augmenter calls the LLM and returns SplitFragment objects
#   2. Every fragment has source="synthetic" and the correct license_key
#   3. Similarity filter is working (all accepted variants are within bounds)
#   4. Cache is populated: a second call returns the same results instantly
#      (cache_hits == 1, wall-clock time near-zero)
#   5. get_statistics() returns sane numbers

print("\n--- Test 4: RareLicenseAugmenter ---")
try:
    from augmentation.rare_license_augmenter import (
        RareLicenseAugmenter,
        RareLicenseAugmenterConfig,
        _jaccard_ngram,
    )
    from augmentation.llm_cache import LLMCache
    from builder.hybrid_merge import DatasetEntry, DataSource

    # Minimal synthetic dataset: one rare license (0 existing fragments).
    TEST_LICENSE_KEY = "DEBUG-MIT-1.0"
    TEST_LICENSE_TEXT = (
        "Permission is hereby granted, free of charge, to any person obtaining "
        "a copy of this software and associated documentation files (the "
        '"Software"), to deal in the Software without restriction, including '
        "without limitation the rights to use, copy, modify, merge, publish, "
        "distribute, sublicense, and/or sell copies of the Software, and to "
        "permit persons to whom the Software is furnished to do so, subject to "
        "the following conditions: The above copyright notice and this permission "
        "notice shall be included in all copies or substantial portions of the "
        'Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY '
        "KIND, EXPRESS OR IMPLIED."
    )
    dataset = [
        DatasetEntry(
            license_key=TEST_LICENSE_KEY,
            short_name=TEST_LICENSE_KEY,
            full_name="Debug MIT License 1.0",
            category="permissive",
            license_text=TEST_LICENSE_TEXT,
            source=DataSource.SCANCODE,
        )
    ]

    cache = LLMCache(cache_dir="cache")
    augmenter_cfg = RareLicenseAugmenterConfig(
        threshold=1,  # augment any license with < 1 fragment
        augment_count=3,  # request 3 variants (keep test fast)
        min_similarity=0.3,
        max_similarity=0.99,
    )
    augmenter = RareLicenseAugmenter(
        config=augmenter_cfg,
        llm_config=cfg,
        cache=cache,
    )

    import time as _time

    print("  [Pass 1] Calling augment() — expects LLM call (cache miss) ...")
    t0 = _time.monotonic()
    fragments = augmenter.augment(base_dataset=dataset, existing_fragments=[])
    elapsed1 = _time.monotonic() - t0

    stats = augmenter.get_statistics(fragments)
    print(f"  Elapsed:          {elapsed1:.2f}s")
    print(f"  Fragments total:  {stats['total']}")
    print(f"  Unique licenses:  {stats['unique_licenses']}")

    if not fragments:
        print(
            "  WARNING: no fragments returned — check LLM output and similarity filter"
        )
    else:
        for i, frag in enumerate(fragments, 1):
            sim = _jaccard_ngram(TEST_LICENSE_TEXT, frag.fragment_text)
            ok_source = frag.source == "synthetic"
            ok_key = frag.license_key == TEST_LICENSE_KEY
            ok_sim = augmenter_cfg.min_similarity <= sim <= augmenter_cfg.max_similarity
            status = "OK" if (ok_source and ok_key and ok_sim) else "FAIL"
            print(
                f"  Variant {i}: [{status}]  source={frag.source!r}  "
                f"key_match={ok_key}  sim={sim:.3f}  "
                f"len={len(frag.fragment_text)}"
            )
            print(f"    Preview: {frag.fragment_text[:120].replace(chr(10), ' ')!r}")

    print("\n  [Pass 2] Calling augment() again — expects cache hit (fast) ...")
    t1 = _time.monotonic()
    fragments2 = augmenter.augment(base_dataset=dataset, existing_fragments=[])
    elapsed2 = _time.monotonic() - t1
    print(f"  Elapsed:          {elapsed2:.3f}s")
    print(f"  Same count:       {len(fragments2) == len(fragments)}")
    match = all(
        f1.fragment_text == f2.fragment_text for f1, f2 in zip(fragments, fragments2)
    )
    print(f"  Identical texts:  {match}")
    print(f"  Cache speedup:    {elapsed1 / max(elapsed2, 0.001):.0f}x")

except Exception as e:
    import traceback

    print(f"  Error: {e}")
    traceback.print_exc()
