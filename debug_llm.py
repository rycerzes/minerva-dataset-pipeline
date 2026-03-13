"""Quick debug script to test litellm call and inspect response."""
import sys
sys.path.insert(0, "src")

from config import LLMConfig

cfg = LLMConfig()
print(f"Model:    {cfg.model}")
print(f"API Base: {cfg.api_base_url}")
print(f"API Key:  {cfg.api_key[:8]}..." if len(cfg.api_key) > 8 else f"API Key: {cfg.api_key}")
print(f"Temp:     {cfg.temperature}")
print(f"Max Tok:  {cfg.max_tokens}")
print(f"RPM:      {cfg.rpm}")

import litellm
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
