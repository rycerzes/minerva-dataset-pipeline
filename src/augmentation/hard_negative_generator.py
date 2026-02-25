from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Literal
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from ..config import LLMConfig, RateLimiter
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import LLMConfig, RateLimiter


class HardNegativeSample(BaseModel):
    model_config = {"protected_namespaces": ()}

    text: str
    negative_type: str
    generation_method: Literal["llm_generated"]
    source_license: Optional[str] = None
    llm_model_used: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class HardNegativeGeneratorError(Exception):
    pass


class HardNegativeGenerator:
    # Category weights for distribution across negative types
    LICENSE_DISCUSSION_WEIGHT: float = 0.3
    TODO_FIXME_WEIGHT: float = 0.3
    COMMENTED_CODE_WEIGHT: float = 0.2
    COPYRIGHT_DISCUSSION_WEIGHT: float = 0.2

    MAX_RETRIES: int = 3
    RETRY_BACKOFF_BASE: float = 2.0

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        samples_per_category: int = 5,
    ):
        self.config = config or LLMConfig()
        self.samples_per_category = samples_per_category
        self._rate_limiter = RateLimiter(self.config.rpm)
        self._llm_client = None

    def _get_llm_client(self):
        import litellm

        litellm.drop_params = True
        self._llm_client = litellm
        return self._llm_client

    def _build_license_discussion_prompt(
        self, license_key: str, context: Optional[str] = None
    ) -> str:
        prompt = f"""You are a developer discussing software licenses in code comments. Generate realistic, natural developer conversations about licenses that are NOT legally binding license texts.

Generate {self.samples_per_category} varied examples of developer discussions about the "{license_key}" license.

Requirements:
1. Write in the style of informal developer comments (not legal language)
2. Questions, opinions, uncertainty about licenses are appropriate
3. Include mentions of: TODO, FIXME, uncertainty, opinions, questions
4. Do NOT include actual license text - only discussions, questions, opinions
5. Make it look like authentic developer communication in source code comments
6. Include realistic typos, casual language
7. Mix of single-line comments (//, #) and multi-line comments (/* */)
8. Each example should be 1-3 sentences
{f"(Optional context: {context})" if context else ""}

Output each example on a new line, nothing else."""
        return prompt

    def _build_todo_fixme_prompt(self, license_key: str) -> str:
        prompt = f"""Generate {self.samples_per_category} realistic TODO and FIXME comments related to software licensing tasks for the "{license_key}" license.

Requirements:
1. Write as authentic developer TODO/FIXME comments
2. Include practical tasks: updating headers, checking compliance, adding licenses
3. Include common developer frustrations around license management
4. Mix of TODO and FIXME prefixes
5. Include realistic file paths, function names when appropriate
6. Each should be 1-2 lines

Output each example on a new line, nothing else."""
        return prompt

    def _build_commented_code_prompt(self, license_key: str) -> str:
        prompt = f"""Generate {self.samples_per_category} examples of code that was commented out or disabled due to licensing issues for code related to "{license_key}" license.

Requirements:
1. Show commented-out code blocks with license-related explanations
2. Include placeholders for removed proprietary code
3. Show code that checks or verifies licenses
4. Include both single-line (#, //) and multi-line (/* */) comment styles
5. Show realistic function names, variables related to licensing
6. Include brief comments explaining why code was disabled

Output each example on a new line, nothing else."""
        return prompt

    def _build_copyright_discussion_prompt(self, license_key: str) -> str:
        prompt = f"""Generate {self.samples_per_category} developer comments discussing copyright issues, ownership, and attribution for the "{license_key}" license.

Requirements:
1. Write as informal developer discussions in comments
2. Questions about: who owns what, updating years, adding names
3. Uncertainty and opinions about copyright are appropriate
4. Include: updating copyright years, adding contributors, ownership transfer
5. Do NOT write actual license text - only discussions
6. Mix of comment styles (#, //, /* */)

Output each example on a new line, nothing else."""
        return prompt

    def _call_llm(self, prompt: str) -> list[str]:
        client = self._get_llm_client()
        last_error: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            self._rate_limiter.acquire()
            try:
                response = client.completion(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    api_base=self.config.api_base_url,
                    api_key=self.config.api_key,
                    custom_llm_provider="openai",
                )
                content = response["choices"][0]["message"]["content"]
                if content is None:
                    raise HardNegativeGeneratorError(
                        "LLM returned None content"
                    )
                text = content.strip()
                if not text:
                    raise HardNegativeGeneratorError("LLM returned empty response")

                lines = [line.strip() for line in text.split("\n") if line.strip()]
                lines = [l for l in lines if len(l) > 5]  # noqa: E741
                return lines
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    wait = self.RETRY_BACKOFF_BASE**attempt
                    logger.warning(
                        "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt,
                        self.MAX_RETRIES,
                        wait,
                        e,
                    )
                    time.sleep(wait)

        raise HardNegativeGeneratorError(
            f"LLM API call failed after {self.MAX_RETRIES} attempts: {last_error}"
        ) from last_error

    def generate_license_discussion(
        self, license_key: str, context: Optional[str] = None
    ) -> list[HardNegativeSample]:
        prompt = self._build_license_discussion_prompt(license_key, context)
        samples = self._call_llm(prompt)

        return [
            HardNegativeSample(
                text=sample,
                negative_type="license_discussion",
                generation_method="llm_generated",
                source_license=license_key,
                llm_model_used=self.config.model,
            )
            for sample in samples
        ]

    def generate_todo_fixme(self, license_key: str) -> list[HardNegativeSample]:
        prompt = self._build_todo_fixme_prompt(license_key)
        samples = self._call_llm(prompt)

        return [
            HardNegativeSample(
                text=sample,
                negative_type="todo_fixme",
                generation_method="llm_generated",
                source_license=license_key,
                llm_model_used=self.config.model,
            )
            for sample in samples
        ]

    def generate_commented_code(self, license_key: str) -> list[HardNegativeSample]:
        prompt = self._build_commented_code_prompt(license_key)
        samples = self._call_llm(prompt)

        return [
            HardNegativeSample(
                text=sample,
                negative_type="commented_code",
                generation_method="llm_generated",
                source_license=license_key,
                llm_model_used=self.config.model,
            )
            for sample in samples
        ]

    def generate_copyright_discussion(
        self, license_key: str
    ) -> list[HardNegativeSample]:
        prompt = self._build_copyright_discussion_prompt(license_key)
        samples = self._call_llm(prompt)

        return [
            HardNegativeSample(
                text=sample,
                negative_type="copyright_discussion",
                generation_method="llm_generated",
                source_license=license_key,
                llm_model_used=self.config.model,
            )
            for sample in samples
        ]

    def generate_for_license(
        self, license_key: str, context: Optional[str] = None
    ) -> list[HardNegativeSample]:
        results: list[HardNegativeSample] = []

        license_discussion_count = max(
            1, int(self.samples_per_category * self.LICENSE_DISCUSSION_WEIGHT)
        )
        todo_fixme_count = max(
            1, int(self.samples_per_category * self.TODO_FIXME_WEIGHT)
        )
        commented_code_count = max(
            1, int(self.samples_per_category * self.COMMENTED_CODE_WEIGHT)
        )
        copyright_discussion_count = max(
            1, int(self.samples_per_category * self.COPYRIGHT_DISCUSSION_WEIGHT)
        )

        results.extend(
            self.generate_license_discussion(license_key, context)[
                :license_discussion_count
            ]
        )
        results.extend(self.generate_todo_fixme(license_key)[:todo_fixme_count])
        results.extend(self.generate_commented_code(license_key)[:commented_code_count])
        results.extend(
            self.generate_copyright_discussion(license_key)[:copyright_discussion_count]
        )

        return results

    def generate_batch(self, license_keys: list[str]) -> list[HardNegativeSample]:
        results: list[HardNegativeSample] = []
        for license_key in license_keys:
            results.extend(self.generate_for_license(license_key))
        return results

    def get_statistics(self, samples: list[HardNegativeSample]) -> dict:
        if not samples:
            return {
                "total": 0,
                "by_type": {},
                "by_method": {},
            }

        by_type: dict[str, int] = {}
        by_method: dict[str, int] = {}

        for sample in samples:
            by_type[sample.negative_type] = by_type.get(sample.negative_type, 0) + 1
            by_method[sample.generation_method] = (
                by_method.get(sample.generation_method, 0) + 1
            )

        return {
            "total": len(samples),
            "by_type": by_type,
            "by_method": by_method,
        }


def generate_hard_negatives(
    license_keys: list[str],
    config: Optional[LLMConfig] = None,
    samples_per_category: int = 5,
) -> list[HardNegativeSample]:
    generator = HardNegativeGenerator(
        config=config,
        samples_per_category=samples_per_category,
    )
    return generator.generate_batch(license_keys)
