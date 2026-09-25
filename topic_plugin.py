"""Autoresearch relevance rule.

A paper is accepted when its title and abstract contain at least one agent or
language model signal and at least one ML engineering or research automation
signal. Both are OR-groups, so the rule cannot be expressed with
``include_any`` and ``include_all`` alone.
"""

import re

from papers_pipeline.models import Paper
from papers_pipeline.topics import TopicDecision

AUTORESEARCH_PHRASES = (
    "mle-bench",
    "mlagentbench",
    "re-bench",
    "mle-dojo",
    "mlr-bench",
    "paperbench",
    "dsbench",
    "kaggle competition",
    "kaggle grandmaster",
    "ml engineering",
    "machine learning engineering",
    "ml research",
    "machine learning research",
    "ai research",
    # Bare "research agent" also means web-search "deep research" agents.
    "autonomous research",
    "autonomous science",
    "ai scientist",
    "autoresearch",
    "automated research",
    "research automation",
    "automate research",
    "automating research",
    "automated scientific discovery",
    "automated machine learning",
    "automl",
    "data science agent",
    "data science task",
    "ml task",
    "machine learning task",
    "model training pipeline",
    "research idea",
)

AGENT_PHRASES = (
    "agent",
    "language model",
    "large language",
    "reasoning model",
    "foundation model",
    "gpt",
    "claude",
    "gemini",
    "deepseek",
    "qwen",
    "llama",
)

# Bare acronyms need word boundaries: "llms" should match, "film" should not.
# AIDE is the MLE-bench reference scaffold; a substring would hit "aided".
AGENT_ACRONYM = re.compile(r"\b(?:llms?|aide)\b")


def accept_topic(paper: Paper) -> TopicDecision:
    """Accept papers about agents that automate ML engineering or research.

    Args:
        paper: Normalized paper to classify.

    Returns:
        Acceptance decision with the rule that decided it.
    """
    text = f"{paper.title} {paper.abstract}".lower()
    if not (
        any(phrase in text for phrase in AGENT_PHRASES) or AGENT_ACRONYM.search(text)
    ):
        return TopicDecision(False, "missing agent or language model signal")
    if not any(phrase in text for phrase in AUTORESEARCH_PHRASES):
        return TopicDecision(False, "missing autoresearch signal")
    return TopicDecision(True, "accepted")
