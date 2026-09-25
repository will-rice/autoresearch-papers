import pytest

from papers_pipeline.config import TopicConfig
from papers_pipeline.models import Paper
from papers_pipeline.topics import TopicDecision, build_topic_gate
from topic_plugin import accept_topic


@pytest.mark.parametrize(
    ("title", "abstract", "decision"),
    [
        (
            "Scaffolds for MLE-bench",
            "An agent solves Kaggle competition tasks end to end.",
            TopicDecision(True, "accepted"),
        ),
        (
            "AIDE: Tree Search over Code",
            "Iterative solution drafting for machine learning engineering.",
            TopicDecision(True, "accepted"),
        ),
        (
            "The AI Scientist",
            "LLMs that generate ideas, run experiments, and write papers.",
            TopicDecision(True, "accepted"),
        ),
        (
            "Computer-Aided Detection in Colonoscopy",
            "Automated machine learning for polyp detection.",
            TopicDecision(False, "missing agent or language model signal"),
        ),
        (
            "Deep Research Agents for Open-Domain QA",
            "A language model agent searches the web to answer questions.",
            TopicDecision(False, "missing autoresearch signal"),
        ),
    ],
)
def test_accept_topic_requires_agent_and_autoresearch_signals(
    paper: Paper, title: str, abstract: str, decision: TopicDecision
) -> None:
    candidate = paper.model_copy(update={"title": title, "abstract": abstract})

    assert accept_topic(candidate) == decision


def test_papers_yml_plugin_reference_builds_repository_gate(paper: Paper) -> None:
    gate = build_topic_gate(TopicConfig(plugin="topic_plugin:accept_topic"))
    candidate = paper.model_copy(
        update={"title": "Autoresearch Loops", "abstract": "An LLM agent on MLE-bench."}
    )

    assert gate(candidate) == TopicDecision(True, "accepted")
