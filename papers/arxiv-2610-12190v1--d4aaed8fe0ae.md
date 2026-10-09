---
identifier: arxiv:2610.12190v1
title: "DataSense-Bench: The First Step Toward an AI Scientist"
authors:
  - Yudi Zhang
  - Mingyu Cao
  - Lu Yin
  - Mykola Pechenizkiy
  - Shiwei Liu
published: "2026-10-08T15:50:57+00:00"
url: https://arxiv.org/abs/2610.12190v1
source: arxiv
doi: null
arxiv_id: 2610.12190v1
categories:
  - cs.LG
---

# DataSense-Bench: The First Step Toward an AI Scientist

Yudi Zhang ^(†)^(†)thanks: Work done during Yudi’s internship at the Max
Planck Institute for Intelligent Systems. Affiliation: Eindhoven
University of Technology    Mingyu Cao Affiliation: University of Surrey
   Lu Yin Affiliation: University of Surrey    Mykola Pechenizkiy
Affiliation: Eindhoven University of Technology    Shiwei Liu
Affiliation: ELLIS Institute Tübingen Affiliation: Max Planck Institute
for Intelligent Systems Affiliation: Tübingen AI Center

###### Abstract

As claims about recursive self-improvement (RSI) and artificial general
intelligence (AGI) proliferate, we ask a simple question: do frontier AI
models have a sense of data, i.e., can they reliably select the right
data for training? We introduce DataSense-Bench¹¹ 1
[Code](https://github.com/DataSense-Bench/DataSense-Bench) and [Project
page](https://datasense-bench.github.io/). to study this capability
through the fundamental problem of data selection and performance
forecasting in machine learning. We ask AI agents to select and rank
candidate training subsets that can be used to fine-tune a small LLM
model. Agents are allowed to inspect the data, write and execute
analysis code, and run model forward passes, but can not train the model
or access the actual evaluation tasks. We then fine-tune the base model
on each selected subset and evaluate its post-training performance under
a standardized protocol. We instantiate the benchmark in terminal
problem solving and tool use, selecting trajectories from
OpenThoughts-Agent and EnvScaler and evaluating on TBLite and BFCL,
respectively. We then evaluate the agents along two complementary
dimensions: the post-training performance of the top-ranked subset,
reflecting the ability to identify high-value training data, and ranking
accuracy, reflecting the ability to predict the relative performance of
the selected subsets. In our experiments, selection gains over random
selection are limited; agents do not reliably rank their selected
groups, and ranking ability does not hold consistently across tasks:
Astra identifies the best group in all three tool-use runs but in only
one of three terminal runs. Analysis of execution traces on both tasks
shows that agents often use similar data signals while interpreting
their training value differently.

|     |
| --- |
|     |

![Refer to caption](2610.12190v1/joint_results.png)

Figure 1: Can CLI agents select useful training data and predict its
value? (a,b) Qwen3-4B performance after training on the predicted
top-ranked group selected by each agent. Panel (a) reports BFCL
accuracy; panel (b) reports TBLite reward. Dashed lines denote random
selection. Scores are averaged across evaluation seeds within
checkpoints, then across three agent runs; error bars show standard
deviation across agent runs (across evaluation seeds for random
selection). (c) Frequency of each predicted rank yielding the best
group, pooling evaluated runs from both tasks with equal weight per run:
six runs for agents evaluated on both tasks and three for TBLite-only
agents. Tied best groups each count as correct; blue outlines mark
$`g_{1}`$.

## 1 Introduction

Large language models (LLMs) are increasingly used for tasks such as
question answering ([Brown et al., 2020](#bib.bib1)), code
generation ([Chen et al., 2021](#bib.bib4)), and data
analysis ([Majumder et al., 2025](#bib.bib16)). Recent advances in
reasoning and tool use have made frontier models capable of tackling
complex scientific problems, supporting AI scientists who generate
research ideas, conduct experiments, analyze results, and write complete
research manuscripts ([Lu et al., 2026](#bib.bib15)). Benchmarks
increasingly assess scientific programming, discovery, and machine
learning experimentation ([Chen et al., 2025](#bib.bib5); [Majumder
et al., 2025](#bib.bib16); [Huang et al., 2024](#bib.bib7); [Rank
et al., 2026](#bib.bib20)). These settings test several abilities
together. They leave a more focused question open: can an AI system
anticipate the value of an experimental choice before observing its
outcome?

Training-data selection provides a concrete setting for this question.
Before committing to training, researchers must choose which examples
will help a particular model learn ([Zhou et al., 2023](#bib.bib30);
[Xia et al., 2024](#bib.bib27)). This is a consequential and difficult
decision: data selection determines the supervision a model receives,
yet observable quality does not directly establish training value. For
example, low loss alone need not imply high training value: an example
may already be well learned ([Mindermann et al., 2022](#bib.bib18)).
Likewise, a successful trajectory does not by itself establish how much
a particular model will learn from it. Reliable judgment requires
connecting properties of the data to what a particular model will learn
under a fixed training procedure ([Li et al., 2024](#bib.bib13); [Xia
et al., 2024](#bib.bib27)).

Our Benchmark. Therefore, we introduce DataSense-Bench to evaluate this
judgment capability of the frontier models. A CLI agent receives a
candidate training data pool, inference-only access to a base model, and
a fixed fine-tuning protocol. It selects disjoint, equal-sized subsets
and ranks them by expected value for the post-training performance. The
agent may inspect data, execute analysis code, and run forward passes,
but cannot train models or access evaluation tasks. Once the submission
is fixed, the benchmark fine-tunes a fresh copy of the base model on
each subset and measures the resulting performance. This separation
makes the submitted ranking a prediction that can be checked against
subsequent experimental evidence. As a consequence, the benchmark asks
about both _Selection quality_ and _Ranking accuracy_, which measure
whether the first group in the LLM’s predicted ordering produces a
useful model and whether the predicted ordering of the subsets agrees
with their observed performance, separately. A strong top choice need
not imply an accurate ordering; conversely, correctly ordering weak
subsets need not produce a useful recommendation. Both are needed to
understand whether an agent can turn data analysis into reliable
decisions.

Our Findings. We instantiate this protocol in terminal problem solving
and multi-turn tool use, using OpenThoughts-Agent and EnvScaler
trajectories for training and TBLite and BFCL for downstream evaluation,
respectively. Our experiments reveal two limitations: gains over random
selection are modest, and agents do not reliably rank their selected
groups. Ranking ability also does not hold consistently across tasks.
(1) The frontier agents produce plausible, data-specific recipes but do
not reliably predict post-training utility. On TBLite, two LLMs select
first groups that underperform random selection, with differences across
LLMs ranging from -0.55 to +0.99 points, as shown in
Table [1](#S4.T1 "Table 1 ‣ 4.2 Selection Quality and Ranking Accuracy ‣ 4 Experiments ‣ DataSense-Bench: The First Step Toward an AI Scientist").
On BFCL, all six agents’ first groups exceed random selection by
0.35–1.78 % points. Astra identifies the best group in all three BFCL
runs but in only one of three TBLite runs. None of the six agents
evaluated on both tasks identifies the best group in all three runs on
each task. (2) Selection recipes vary substantially across three
independent runs of the same agent. Astra’s TBLite runs change their
quality-assessment procedures, while Kimi’s BFCL runs move from
loss-outlier rejection to intermediate-loss and then lower-loss
preferences. The selection rule itself changes across runs. (3) Agents
draw different conclusions from similar data signals. For example,
agents use base-model loss in substantially different ways across runs
and tasks. Section 4.3 examines the concrete recipes in detail.

## 2 Related Work

We review benchmarks for CLI agents, training-data selection, and
performance forecasting.

Benchmarks for CLI Agents. Command-line interface (CLI) agents use shell
commands and scripts to complete tasks in OS environments.
Terminal-Bench evaluates realistic terminal tasks with executable
verification, while SWE-bench focuses on resolving real GitHub issues
through repository modifications ([Merrill et al., 2026](#bib.bib17);
[Jimenez et al., 2024](#bib.bib9)). Research-oriented benchmarks address
scientific programming and reproducibility ([Chen et al.,
2025](#bib.bib5); [Siegel et al., 2024](#bib.bib22)), as well as machine
learning experimentation and research engineering ([Huang et al.,
2024](#bib.bib7); [Chan et al., 2025](#bib.bib2); [Wijk et al.,
2025](#bib.bib26)). Most closely related, PostTrainBench evaluates
agents’ choices of data, methods, and hyperparameters for LLM
post-training under a compute budget ([Rank et al., 2026](#bib.bib20)).
These benchmarks measure end-to-end outcomes that depend on both
decision quality and execution. DataSense-Bench focuses on
data-selection judgment: agents inspect candidate data, construct
selection procedures, and submit ranked subsets for a fixed base model
and training protocol before receiving any training feedback.

Training Data Selection. Existing selection methods differ in the
signals they use to identify useful training data. Curation and scoring
approaches emphasize the value of small, high-quality instruction
sets ([Zhou et al., 2023](#bib.bib30)) and use model-based assessments
of quality or instruction-following difficulty to select examples ([Chen
et al., 2024](#bib.bib3); [Li et al., 2024](#bib.bib13)). Other
approaches seek to use importance resampling ([Xie et al.,
2023b](#bib.bib29)) or improve coverage through deduplication and
embedding-based data selection ([Tirumala et al., 2023](#bib.bib25)).
RHO-LOSS uses a holdout-trained model to prioritize examples expected to
reduce generalization loss ([Mindermann et al., 2022](#bib.bib18)). More
recently, FEEDER assesses the sufficiency and necessity of
demonstrations for in-context learning, with an extension to
fine-tuning ([Jin et al., 2025](#bib.bib10)). Together, these methods
motivate the quality, loss, and coverage signals examined in our
procedure analysis.

Performance Forecasting. A complementary line estimates how training
examples affect model behavior through derivative-based influence
approximations ([Koh & Liang, 2017](#bib.bib11); [Pruthi et al.,
2020](#bib.bib19); [Kwon et al., 2024](#bib.bib12)), marginal
contributions to predictive performance ([Ghorbani & Zou,
2019](#bib.bib6)), or gradient similarity for targeted instruction
tuning ([Xia et al., 2024](#bib.bib27)). At the subset or mixture level,
datamodels predict model outputs from training-subset membership ([Ilyas
et al., 2022](#bib.bib8)), while DoReMi learns domain weights with a
proxy model and RegMix predicts mixture performance from small training
runs ([Xie et al., 2023a](#bib.bib28); [Liu et al., 2025](#bib.bib14)).
These approaches differ in their use of target examples, gradients, and
training feedback. DataSense-Bench provides a standardized protocol for
evaluating whether frontier models can select useful subsets and
forecast their relative downstream performance. Subsequent training and
evaluation measure both top-subset performance and ranking accuracy.

## 3 Setup of DataSense-Bench

DataSense-Bench aims to evaluate whether a CLI agent can judge the
training value of data _before observing training outcomes_. We explain
how we set up our benchmark in detail as follows.

Figure 2: From data-selection predictions to post-hoc verification.
Illustrated for terminal problem solving, the CLI agent analyzes
candidate trajectories with inference-only access to the base model. It
delivers a row-to-group assignment CSV and a method report specifying
its selection recipe. We validate five disjoint groups of the specified
size and freeze their membership and predicted order. Only then does
evaluation perform SFT and downstream scoring, testing selection
quality, best-group accuracy, and agreement with the full predicted
ordering.

### 3.1 Task for CLI Agents: Select Useful Data and Predict Its Value

Let $`D`$ be a pool of $`N`$ candidate trajectories, $`M`$ a base model,
and $`R`$ the fine-tuning procedure used to test a submission. The agent
selects $`K`$ subsets $`G=(g_{1},\ldots,g_{K})`$ with
$`g_{k}\subseteq D`$, $`|g_{k}|=n`$, and $`g_{i}\cap g_{j}=\varnothing`$
for $`i\neq j`$. The group labels encode its prediction: training on
$`g_{1}`$ should give the highest downstream score, followed by
$`g_{2}`$, through $`g_{K}`$.
Figure [3](#S3.F3 "Figure 3 ‣ 3.1 Task for CLI Agents: Select Useful Data and Predict Its Value ‣ 3 Setup of DataSense-Bench ‣ DataSense-Bench: The First Step Toward an AI Scientist")
provides a condensed prompt example used in our benchmark, and the full
prompt appears in
Appendix [C](#A3 "Appendix C Selection Prompts ‣ DataSense-Bench: The First Step Toward an AI Scientist").

Post-training Data. We study two tasks that require learning from
multi-step interactions: terminal problem solving and multi-turn tool
use. Both expose intermediate actions and feedback, allowing agents to
assess which trajectories may provide useful supervision. For each task,
agents select five disjoint groups of complete trajectories, and we
independently fine-tune Qwen3-4B ²² 2
[https://huggingface.co/Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
on each group using supervised fine-tuning (SFT). Whole trajectories are
selected before the training procedure converts them into training
samples.

_Tool usage._ The training data come from the EnvScaler SFT pool ([Song
et al., 2026](#bib.bib23)), with validation trajectories excluded. These
trajectories involve selecting functions, supplying arguments, and
responding to tool feedback. Each selected group contains 50 complete
trajectories. We evaluate the fine-tuned models on BFCL V3³³ 3
[https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html),
measuring multi-turn tool-use performance.

_Terminal problem solving._ The training pool is
OpenThoughts-Agent-SFT-10K ([Raoof et al., 2026](#bib.bib21))⁴⁴ 4
Training data:
[open-thoughts/OpenThoughts-Agent-SFT-10K](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-10K),
which contains trajectories pairing task instructions with terminal
interactions, including commands, their outputs, and error recovery.
Each selected group contains 1,000 trajectories. We evaluate the
fine-tuned models on TBLite ([Raoof et al., 2026](#bib.bib21)), where
automated task verifiers measure success on terminal tasks.

We will explain BFCL V3 and TBLite later, and the agents are not allowed
to access these downstream evaluation tasks.

Agent’s Objectives. We ask agents to select five groups and specify two
objectives, as shown in
Figure [3](#S3.F3 "Figure 3 ‣ 3.1 Task for CLI Agents: Select Useful Data and Predict Its Value ‣ 3 Setup of DataSense-Bench ‣ DataSense-Bench: The First Step Toward an AI Scientist"):

- •
  to make $`g_{1}`$ as strong as possible, ideally outperforming random
  selection at the same size.
- •
  to order the groups by expected post-training performance.

These objectives assess complementary abilities: ranking evaluates the
ordering of weak groups, whereas selection evaluates the ability to
identify a useful subset.

Resources that the agent can access. The agent receives the task prompt,
candidate data, base model, task configuration, and a supervised
fine-tuning script. The prompt specifies the downstream task types
without revealing evaluation instances, while leaving the choice and
combination of selection signals to the agent. Each selection run has
inference-only access to the base model and is allocated one H100 80 GB
GPU and four hours to analyze the data and submit a selection.
Subsequent training and evaluation are performed outside this time
budget. Within each task, all agents receive the same fixed training
recipe, which is used to fine-tune every selected group. We evaluate ten
CLI agents on TBLite and six on BFCL, with three independent selection
runs per agent and task. Please refer to
Table [3](#A1.T3 "Table 3 ‣ A.1 Training Configuration ‣ Appendix A Experimental Details ‣ DataSense-Bench: The First Step Toward an AI Scientist")
in
Appendix [A](#A1 "Appendix A Experimental Details ‣ DataSense-Bench: The First Step Toward an AI Scientist")
for more experimental configuration.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzMuRjMucGljMSIgY2xhc3M9Imx0eF9waWN0dXJlIiBoZWlnaHQ9IjUxNS40NiIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDU1MCA1MTUuNDYiIHdpZHRoPSI1NTAiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsNTE1LjQ2KSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0JGQkZGRjsiIGZpbGw9IiNCRkJGRkYiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMCA0LjQ5IEwgMCA1MTAuOTcgQyAwIDUxMy40NSAyLjAxIDUxNS40NiA0LjQ5IDUxNS40NiBMIDU0NS41MSA1MTUuNDYgQyA1NDcuOTkgNTE1LjQ2IDU1MCA1MTMuNDUgNTUwIDUxMC45NyBMIDU1MCA0LjQ5IEMgNTUwIDIuMDEgNTQ3Ljk5IDAgNTQ1LjUxIDAgTCA0LjQ5IDAgQyAyLjAxIDAgMCAyLjAxIDAgNC40OSBaIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojRkFGQUZGOyIgZmlsbD0iI0ZBRkFGRiIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwLjU1IDQuNDkgTCAwLjU1IDQ5Mi42NSBMIDU0OS40NSA0OTIuNjUgTCA1NDkuNDUgNC40OSBDIDU0OS40NSAyLjMyIDU0Ny42OCAwLjU1IDU0NS41MSAwLjU1IEwgNC40OSAwLjU1IEMgMi4zMiAwLjU1IDAuNTUgMi4zMiAwLjU1IDQuNDkgWiIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0VCRUJGRjsiIGZpbGw9IiNFQkVCRkYiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMC41NSA0OTMuMiBMIDAuNTUgNTEwLjk3IEMgMC41NSA1MTMuMTUgMi4zMiA1MTQuOTEgNC40OSA1MTQuOTEgTCA1NDUuNTEgNTE0LjkxIEMgNTQ3LjY4IDUxNC45MSA1NDkuNDUgNTEzLjE1IDU0OS40NSA1MTAuOTcgTCA1NDkuNDUgNDkzLjIgWiIgLz48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxMi43OSA1MDAuNikiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDozMi45NmVtOy0tbHR4LWZvLWhlaWdodDowLjc1ZW07LS1sdHgtZm8tZGVwdGg6MC4yNWVtO2ZvbnQtc2l6ZToxMHB0OyIgaGVpZ2h0PSIxMy44NCIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTAuMzgpIiB3aWR0aD0iNDU2LjA3Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+CjxzcGFuIGlkPSJTMy5GMy5waWMxLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6MzIuOTZlbTsiPgo8c3BhbiBpZD0iUzMuRjMucGljMS4xLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlMzLkYzLnBpYzEuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkV4YW1wbGUgc2VsZWN0aW9uIHByb21wdCAoY29uZGVuc2VkKTwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxMi43OSAxMS40MSkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDozNy45ZW07LS1sdHgtZm8taGVpZ2h0OjMzLjk5ZW07LS1sdHgtZm8tZGVwdGg6MGVtO2ZvbnQtc2l6ZToxMHB0OyIgaGVpZ2h0PSI0NzAuMzkiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDQ3MC4zOSkiIHdpZHRoPSI1MjQuNDMiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij4KPHNwYW4gaWQ9IlMzLkYzLnBpYzEuMiIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDozNy45ZW07Ij4KPHNwYW4gaWQ9IlMzLkYzLnBpYzEuMi4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTMy5GMy5waWMxLjIuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5UYXNrLjxzcGFuIGlkPSJTMy5GMy5waWMxLjIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9tZWRpdW0iPiBTZWxlY3QgdHJhaW5pbmcgZGF0YSBmb3IgUXdlbjMtNEIgZnJvbSBhIHBvb2wgb2YgMTAsMDAwIHRlcm1pbmFsCnRyYWplY3Rvcmllcy4gUmVhZCA8c3BhbiBpZD0iUzMuRjMucGljMS4yLjEuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5jb25maWcueWFtbDwvc3Bhbj4gZm9yIHRoZSBkYXRhIGFuZCBtb2RlbCBwYXRocywKc2VxdWVuY2UtbGVuZ3RoIGxpbWl0LCBhbmQgZ3JvdXAgc2l6ZXMuIEluc3BlY3QgdGhlIGNhbmRpZGF0ZSBjb252ZXJzYXRpb25zCmFuZCA8c3BhbiBpZD0iUzMuRjMucGljMS4yLjEuMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5yZWNpcGUvc2Z0LnB5PC9zcGFuPiB0byB1bmRlcnN0YW5kIHRoZSBTRlQgcHJvY2VkdXJlLCBpbmNsdWRpbmcKZm9ybWF0dGluZywgdHJ1bmNhdGlvbiwgYW5kIGFzc2lzdGFudC10b2tlbiBzdXBlcnZpc2lvbi48L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlMzLkYzLnBpYzEuMi4yIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTMy5GMy5waWMxLjIuMi4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5PYmplY3RpdmVzLjxzcGFuIGlkPSJTMy5GMy5waWMxLjIuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9tZWRpdW0iPiBQcm9kdWNlIGZpdmUgZGlzam9pbnQgZ3JvdXBzIG9mIDEsMDAwIGNvbXBsZXRlIHRyYWplY3RvcmllcyBlYWNoLiBNYWtlIDxtYXRoIGlkPSJTMy5GMy5waWMxLm0xIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9ImdfezF9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5nPC9taT48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjE8L21uPjwvbXN1Yj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPmdfezF9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4KYXMgc3Ryb25nIGFzIHBvc3NpYmxlLCBpZGVhbGx5IGJldHRlciB0aGFuIGEgcmFuZG9tIHN1YnNldCBvZiB0aGUgc2FtZQpzaXplLCBhbmQgcmFuayA8bWF0aCBpZD0iUzMuRjMucGljMS5tMiIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJnX3sxfSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+ZzwvbWk+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4xPC9tbj48L21zdWI+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5nX3sxfTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+4oCTPG1hdGggaWQ9IlMzLkYzLnBpYzEubTMiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iZ197NX0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPmc8L21pPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NTwvbW4+PC9tc3ViPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+Z197NX08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiBieSBleHBlY3RlZCBwb3N0LXRyYWluaW5nIHBlcmZvcm1hbmNlLiBFYWNoIHNlbGVjdGVkCnN1YnNldCB3aWxsIGJlIHVzZWQgdG8gZmluZS10dW5lIGEgZnJlc2ggbW9kZWwsIHdob3NlIHRlcm1pbmFsLXRhc2sKcGVyZm9ybWFuY2Ugd2lsbCBiZSBjaGVja2VkIGJ5IGF1dG9tYXRlZCB0ZXN0cy48L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlMzLkYzLnBpYzEuMi4zIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTMy5GMy5waWMxLjIuMy4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5SZXNvdXJjZXMgYW5kIGxpbWl0cy48c3BhbiBpZD0iUzMuRjMucGljMS4yLjMuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfbWVkaXVtIj4gQ2hvb3NlIHlvdXIgb3duIHNlbGVjdGlvbiBtZXRob2QuIFlvdSBtYXkgaW5zcGVjdCBkYXRhLCBleGVjdXRlIGFuYWx5c2lzCmNvZGUsIGFuZCBydW4gYmFzZS1tb2RlbCBmb3J3YXJkIHBhc3Nlcy4gWW91IGhhdmUgb25lIEgxMDAgODDigIlHQiBHUFUgYW5kCmZvdXIgaG91cnMuIERvIG5vdCB0cmFpbiBtb2RlbHMsIHVwZGF0ZSB3ZWlnaHRzLCBtb2RpZnkgb3IgZXhlY3V0ZSB0aGUgU0ZUCnNjcmlwdCwgb3IgYWNjZXNzIGV2YWx1YXRpb24gdGFza3MgYW5kIHRlc3RzLjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iUzMuRjMucGljMS4yLjQiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlMzLkYzLnBpYzEuMi40LjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkRlbGl2ZXJhYmxlcy48c3BhbiBpZD0iUzMuRjMucGljMS4yLjQuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfbWVkaXVtIj4gU3VibWl0IDxzcGFuIGlkPSJTMy5GMy5waWMxLjIuNC4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPm91dHB1dHMvZ3JvdXBfYXNzaWdubWVudC5jc3Y8L3NwYW4+IHdpdGggY29sdW1ucwo8c3BhbiBpZD0iUzMuRjMucGljMS4yLjQuMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5pbmRleCxncm91cF9pZDwvc3Bhbj4gYW5kIDxzcGFuIGlkPSJTMy5GMy5waWMxLjIuNC4xLjEuMyIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPm91dHB1dHMvbWV0aG9kLm1kPC9zcGFuPiBleHBsYWluaW5nIHlvdXIKcmVjaXBlLiBDaGVjayB2YWxpZCByb3cgaW5kaWNlcywgZXhhY3QgZ3JvdXAgc2l6ZXMsIGFuZCBubyBvdmVybGFwIGJlZm9yZQpmaW5pc2hpbmcuIFlvdXIgZ3JvdXBzIGFuZCBwcmVkaWN0ZWQgb3JkZXIgbXVzdCBiZSBmaXhlZCBiZWZvcmUgc2VlaW5nCnRyYWluaW5nIG9yIGV2YWx1YXRpb24gb3V0Y29tZXMuPC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PC9nPjwvc3ZnPg==)

Figure 3: Condensed terminal-task prompt specifying the selection and
ranking objectives, permitted tools, computation budget, and required
deliverables. The prompt for BFCL is described in
Appendix [C.3](#A3.SS3 "C.3 Multi-turn Tool-use Selection ‣ Appendix C Selection Prompts ‣ DataSense-Bench: The First Step Toward an AI Scientist").

Submission Protocol. Each agent is asked to submit a group-assignment
CSV and a short method report explaining the selection recipe and its
rationale. The CSV assigns valid candidate-pool indices to five disjoint
groups: 50 for BFCL and 1,000 complete trajectories per group for
TBLite, omitting unselected rows. Group labels $`g_{1}`$ through
$`g_{5}`$ indicate the predicted ranking by expected post-training
performance, from highest to lowest. The agent must write and verify
both files within the selection time budget, finalizing group
composition and ordering before observing any training or evaluation
outcomes; no trained model is submitted.

### 3.2 Post-check and Post-training Evaluation

After the agent submits its selection, the benchmark validates the
submission and tests its predicted ordering, shown in
Figure [2](#S3.F2 "Figure 2 ‣ 3 Setup of DataSense-Bench ‣ DataSense-Bench: The First Step Toward an AI Scientist").

Submission validation. We check the assignment against the submission
protocol in
Section [3.1](#S3.SS1 "3.1 Task for CLI Agents: Select Useful Data and Predict Its Value ‣ 3 Setup of DataSense-Bench ‣ DataSense-Bench: The First Step Toward an AI Scientist")
and verify that the method report is present. For BFCL, we also
cross-check row positions against source trajectory identifiers and
exclude validation trajectories. CLI completion and valid deliverables
are checked separately. If the first attempt fails to produce a valid
submission, the agent may retry once using the same prompt within the
remaining four-hour selection budget. The two attempts share this
budget. If the second attempt also fails, or no valid submission is
produced before the budget expires, we apply the failure scores defined
in
Section [3.3](#S3.SS3 "3.3 Evaluation Metrics ‣ 3 Setup of DataSense-Bench ‣ DataSense-Bench: The First Step Toward an AI Scientist").

Post-training evaluation. Each group is used to fine-tune a fresh copy
of the base model under its task-specific training procedure. Within
each task, all resulting models use the same evaluation suite and
scoring protocol. Let $`s_{k}`$ denote the downstream score (TBLite
average reward or BFCL accuracy) for the model trained on $`g_{k}`$,
averaged across evaluation tasks and evaluation seeds. These outcomes
are used only to assess the submitted selection and ordering, not to
revise them.
Section [4.1](#S4.SS1 "4.1 Downstream Evaluation ‣ 4 Experiments ‣ DataSense-Bench: The First Step Toward an AI Scientist")
describes both downstream evaluation protocols.

### 3.3 Evaluation Metrics

We want to evaluate two complementary capabilities of the data selection
agents: whether the agent selects useful training data, and whether it
can anticipate which data choices will be more useful. We therefore
evaluate the performance of its first-ranked group and the agreement
between its predicted and observed ordering. Selection quality. The
score $`s_{1}`$ measures the utility of the data the agent recommends
most strongly. To assess whether this choice improves on an uninformed
selection, we also report its gain over random selection,
$`\Delta_{\mathrm{random}}=s_{1}-s_{\mathrm{random}}`$, using the same
group size and training procedure. A positive gain supports the
selection objective, but does not establish that the agent can compare
its five groups. Ranking accuracy. We test that comparison at two
levels. Best-group accuracy asks whether the agent’s first choice is
actually the best among its selected groups:
$`A_{\mathrm{best}}=\mathbb{1}[s_{1}=\max_{k=1,\ldots,5}s_{k}]`$.
Spearman correlation ([Spearman, 1904](#bib.bib24)) tests the full
predicted ordering:
$`S_{\mathrm{rank}}=\rho_{\mathrm{Spearman}}\big((4,3,2,1,0),(s_{1},\ldots,s_{5})\big)`$.
These measures distinguish identifying the best group from correctly
ordering all alternatives. We reverse the group-index scale so that
perfect rank agreement gives $`+1`$, whereas a reversed ordering gives
$`-1`$.

For valid submissions, we compute both ranking metrics within each
selection run after evaluating all five groups. Best-group accuracy
counts a tie for the highest unrounded score as correct. Spearman
correlation uses average ranks for ties and is undefined when all five
outcomes are equal. Such runs are excluded only from the mean Spearman
correlation; they remain in the averages of selection scores and
best-group accuracy. For failed submissions under
Section [3.2](#S3.SS2 "3.2 Post-check and Post-training Evaluation ‣ 3 Setup of DataSense-Bench ‣ DataSense-Bench: The First Step Toward an AI Scientist"),
we set $`s_{1}=s_{\mathrm{base}}`$, $`S_{\mathrm{rank}}=-1`$, and
$`A_{\mathrm{best}}=0`$. These are failure penalties, not correlations
computed from tied base-model scores. Failed runs remain in the averages
across selection runs; $`\Delta_{\mathrm{random}}`$ uses the fallback
$`s_{1}`$.

## 4 Experiments

Our experiments ask whether agents select data that improves downstream
performance, whether their predicted group order is reliable, and how
they construct these judgments. We report outcomes first, then analyze
the selection procedures and their resource requirements.

### 4.1 Downstream Evaluation

We evaluate the fine-tuned models on the two downstream tasks introduced
in
Section [3.1](#S3.SS1 "3.1 Task for CLI Agents: Select Useful Data and Predict Its Value ‣ 3 Setup of DataSense-Bench ‣ DataSense-Bench: The First Step Toward an AI Scientist").
The data-selection agents do not perform these evaluations or access
their test instances during the data selection.

TBLite. OpenThoughts-TBLite contains 100 terminal tasks across nine
categories, including software engineering, debugging, data processing,
and security ([Raoof et al., 2026](#bib.bib21)). We use Harbor’s⁵⁵ 5
[https://www.harborframework.com/](https://www.harborframework.com/)
Terminus-2 scaffold to execute model-generated commands and return
terminal outputs. Each attempt starts in a fresh Daytona⁶⁶ 6
[https://www.daytona.io/docs/en/](https://www.daytona.io/docs/en/)
sandbox and ends when the model finishes or reaches the task’s time
limit. Task-supplied verifiers check the resulting environment or
artifacts and assign rewards, including fractional credit where
supported. We report average reward over 100 tasks with three attempts
per task, giving 300 equally weighted scores per checkpoint.

BFCL. For models trained on EnvScaler trajectories, we use BFCL V3: 200
cases each for base interactions, missing functions, missing parameters,
and long context. The evaluator checks tool-use outcomes against the
required state and execution behavior. Each evaluation seed covers all
800 cases; its overall accuracy averages the four equally sized
categories. We use three evaluation seeds per checkpoint.

### 4.2 Selection Quality and Ranking Accuracy

[TABLE]

Table 1: Selection performance, best-group accuracy, and rank
performance on BFCL and TBLite. Agent-group scores ($`s_{1}`$-$`s_{5}`$)
report the post-training performance, accuracy (%) for BFCL, and reward
for TBLite (scale from 0-1 to 0-100). We report the average performance
across three evaluation seeds and three data selection runs, standard
deviation (subscript), and maximum score among different data selection
runs. Shading compares group averages within a row; red outlines mark
each task’s largest maximum among agent-selected groups. $`g_{1}`$ gain
$`\Delta_{\text{random}}`$ is relative to random selection. Best-group
accuracy $`A_{\mathrm{best}}`$ measures how often the agent identifies
its best group; rank performance $`S_{\mathrm{rank}}`$ measures Spearman
correlation between predicted and observed group orderings
(Section [3.3](#S3.SS3 "3.3 Evaluation Metrics ‣ 3 Setup of DataSense-Bench ‣ DataSense-Bench: The First Step Toward an AI Scientist")).
More detailed experimental results are in
Appendix [A.3](#A1.SS3 "A.3 Downstream Evaluation and Result Aggregation ‣ Appendix A Experimental Details ‣ DataSense-Bench: The First Step Toward an AI Scientist").

Table [1](#S4.T1 "Table 1 ‣ 4.2 Selection Quality and Ranking Accuracy ‣ 4 Experiments ‣ DataSense-Bench: The First Step Toward an AI Scientist")
separates two questions: whether agents choose useful training data and
whether they correctly rank the groups they select.

Selection yields limited gains over random selection on both tasks. On
BFCL, all six agents’ average $`g_{1}`$ accuracies exceed random
selection’s 34.75%, but gains range from only 0.35 to 1.78 % points.
DeepSeek V4 Pro has the highest average at 36.53%. Its run-3 $`g_{1}`$
achieves the highest accuracy among agent-selected checkpoints, 38.62%.
On TBLite, average $`g_{1}`$ rewards range from 11.00% to 12.54%,
compared with 11.55% for random selection. Two of ten agents
underperform this reference, and DeepSeek V4 Pro’s leading score
improves on it by only 0.99 points. Thus, elaborate selection recipes
yield modest gains on both tasks, with less consistent improvements on
TBLite.

Selecting useful data does not imply reliably ranking the selected
groups. BFCL reveals a gap between selection quality and ranking
accuracy: all six agents improve over Random with $`g_{1}`$, yet their
average Spearman correlations range from $`-0.27`$ to $`0.83`$. GPT-6
Astra identifies the best group in all three runs and achieves a
correlation of 0.83. In contrast, Kimi K3 achieves 36.29% average
$`g_{1}`$ accuracy but a correlation of only 0.02, while Gemini’s
negative correlation indicates disagreement with the predicted ordering.
Conversely, Opus has a correlation of 0.77 but never identifies the best
group, illustrating that agreement over all five positions can coexist
with an incorrect first choice. The difficulty is more widespread on
TBLite, where correlations range from $`-0.50`$ to $`0.30`$. Even
DeepSeek V4 Pro, the strongest agent by average $`g_{1}`$ reward,
achieves only 0.13. Kimi K3 places the highest-performing agent-selected
subset fourth: its run-3 $`g_{4}`$ reaches 15.16% average reward.
Together, the two tasks show that gains over Random do not establish an
ability to compare alternative data choices; no agent identifies the
best group in all three runs on both tasks. Astra, for example, does so
in three of three BFCL runs but only one of three TBLite runs. Appendix
Tables [7](#A2.T7 "Table 7 ‣ B.3 Results by Selection ‣ Appendix B Detailed Experimental Results ‣ DataSense-Bench: The First Step Toward an AI Scientist")
and [8](#A2.T8 "Table 8 ‣ B.3 Results by Selection ‣ Appendix B Detailed Experimental Results ‣ DataSense-Bench: The First Step Toward an AI Scientist")
report the individual selection runs.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzQuRjQucGljMSIgY2xhc3M9Imx0eF9waWN0dXJlIGx0eF9jZW50ZXJpbmciIGhlaWdodD0iMTE0My4yNyIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDU1MCAxMTQzLjI3IiB3aWR0aD0iNTUwIj48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMDAwMDA7LS1sdHgtZmlsbC1jb2xvcjojMDAwMDAwOyIgZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjAuNHB0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDExNDMuMjcpIG1hdHJpeCgxIDAgMCAtMSAwIDApIj48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojNzM0RDI2OyIgZmlsbD0iIzczNEQyNiIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwIDAuNTUgTCAwIDExNDIuNzEgQyAwIDExNDMuMDIgMC4yNSAxMTQzLjI3IDAuNTUgMTE0My4yNyBMIDU0OS40NSAxMTQzLjI3IEMgNTQ5Ljc1IDExNDMuMjcgNTUwIDExNDMuMDIgNTUwIDExNDIuNzEgTCA1NTAgMC41NSBDIDU1MCAwLjI1IDU0OS43NSAwIDU0OS40NSAwIEwgMC41NSAwIEMgMC4yNSAwIDAgMC4yNSAwIDAuNTUgWiIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0ZERkJGOTsiIGZpbGw9IiNGREZCRjkiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMC41NSAwLjU1IEwgMC41NSAxMTQyLjcxIEwgNTQ5LjQ1IDExNDIuNzEgTCA1NDkuNDUgMC41NSBaIiAvPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEyLjc5IDExLjQxKSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjM3LjllbTstLWx0eC1mby1oZWlnaHQ6ODAuOThlbTstLWx0eC1mby1kZXB0aDowZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9IjExMjAuNDYiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDExMjAuNDYpIiB3aWR0aD0iNTI0LjQzIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6MzcuOWVtOyI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMSIgY2xhc3M9Imx0eF90YWJ1bGFyIGx0eF9ndWVzc2VkX2hlYWRlcnMgbHR4X2FsaWduX21pZGRsZSI+CjxzcGFuIGNsYXNzPSJsdHhfdGhlYWQiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuMSIgY2xhc3M9Imx0eF90ciI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS4xLjEiIGNsYXNzPSJsdHhfdGQgbHR4X2FsaWduX2xlZnQgbHR4X2FsaWduX3RvcCBsdHhfdGggbHR4X3RoX2NvbHVtbiBsdHhfYm9yZGVyX3R0IiBzdHlsZT0icGFkZGluZy10b3A6MC42cHQ7cGFkZGluZy1ib3R0b206MC42cHQ7Ij4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjEuMS4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfYWxpZ25fdG9wIiBzdHlsZT0id2lkdGg6NDUuNXB0OyI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS4xLjEuMS4xIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS4xLjEuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5UaW1lPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjEuMiIgY2xhc3M9Imx0eF90ZCBsdHhfbm9wYWRfciBsdHhfYWxpZ25fbGVmdCBsdHhfYWxpZ25fdG9wIGx0eF90aCBsdHhfdGhfY29sdW1uIGx0eF9ib3JkZXJfdHQiIHN0eWxlPSJwYWRkaW5nLXRvcDowLjZwdDtwYWRkaW5nLWJvdHRvbTowLjZwdDsiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuMS4yLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9hbGlnbl90b3AiIHN0eWxlPSJ3aWR0aDozMTAuOHB0OyI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS4xLjIuMS4xIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS4xLjIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5FeGVjdXRlZCBjb21tYW5kcyBhbmQgcmVzdWx0aW5nIGRlY2lzaW9uczwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8L3NwYW4+CjxzcGFuIGNsYXNzPSJsdHhfdGJvZHkiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuMiIgY2xhc3M9Imx0eF90ciI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS4yLjEiIGNsYXNzPSJsdHhfdGQgbHR4X2FsaWduX2xlZnQgbHR4X2FsaWduX3RvcCBsdHhfYm9yZGVyX3QiIHN0eWxlPSJwYWRkaW5nLWJvdHRvbTogMy4wcHQ7cGFkZGluZy10b3A6MC42cHQ7cGFkZGluZy1ib3R0b206MC42cHQ7Ij4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjIuMS4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfYWxpZ25fdG9wIiBzdHlsZT0id2lkdGg6NDUuNXB0OyI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS4yLjEuMS4xIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS4yLjEuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4wMDowMDowMjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS4yLjIiIGNsYXNzPSJsdHhfdGQgbHR4X25vcGFkX3IgbHR4X2FsaWduX2xlZnQgbHR4X2FsaWduX3RvcCBsdHhfYm9yZGVyX3QiIHN0eWxlPSJwYWRkaW5nLWJvdHRvbTogMy4wcHQ7cGFkZGluZy10b3A6MC42cHQ7cGFkZGluZy1ib3R0b206MC42cHQ7Ij4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjIuMi4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfYWxpZ25fdG9wIiBzdHlsZT0id2lkdGg6MzEwLjhwdDsiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuMi4yLjEuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuMi4yLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5jYXQgY29uZmlnLnlhbWw8L3NwYW4+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjIuMi4xLjEuMiIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPjsgPC9zcGFuPjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS4yLjIuMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPmNhdCByZWNpcGUvc2Z0LnB5PC9zcGFuPjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS4yLjIuMS4xLjQiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij47IDwvc3Bhbj48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuMi4yLjEuMS41IiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5udmlkaWEtc21pPC9zcGFuPjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS4yLjIuMS4xLjYiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4uIFJlYWQgdGhlIHNlbGVjdGlvbiBjb25maWd1cmF0aW9uIGFuZCBTRlQgcmVjaXBlOyBpbnNwZWN0IGF2YWlsYWJsZSBjb21wdXRlLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuMyIgY2xhc3M9Imx0eF90ciI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS4zLjEiIGNsYXNzPSJsdHhfdGQgbHR4X2FsaWduX2xlZnQgbHR4X2FsaWduX3RvcCIgc3R5bGU9InBhZGRpbmctYm90dG9tOiAzLjBwdDtwYWRkaW5nLXRvcDowLjZwdDtwYWRkaW5nLWJvdHRvbTowLjZwdDsiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuMy4xLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9hbGlnbl90b3AiIHN0eWxlPSJ3aWR0aDo0NS41cHQ7Ij4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjMuMS4xLjEiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjMuMS4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPjAwOjAwOjQ54oCTMDA6MDM6MTM8L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuMy4yIiBjbGFzcz0ibHR4X3RkIGx0eF9ub3BhZF9yIGx0eF9hbGlnbl9sZWZ0IGx0eF9hbGlnbl90b3AiIHN0eWxlPSJwYWRkaW5nLWJvdHRvbTogMy4wcHQ7cGFkZGluZy10b3A6MC42cHQ7cGFkZGluZy1ib3R0b206MC42cHQ7Ij4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjMuMi4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfYWxpZ25fdG9wIiBzdHlsZT0id2lkdGg6MzEwLjhwdDsiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuMy4yLjEuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuMy4yLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5weXRob24gLSAmbHQ7Jmx0O+KAmUVPRuKAmTwvc3Bhbj48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuMy4yLjEuMS4yIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+OiByZWFkIDwvc3Bhbj48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuMy4yLjEuMS4zIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5kYXRhL3RyYWluLnBhcnF1ZXQ8L3NwYW4+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjMuMi4xLjEuNCIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiwgcHJpbnQgZXhhbXBsZSB0dXJucyBhbmQgdGFzayBkZXNjcmlwdGlvbnMsIGFuZCBjb3VudCBjb21wbGV0aW9uIG1hcmtlcnMsIHBhcnNlciB3YXJuaW5ncywgYW5kIHJlc3BvbnNlIGZvcm1hdHMuPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS40IiBjbGFzcz0ibHR4X3RyIj4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjQuMSIgY2xhc3M9Imx0eF90ZCBsdHhfYWxpZ25fbGVmdCBsdHhfYWxpZ25fdG9wIiBzdHlsZT0icGFkZGluZy1ib3R0b206IDMuMHB0O3BhZGRpbmctdG9wOjAuNnB0O3BhZGRpbmctYm90dG9tOjAuNnB0OyI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS40LjEuMSIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X2FsaWduX3RvcCIgc3R5bGU9IndpZHRoOjQ1LjVwdDsiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNC4xLjEuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNC4xLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+MDA6MDQ6MDQ8L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNC4yIiBjbGFzcz0ibHR4X3RkIGx0eF9ub3BhZF9yIGx0eF9hbGlnbl9sZWZ0IGx0eF9hbGlnbl90b3AiIHN0eWxlPSJwYWRkaW5nLWJvdHRvbTogMy4wcHQ7cGFkZGluZy10b3A6MC42cHQ7cGFkZGluZy1ib3R0b206MC42cHQ7Ij4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjQuMi4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfYWxpZ25fdG9wIiBzdHlsZT0id2lkdGg6MzEwLjhwdDsiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNC4yLjEuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNC4yLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+V3JpdGUgYW5kIHJ1biA8L3NwYW4+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjQuMi4xLjEuMiIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+cHl0aG9uIHdvcmsvZmVhdHVyZXMucHk8L3NwYW4+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjQuMi4xLjEuMyIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPi4gRXh0cmFjdCBzdHJ1Y3R1cmFsIGZlYXR1cmVzIGFuZCB0b2tlbiBjb3VudHMgaW50byA8L3NwYW4+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjQuMi4xLjEuNCIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+ZmVhdHVyZXMucGFycXVldDwvc3Bhbj48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNC4yLjEuMS41IiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+OyBzYXZlIHRhc2sgZGVzY3JpcHRpb25zIGluIDwvc3Bhbj48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNC4yLjEuMS42IiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij50YXNrcy5wYXJxdWV0PC9zcGFuPjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS40LjIuMS4xLjciIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4uPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS41IiBjbGFzcz0ibHR4X3RyIj4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjUuMSIgY2xhc3M9Imx0eF90ZCBsdHhfYWxpZ25fbGVmdCBsdHhfYWxpZ25fdG9wIiBzdHlsZT0icGFkZGluZy1ib3R0b206IDMuMHB0O3BhZGRpbmctdG9wOjAuNnB0O3BhZGRpbmctYm90dG9tOjAuNnB0OyI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS41LjEuMSIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X2FsaWduX3RvcCIgc3R5bGU9IndpZHRoOjQ1LjVwdDsiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNS4xLjEuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNS4xLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+MDA6MDY6NDTigJMwMDowODo0Mzwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS41LjIiIGNsYXNzPSJsdHhfdGQgbHR4X25vcGFkX3IgbHR4X2FsaWduX2xlZnQgbHR4X2FsaWduX3RvcCIgc3R5bGU9InBhZGRpbmctYm90dG9tOiAzLjBwdDtwYWRkaW5nLXRvcDowLjZwdDtwYWRkaW5nLWJvdHRvbTowLjZwdDsiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNS4yLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9hbGlnbl90b3AiIHN0eWxlPSJ3aWR0aDozMTAuOHB0OyI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS41LjIuMS4xIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS41LjIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5Xcml0ZSA8L3NwYW4+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjUuMi4xLjEuMiIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+d29yay9zY29yZV9ncHUucHk8L3NwYW4+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjUuMi4xLjEuMyIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPjsgcGlsb3Qgd2l0aCA8L3NwYW4+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjUuMi4xLjEuNCIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+dGltZW91dCA5MDAgcHl0aG9uIHdvcmsvc2NvcmVfdGVzdC5weTwvc3Bhbj48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNS4yLjEuMS41IiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+LiBSZXZpc2Ugc2NvcmluZyBwcmlvcml0eSwgdGhlbiBsYXVuY2ggPC9zcGFuPjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS41LjIuMS4xLjYiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPnB5dGhvbiB3b3JrL3Njb3JlX2dwdS5weSAmZ3Q7IHdvcmsvZ3B1X2xvZy50eHQgMiZndDsmYW1wOzE8L3NwYW4+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjUuMi4xLjEuNyIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPjogZnJvemVuLW1vZGVsIGFzc2lzdGFudC10b2tlbiBsb3NzLCB1cCB0byAzMiw3NjggdG9rZW5zLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNiIgY2xhc3M9Imx0eF90ciI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS42LjEiIGNsYXNzPSJsdHhfdGQgbHR4X2FsaWduX2xlZnQgbHR4X2FsaWduX3RvcCIgc3R5bGU9InBhZGRpbmctYm90dG9tOiAzLjBwdDtwYWRkaW5nLXRvcDowLjZwdDtwYWRkaW5nLWJvdHRvbTowLjZwdDsiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNi4xLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9hbGlnbl90b3AiIHN0eWxlPSJ3aWR0aDo0NS41cHQ7Ij4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjYuMS4xLjEiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjYuMS4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPjAwOjEwOjI0PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjYuMiIgY2xhc3M9Imx0eF90ZCBsdHhfbm9wYWRfciBsdHhfYWxpZ25fbGVmdCBsdHhfYWxpZ25fdG9wIiBzdHlsZT0icGFkZGluZy1ib3R0b206IDMuMHB0O3BhZGRpbmctdG9wOjAuNnB0O3BhZGRpbmctYm90dG9tOjAuNnB0OyI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS42LjIuMSIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X2FsaWduX3RvcCIgc3R5bGU9IndpZHRoOjMxMC44cHQ7Ij4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjYuMi4xLjEiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjYuMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPklubGluZSBQeXRob24gY29tcHV0ZXMgdGFzayBkdXBsaWNhdGVzIGFuZCBURi1JREYgc2ltaWxhcml0eSwgZml0cyA8L3NwYW4+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjYuMi4xLjEuMiIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+TWluaUJhdGNoS01lYW5zKG5fY2x1c3RlcnM9NTApPC9zcGFuPjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS42LjIuMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4sIGFuZCBzYXZlcyA8L3NwYW4+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjYuMi4xLjEuNCIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+d29yay9mZWF0dXJlczIucGFycXVldDwvc3Bhbj48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNi4yLjEuMS41IiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Ljwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNyIgY2xhc3M9Imx0eF90ciI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS43LjEiIGNsYXNzPSJsdHhfdGQgbHR4X2FsaWduX2xlZnQgbHR4X2FsaWduX3RvcCIgc3R5bGU9InBhZGRpbmctYm90dG9tOiAzLjBwdDtwYWRkaW5nLXRvcDowLjZwdDtwYWRkaW5nLWJvdHRvbTowLjZwdDsiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNy4xLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9hbGlnbl90b3AiIHN0eWxlPSJ3aWR0aDo0NS41cHQ7Ij4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjcuMS4xLjEiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjcuMS4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPjAwOjEyOjAx4oCTMDA6MTQ6MzU8L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNy4yIiBjbGFzcz0ibHR4X3RkIGx0eF9ub3BhZF9yIGx0eF9hbGlnbl9sZWZ0IGx0eF9hbGlnbl90b3AiIHN0eWxlPSJwYWRkaW5nLWJvdHRvbTogMy4wcHQ7cGFkZGluZy10b3A6MC42cHQ7cGFkZGluZy1ib3R0b206MC42cHQ7Ij4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjcuMi4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfYWxpZ25fdG9wIiBzdHlsZT0id2lkdGg6MzEwLjhwdDsiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNy4yLjEuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNy4yLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+V3JpdGUgYW5kIHJlcGVhdGVkbHkgcnVuIDwvc3Bhbj48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuNy4yLjEuMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5weXRob24gd29yay9zZWxlY3QucHk8L3NwYW4+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjcuMi4xLjEuMyIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPi4gSW5zcGVjdCB0aGUgcHJvdmlzaW9uYWwgZ3JvdXBzOyByZXBsYWNlIGZpeGVkIGNsdXN0ZXIgY2FwcyB3aXRoIHBvb2wtcmVsYXRpdmUgY2FwcywgYWRkIGFuIDglIFRlem9zIGNhcCwgdGhlbiBhZGQgc291cmNlIGNhcHMuIFRoZXNlIGNoYW5nZXMgcHJlY2VkZSBkb3duc3RyZWFtIGV2YWx1YXRpb24uPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS44IiBjbGFzcz0ibHR4X3RyIj4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjguMSIgY2xhc3M9Imx0eF90ZCBsdHhfYWxpZ25fbGVmdCBsdHhfYWxpZ25fdG9wIiBzdHlsZT0icGFkZGluZy1ib3R0b206IDMuMHB0O3BhZGRpbmctdG9wOjAuNnB0O3BhZGRpbmctYm90dG9tOjAuNnB0OyI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS44LjEuMSIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X2FsaWduX3RvcCIgc3R5bGU9IndpZHRoOjQ1LjVwdDsiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuOC4xLjEuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuOC4xLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+MDA6NDA6NDM8L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuOC4yIiBjbGFzcz0ibHR4X3RkIGx0eF9ub3BhZF9yIGx0eF9hbGlnbl9sZWZ0IGx0eF9hbGlnbl90b3AiIHN0eWxlPSJwYWRkaW5nLWJvdHRvbTogMy4wcHQ7cGFkZGluZy10b3A6MC42cHQ7cGFkZGluZy1ib3R0b206MC42cHQ7Ij4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjguMi4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfYWxpZ25fdG9wIiBzdHlsZT0id2lkdGg6MzEwLjhwdDsiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuOC4yLjEuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuOC4yLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+SW5saW5lIFB5dGhvbiByZWFkcyA8L3NwYW4+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjguMi4xLjEuMiIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+d29yay9ncHVfc2NvcmVzLnBrbDwvc3Bhbj48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuOC4yLjEuMS4zIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+IGFuZCBjaGVja3MgbG9zcyBkaXN0cmlidXRpb25zLCBzb3VyY2UtbGV2ZWwgc3RhdGlzdGljcywgY29ycmVsYXRpb25zLCBhbmQgZXh0cmVtZS1sb3NzIGV4YW1wbGVzIHdoaWxlIEdQVSBzY29yaW5nIGNvbnRpbnVlcy48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjkiIGNsYXNzPSJsdHhfdHIiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuOS4xIiBjbGFzcz0ibHR4X3RkIGx0eF9hbGlnbl9sZWZ0IGx0eF9hbGlnbl90b3AiIHN0eWxlPSJwYWRkaW5nLWJvdHRvbTogMy4wcHQ7cGFkZGluZy10b3A6MC42cHQ7cGFkZGluZy1ib3R0b206MC42cHQ7Ij4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjkuMS4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfYWxpZ25fdG9wIiBzdHlsZT0id2lkdGg6NDUuNXB0OyI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS45LjEuMS4xIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS45LjEuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4wMjowNDozNzwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS45LjIiIGNsYXNzPSJsdHhfdGQgbHR4X25vcGFkX3IgbHR4X2FsaWduX2xlZnQgbHR4X2FsaWduX3RvcCIgc3R5bGU9InBhZGRpbmctYm90dG9tOiAzLjBwdDtwYWRkaW5nLXRvcDowLjZwdDtwYWRkaW5nLWJvdHRvbTowLjZwdDsiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuOS4yLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9hbGlnbl90b3AiIHN0eWxlPSJ3aWR0aDozMTAuOHB0OyI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS45LjIuMS4xIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS45LjIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5SZXJ1biA8L3NwYW4+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjkuMi4xLjEuMiIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+cHl0aG9uIHdvcmsvc2VsZWN0LnB5PC9zcGFuPjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS45LjIuMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij47IGluc3BlY3QgdGhlIENTViB3aXRoIDwvc3Bhbj48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuOS4yLjEuMS40IiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5oZWFkIC0zPC9zcGFuPjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS45LjIuMS4xLjUiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gYW5kIDwvc3Bhbj48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuOS4yLjEuMS42IiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij53YyAtbDwvc3Bhbj48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuOS4yLjEuMS43IiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+LiBJbmxpbmUgYXNzZXJ0aW9ucyBjaGVjayBjb2x1bW4gbmFtZXMsIHVuaXF1ZSBpbi1yYW5nZSBpbmRpY2VzLCBncm91cCBJRHMsIGFuZCBleGFjdGx5IDEsMDAwIHJvd3MgcGVyIGdyb3VwLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuMTAiIGNsYXNzPSJsdHhfdHIiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuMTAuMSIgY2xhc3M9Imx0eF90ZCBsdHhfYWxpZ25fbGVmdCBsdHhfYWxpZ25fdG9wIGx0eF9ib3JkZXJfYmIiIHN0eWxlPSJwYWRkaW5nLXRvcDowLjZwdDtwYWRkaW5nLWJvdHRvbTowLjZwdDsiPgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuMTAuMS4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfYWxpZ25fdG9wIiBzdHlsZT0id2lkdGg6NDUuNXB0OyI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS4xMC4xLjEuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iUzQuRjQucGljMS4xLjEuMTAuMS4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPjAyOjA1OjE4PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjEwLjIiIGNsYXNzPSJsdHhfdGQgbHR4X25vcGFkX3IgbHR4X2FsaWduX2xlZnQgbHR4X2FsaWduX3RvcCBsdHhfYm9yZGVyX2JiIiBzdHlsZT0icGFkZGluZy10b3A6MC42cHQ7cGFkZGluZy1ib3R0b206MC42cHQ7Ij4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjEwLjIuMSIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X2FsaWduX3RvcCIgc3R5bGU9IndpZHRoOjMxMC44cHQ7Ij4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjEwLjIuMS4xIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS4xMC4yLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5jYXQgJmd0OyZndDsgb3V0cHV0cy9tZXRob2QubWQgJmx0OyZsdDvigJlFT0bigJk8L3NwYW4+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjEwLjIuMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij46IGFwcGVuZCBmaW5hbCBncm91cCBzdGF0aXN0aWNzIGFuZCBjb21wdXRlIGRldGFpbHM7IGxpc3QgdGhlIGRlbGl2ZXJlZCBmaWxlcyBpbiA8L3NwYW4+PHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4xLjEwLjIuMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPm91dHB1dHMvPC9zcGFuPjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMS4xMC4yLjEuMS40IiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Ljwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4yIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMi4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5GaW5hbCByZWNpcGUuPHNwYW4gaWQ9IlM0LkY0LnBpYzEuMS4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X21lZGl1bSI+CjxzcGFuIGlkPSJTNC5GNC5waWMxLjEuMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2l0YWxpYyIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNDQzY2MDA7Ij5TY29yZS48L3NwYW4+IENvbWJpbmUgY29tcGxldGlvbiAoPG1hdGggaWQ9IlM0LkY0LnBpYzEubTEiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iKzIiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPis8L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MjwvbW4+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+KzI8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiksIHVzYWJsZSBhc3Npc3RhbnQtdG9rZW4gc3VwZXJ2aXNpb24sCmNvbW1hbmQgc3Vic3RhbmNlLCBhbmQgd3JpdGluZy90ZXN0aW5nIGN1ZXMgd2l0aCBzb2Z0IHBlbmFsdGllcyBmb3IgZm9ybWF0CmZhdWx0cywgY29udGV4dCBoYW5kb2ZmLCB0cnVuY2F0aW9uLCBlbXB0eSBvciByZXBldGl0aXZlIGFjdGlvbnMsIGFuZCBuYXJyb3cKZG9tYWluIGNvdmVyYWdlLiBBZGQgbG9zcy1vdXRsaWVyIHBlbmFsdGllcyBmb3IgYm90aCB1bnVzdWFsbHkgaGlnaCBhbmQgbG93CmF2ZXJhZ2UgbG9zcywgaGlnaC1sb3NzIHRva2VuIGZyYWN0aW9ucywgYW5kIGV4dHJlbWUgbGFzdC9tYXgtdHVybiBsb3NzOwpsb3NzIGlzIG5vdCBhIG1vbm90b25pYyBwcmVmZXJlbmNlIGZvciBoYXJkZXIgb3IgZWFzaWVyIGV4YW1wbGVzLgo8c3BhbiBpZD0iUzQuRjQucGljMS4xLjIuMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9pdGFsaWMiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojQ0M2NjAwOyI+Q29uc3RydWN0IGdyb3Vwcy48L3NwYW4+IFNvcnQgYnkgY29tcG9zaXRlIHNjb3JlLiBGb3IgPG1hdGggaWQ9IlM0LkY0LnBpYzEubTIiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iZ197MX0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPmc8L21pPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MTwvbW4+PC9tc3ViPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+Z197MX08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiwgZ3JlZWRpbHkgdGFrZQoxLDAwMCByb3dzIHdpdGggb25lIHRyYWplY3RvcnkgcGVyIHRhc2sgYW5kIGNhcCBjbHVzdGVyIDxtYXRoIGlkPSJTNC5GNC5waWMxLm0zIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9ImMiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPmM8L21pPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+YzwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+IGF0CjxtYXRoIGlkPSJTNC5GNC5waWMxLm00IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxtYXgoMTIsXGxjZWlsIDAuMjBuX3tjfVxyY2VpbCkiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPm1heDwvbWk+PG1vPuKBoTwvbW8+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPig8L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MTI8L21uPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+LDwvbW8+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPuKMiDwvbW8+PG1yb3c+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4wLjIwPC9tbj48bW8gbHNwYWNlPSIwZW0iIHJzcGFjZT0iMGVtIj7igIs8L21vPjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+bjwvbWk+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5jPC9taT48L21zdWI+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj7ijIk8L21vPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIHN0cmV0Y2h5PSJmYWxzZSI+KTwvbW8+PC9tcm93PjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlxtYXgoMTIsXGxjZWlsIDAuMjBuX3tjfVxyY2VpbCk8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiwgd2hlcmUgPG1hdGggaWQ9IlM0LkY0LnBpYzEubTUiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0ibl97Y30iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPm48L21pPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+YzwvbWk+PC9tc3ViPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+bl97Y308L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiBpcyBpdHMgcG9vbCBzaXplLiBSZXBlYXQgb24KdW5hc3NpZ25lZCByb3dzIGZvciA8bWF0aCBpZD0iUzQuRjQucGljMS5tNiIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJnX3syfSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+ZzwvbWk+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4yPC9tbj48L21zdWI+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5nX3syfTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+LCB1c2luZyA8bWF0aCBpZD0iUzQuRjQucGljMS5tNyIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSIwLjI1bl97Y30iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjAuMjU8L21uPjxtbyBsc3BhY2U9IjBlbSIgcnNwYWNlPSIwZW0iPuKAizwvbW8+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5uPC9taT48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPmM8L21pPjwvbXN1Yj48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij4wLjI1bl97Y308L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPi4gQm90aCBncm91cHMgY2FwIHNvZnR3YXJlLWVuZ2luZWVyaW5nLApzeW50aGV0aWMsIGFuZCBvdGhlciBzb3VyY2VzIGF0IDQwJSwgNDUlLCBhbmQgMzUlLCByZXNwZWN0aXZlbHksIGFuZCBUZXpvcwphdCA4JS4gQXNzaWduIHRoZSBuZXh0IHR3byBoaWdoZXN0LXNjb3JpbmcgYmxvY2tzIHRvIDxtYXRoIGlkPSJTNC5GNC5waWMxLm04IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9ImdfezN9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5nPC9taT48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjM8L21uPjwvbXN1Yj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPmdfezN9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4gYW5kIDxtYXRoIGlkPSJTNC5GNC5waWMxLm05IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9ImdfezR9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5nPC9taT48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjQ8L21uPjwvbXN1Yj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPmdfezR9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4gd2l0aG91dAp0aGVzZSBkaXZlcnNpdHkgY2FwczsgYXNzaWduIHRoZSBsb3dlc3Qtc2NvcmluZyByZW1haW5pbmcgMSwwMDAgcm93cyB0bwo8bWF0aCBpZD0iUzQuRjQucGljMS5tMTAiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iZ197NX0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPmc8L21pPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NTwvbW4+PC9tc3ViPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+Z197NX08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPi4gRGVsaXZlciBmaXZlIGRpc2pvaW50IGdyb3VwcyBpbiA8c3BhbiBpZD0iUzQuRjQucGljMS4xLjIuMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5ncm91cF9hc3NpZ25tZW50LmNzdjwvc3Bhbj4gYW5kCmV4cGxhaW4gdGhlIHByb2NlZHVyZSBpbiA8c3BhbiBpZD0iUzQuRjQucGljMS4xLjIuMS4xLjQiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5tZXRob2QubWQ8L3NwYW4+Ljwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=)

Figure 4: Fable 5.1’s data-selection workflow, from inspection to final
recipe.

### 4.3 How Agents Make Selection Decisions

We analyze 30 TBLite and 18 BFCL selection procedures and identify three
patterns. Common signals, different value judgments. Across TBLite
recipes, completion cues appear in 30 procedures, length features in 29,
format checks in 28, base-model loss in 23, and diversity controls in 21. Yet agents interpret these signals differently. One DeepSeek V4 Pro
recipe rewards higher first-assistant-turn loss after penalizing
formatting defects:
$`q_{\mathrm{DS}}=\ell(1-0.10e-0.05w-0.50j)(0.5+0.5c)`$, where $`e,w,j`$
are extra-text, warning, and invalid-JSON rates, and $`c`$ indicates
completion. Kimi K3 instead favors lower length-normalized loss:
$`q_{\mathrm{Kimi}}=-z_{\mathrm{length}}(\ell)+1.2c-2t,`$ where loss is
standardized within length deciles and $`t`$ indicates truncation. Fable
5.1 uses several loss statistics as anomaly penalties on a structural
score, with task coverage and source caps. These are different judgments
of training value, although their loss measurements also differ in
coverage and normalization. Recipes change across independent runs. The
same agent can use different data-selection rules across runs. For
example, Astra’s three TBLite runs assess quality through generated
correctness audits, separate prompted judgments of demonstration
quality, task accomplishment and complexity, and a five-level rating
with execution-evidence checks, respectively. The variation also occurs
in BFCL: Kimi moves from rejecting loss outliers to preferring
intermediate loss and then lower loss. Thus, a single run’s recipe does
not fully characterize an agent’s selection strategy. Agents construct
the predicted ordering in different ways. The ordering is part of the
selection procedure: agents can partition a scored candidate list or
apply different selection criteria to different groups. Astra’s third
TBLite run combines these approaches, constructing $`g_{1}`$ with
deduplication and skill coverage before drawing the remaining groups
from descending score bands. In BFCL, Fable 5.1’s first run places its
lowest-scoring candidates in $`g_{5}`$, whereas Gemini’s second run
explicitly assigns looping and error-heavy trajectories to that group.
These procedures encode the agents’ predictions of relative training
value;
Section [1](#S4.T1 "Table 1 ‣ 4.2 Selection Quality and Ranking Accuracy ‣ 4 Experiments ‣ DataSense-Bench: The First Step Toward an AI Scientist")
evaluates how well those predictions match post-training performance.

### 4.4 Case Study

Figure [4](#S4.F4 "Figure 4 ‣ 4.2 Selection Quality and Ranking Accuracy ‣ 4 Experiments ‣ DataSense-Bench: The First Step Toward an AI Scientist")
shows Fable 5.1’s data selection for the terminal task, including data
inspection and the final recipe. The agent first builds structural
features and base-model loss statistics, then revises its grouping rules
after inspecting the resulting subsets. Its final recipe uses loss to
penalize outliers. It also applies task, cluster, and source constraints
to $`g_{1}`$ and $`g_{2}`$, while constructing later groups from score
bands without those diversity caps. This case shows that the submitted
ordering combines example-level scores with group-level coverage
choices. In this run, $`g_{1}`$ achieves 14.83% reward, but $`g_{2}`$
(11.08%) falls below $`g_{3}`$ (12.12%). The best first choice therefore
coexists with an error in the remaining order.
Appendix [B.1](#A2.SS1 "B.1 Selection Procedures ‣ Appendix B Detailed Experimental Results ‣ DataSense-Bench: The First Step Toward an AI Scientist")
and
Table [6](#A2.T6 "Table 6 ‣ B.2 BFCL Selection Procedures ‣ Appendix B Detailed Experimental Results ‣ DataSense-Bench: The First Step Toward an AI Scientist")
provide the remaining recipes.

### 4.5 Resource Requirements

The benchmark separates the cost of making a data recommendation from
the cost of testing it through SFT and downstream evaluation.

CLI agent cost and time budget.
Table [2](#S4.T2 "Table 2 ‣ 4.5 Resource Requirements ‣ 4 Experiments ‣ DataSense-Bench: The First Step Toward an AI Scientist")
summarizes the agent API cost for data selection. Selection estimates
use cumulative token ledgers and exclude local inference, training, and
evaluation charges.
Figure [2](#S4.T2 "Table 2 ‣ 4.5 Resource Requirements ‣ 4 Experiments ‣ DataSense-Bench: The First Step Toward an AI Scientist")
shows time cost against the four-hour time budget, including tool
execution and waits. Measurement details and pricing assumptions are in
Appendix [A.5](#A1.SS5 "A.5 Cost and Runtime Measurement ‣ Appendix A Experimental Details ‣ DataSense-Bench: The First Step Toward an AI Scientist").

|                                                                             |           |              |             |           |           |
| --------------------------------------------------------------------------- | --------- | ------------ | ----------- | --------- | --------- |
| Agent                                                                       | TBLite    | BFCL         | Agent       | TBLite    | BFCL      |
| Fable 5.1                                                                   | 5.98      | 5.35         | Opus 5      | 3.69      | 4.91      |
| GPT-6 Astra                                                                 | 11.11     | $`\geq`$9.35 | Kimi K3     | 2.41      | 1.38      |
| DeepSeek                                                                    | 0.28–0.57 | 0.38–0.78    | Gemini      | 1.11–1.34 | 0.84–1.01 |
| Fable 5                                                                     | 5.10      | –            | Sonnet 5    | 3.63      | –         |
| GPT-5.5                                                                     | 2.54      | –            | GPT-5.6 Sol | 0.70      | –         |
| Training / evaluation (h): TBLite 2.04 / $`\approx`$3.93; BFCL 0.29 / 3.30. |           |              |             |           |           |

Table 2: Mean selection cost (USD; ranges reflect billing assumptions)
and training/evaluation hours per checkpoint (medians; TBLite evaluation
estimated).

Figure 5: Time cost in data selection. The dashed line marks the time
budget.

Training and evaluation time. Evaluation is the larger time component on
both tasks
(Table [2](#S4.T2 "Table 2 ‣ 4.5 Resource Requirements ‣ 4 Experiments ‣ DataSense-Bench: The First Step Toward an AI Scientist")).
Each selection requires five trained checkpoints. TBLite evaluation time
is a scheduling estimate for 300 attempts with 32 concurrent trials on
two H100 GPUs, including model loading; it is not a measured end-to-end
runtime. BFCL medians cover 90 training runs and 35 complete checkpoint
evaluations, with all three evaluation seeds sharing one model server on
one GPU.

## 5 Conclusion

Anticipating which changes will improve a model can help recursive
self-improvement allocate training resources more effectively. As these
systems take on research roles, data sense becomes a concrete part of
that challenge: connecting properties of training examples to what a
model will learn from them. We introduce DataSense-Bench to evaluate
this data sense through selecting useful subsets and predicting their
relative post-training performance. Across terminal problem solving and
multi-turn tool use, agents construct plausible, data-specific recipes,
but selection gains over random selection are limited. Agents do not
reliably rank their selected groups, and this ability does not hold
consistently across tasks. These results highlight a gap between
producing a convincing analysis of data and making reliable decisions
about its training value, motivating data sense as a capability to
evaluate when assessing progress toward AI scientists and their
recursive self-improvement.

Limitations and future work. Our benchmark studies data sense through
selecting and ranking existing data for SFT. Future work should extend
this scope to reinforcement-learning post-training and the generation of
new training data, testing whether agents can anticipate which
experiences to collect or create and how those choices will improve
learning.

## AI Use Statement

OpenAI Codex assists with drafting data-selection prompts, preparing
code snapshots, monitoring experiments, and polishing the manuscript.
Frontier models also act as experimental participants, as described in
Section [3.1](#S3.SS1 "3.1 Task for CLI Agents: Select Useful Data and Predict Its Value ‣ 3 Setup of DataSense-Bench ‣ DataSense-Bench: The First Step Toward an AI Scientist").
Numerical results are computed from the recorded selections and
downstream evaluations. The authors are responsible for the final text,
methods, results, and claims.

## Broader Impact

As AI systems increasingly influence how models are trained, their data
choices can shape whose needs and experiences those models serve.
DataSense-Bench makes these choices testable, supporting closer scrutiny
of claims about autonomous AI research and recursive self-improvement.
Better data selection could reduce wasted training compute, but
optimizing benchmark performance alone may favor narrow capabilities or
amplify biases in the candidate data. Strong results should therefore be
considered alongside data provenance, privacy, and representativeness.
Repeatedly selecting data for aggregate performance could exclude rare
but important cases. Human review of these choices remains necessary,
especially when the resulting models are used in settings where errors
affect people. We hope this work supports evidence-based oversight of AI
research agents.

## References

- Brown et al. (2020) Tom Brown, Benjamin Mann, Nick Ryder, Melanie
  Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav
  Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel
  Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya
  Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark
  Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack
  Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya
  Sutskever, and Dario Amodei. Language models are few-shot learners. In
  H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.),
  _Advances in Neural Information Processing Systems_, volume 33, pp.
  1877–1901. Curran Associates, Inc., 2020. URL
  [https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf).
- Chan et al. (2025) Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James
  Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon
  Maksin, Tejal Patwardhan, Aleksander Madry, and Lilian Weng.
  Mle-bench: Evaluating machine learning agents on machine learning
  engineering. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.),
  _International Conference on Learning Representations_, volume 2025,
  pp. 50466–50494, 2025. URL
  [https://proceedings.iclr.cc/paper_files/paper/2025/file/7e3767db483c942b883eb4f8cfb74e31-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/7e3767db483c942b883eb4f8cfb74e31-Paper-Conference.pdf).
- Chen et al. (2024) Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa
  Gunaratna, Vikas Yadav, Zheng Tang, Vijay Srinivasan, Tianyi Zhou,
  Heng Huang, and Hongxia Jin. Alpagasus: Training a better alpaca with
  fewer data. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan,
  and Y. Sun (eds.), _International Conference on Learning
  Representations_, volume 2024, pp. 34767–34797, 2024. URL
  [https://proceedings.iclr.cc/paper_files/paper/2024/file/9543942c237ded1b39b1fd37259ff88e-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2024/file/9543942c237ded1b39b1fd37259ff88e-Paper-Conference.pdf).
- Chen et al. (2021) Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan,
  Henrique Pondé, Jared Kaplan, Harrison Edwards, Yura Burda, Nicholas
  Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael
  Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan,
  Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser,
  Mo Bavarian, Clemens S. Winter, Phil Tillet, Felipe Petroski Such,
  David W. Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth
  Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Igor
  Babuschkin, Suchir Balaji, Shantanu Jain, Andrew Carr, Jan Leike, Josh
  Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew M. Knight,
  Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew,
  Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba.
  Evaluating large language models trained on code. _ArXiv_,
  abs/2107.03374, 2021. URL
  [https://api.semanticscholar.org/CorpusID:235755472](https://api.semanticscholar.org/CorpusID:235755472).
- Chen et al. (2025) Ziru Chen, Shijie Chen, Yuting Ning, Qianheng
  Zhang, Boshi Wang, Botao Yu, Yifei Li, Zeyi Liao, Chen Wei, Zitong Lu,
  Vishal Dey, Mingyi Xue, Frazier N. Baker, Benjamin Burns, Daniel
  Adu-Ampratwum, Xuhui Huang, Xia Ning, Song Gao, Yu Su, and Huan Sun.
  Scienceagentbench: Toward rigorous assessment of language agents for
  data-driven scientific discovery. In Y. Yue, A. Garg, N. Peng, F. Sha,
  and R. Yu (eds.), _International Conference on Learning
  Representations_, volume 2025, pp. 96934–96990, 2025. URL
  [https://proceedings.iclr.cc/paper_files/paper/2025/file/f12b4df26344f3be803c06b555252efe-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/f12b4df26344f3be803c06b555252efe-Paper-Conference.pdf).
- Ghorbani & Zou (2019) Amirata Ghorbani and James Zou. Data shapley:
  Equitable valuation of data for machine learning. In Kamalika
  Chaudhuri and Ruslan Salakhutdinov (eds.), _Proceedings of the 36th
  International Conference on Machine Learning_, volume 97 of
  _Proceedings of Machine Learning Research_, pp. 2242–2251. PMLR, 09–15
  Jun 2019. URL
  [https://proceedings.mlr.press/v97/ghorbani19c.html](https://proceedings.mlr.press/v97/ghorbani19c.html).
- Huang et al. (2024) Qian Huang, Jian Vora, Percy Liang, and Jure
  Leskovec. MLAgentBench: Evaluating language agents on machine learning
  experimentation. In Ruslan Salakhutdinov, Zico Kolter, Katherine
  Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix
  Berkenkamp (eds.), _Proceedings of the 41st International Conference
  on Machine Learning_, volume 235 of _Proceedings of Machine Learning
  Research_, pp. 20271–20309. PMLR, 21–27 Jul 2024. URL
  [https://proceedings.mlr.press/v235/huang24y.html](https://proceedings.mlr.press/v235/huang24y.html).
- Ilyas et al. (2022) Andrew Ilyas, Sung Min Park, Logan Engstrom,
  Guillaume Leclerc, and Aleksander Madry. Datamodels: Understanding
  predictions with data and data with predictions. In Kamalika
  Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and
  Sivan Sabato (eds.), _Proceedings of the 39th International Conference
  on Machine Learning_, volume 162 of _Proceedings of Machine Learning
  Research_, pp. 9525–9587. PMLR, 17–23 Jul 2022. URL
  [https://proceedings.mlr.press/v162/ilyas22a.html](https://proceedings.mlr.press/v162/ilyas22a.html).
- Jimenez et al. (2024) Carlos E Jimenez, John Yang, Alexander Wettig,
  Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench:
  Can language models resolve real-world github issues? In B. Kim,
  Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.),
  _International Conference on Learning Representations_, volume 2024,
  pp. 54107–54157, 2024. URL
  [https://proceedings.iclr.cc/paper_files/paper/2024/file/edac78c3e300629acfe6cbe9ca88fb84-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2024/file/edac78c3e300629acfe6cbe9ca88fb84-Paper-Conference.pdf).
- Jin et al. (2025) Jiarui Jin, Yuwei Wu, Haoxuan Li, Xiaoting He,
  Weinan Zhang, Yiming Yang, Yong Yu, Jun Wang, and Mengyue Yang. Large
  language models are demonstration pre-selectors for themselves. In
  Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix
  Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.),
  _Proceedings of the 42nd International Conference on Machine
  Learning_, volume 267 of _Proceedings of Machine Learning Research_,
  pp. 28157–28186. PMLR, 13–19 Jul 2025. URL
  [https://proceedings.mlr.press/v267/jin25i.html](https://proceedings.mlr.press/v267/jin25i.html).
- Koh & Liang (2017) Pang Wei Koh and Percy Liang. Understanding
  black-box predictions via influence functions. In Doina Precup and
  Yee Whye Teh (eds.), _Proceedings of the 34th International Conference
  on Machine Learning_, volume 70 of _Proceedings of Machine Learning
  Research_, pp. 1885–1894. PMLR, 06–11 Aug 2017. URL
  [https://proceedings.mlr.press/v70/koh17a.html](https://proceedings.mlr.press/v70/koh17a.html).
- Kwon et al. (2024) Yongchan Kwon, Eric Wu, Kevin Wu, and James Y Zou.
  Datainf: Efficiently estimating data influence in lora-tuned llms and
  diffusion models. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki,
  M. Khan, and Y. Sun (eds.), _International Conference on Learning
  Representations_, volume 2024, pp. 21921–21942, 2024. URL
  [https://proceedings.iclr.cc/paper_files/paper/2024/file/5e84a0f233611a1dc8fb794dc52415a3-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2024/file/5e84a0f233611a1dc8fb794dc52415a3-Paper-Conference.pdf).
- Li et al. (2024) Ming Li, Yong Zhang, Zhitao Li, Jiuhai Chen, Lichang
  Chen, Ning Cheng, Jianzong Wang, Tianyi Zhou, and Jing Xiao. From
  quantity to quality: Boosting LLM performance with self-guided data
  selection for instruction tuning. In Kevin Duh, Helena Gomez, and
  Steven Bethard (eds.), _Proceedings of the 2024 Conference of the
  North American Chapter of the Association for Computational
  Linguistics: Human Language Technologies (Volume 1: Long Papers)_, pp.
  7602–7635, Mexico City, Mexico, June 2024. Association for
  Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.421. URL
  [https://aclanthology.org/2024.naacl-long.421/](https://aclanthology.org/2024.naacl-long.421/).
- Liu et al. (2025) Qian Liu, Xiaosen Zheng, Niklas Muennighoff,
  Guangtao Zeng, Longxu Dou, Tianyu Pang, Jing Jiang, and Min Lin.
  Regmix: Data mixture as regression for language model pre-training. In
  _The Thirteenth International Conference on Learning
  Representations_, 2025. URL
  [https://openreview.net/forum?id=5BjQOUXq7i](https://openreview.net/forum?id=5BjQOUXq7i).
- Lu et al. (2026) Chris Lu, Cong Lu, Robert Tjarko Lange, Yutaro
  Yamada, Shengran Hu, Jakob Foerster, David Ha, and Jeff Clune. Towards
  end-to-end automation of AI research. _Nature_, 651:914–919, 2026.
  doi: 10.1038/s41586-026-10265-5. URL
  [https://www.nature.com/articles/s41586-026-10265-5](https://www.nature.com/articles/s41586-026-10265-5).
- Majumder et al. (2025) Bodhisattwa Prasad Majumder, Harshit Surana,
  Dhruv Agarwal, Bhavana Dalvi Mishra, Abhijeetsingh Meena, Aryan
  Prakhar, Tirth Vora, Tushar Khot, Ashish Sabharwal, and Peter Clark.
  Discoverybench: Towards data-driven discovery with large language
  models. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.),
  _International Conference on Learning Representations_, volume 2025,
  pp. 4556–4579, 2025. URL
  [https://proceedings.iclr.cc/paper_files/paper/2025/file/0d70af566e69f1dfb687791ecf955e28-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/0d70af566e69f1dfb687791ecf955e28-Paper-Conference.pdf).
- Merrill et al. (2026) Mike Merrill, Alexander Shaw, Nicholas Carlini,
  Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Shin, Thomas
  Walshe, E. Kelly Buchanan, Junhong Shen, Guanghao Ye, Haowei Lin,
  Jason Poulos, Maoyu Wang, Marianna Nezhurina, Di Lu, Orfeas
  Menis Mastromichalakis, Zhiwei Xu, Zizhao Chen, Yue Liu, Robert Zhang,
  Leon Liangyu Chen, Anurag Kashyap, Jan-Lucas Uslu, Jeffrey Li, Jianbo
  Wu, Minghao Yan, Song Bian, Vedang Sharma, Ke Sun, Steven Dillmann,
  Akshay Anand, Andrew Lanpouthakoun, Bardia Koopah, Changran Hu, Etash
  Guha, Gabriel Dreiman, Jiacheng Zhu, Karl Krauth, Li Zhong, Niklas
  Muennighoff, Robert Amanfu, Shangyin Tan, Shreyas Pimpalgaonkar,
  Tushar Aggarwal, Xiangning Lin, Xin Lan, Xuandong Zhao, Yiqing Liang,
  Yuanli Wang, Zilong (Ryan) Wang, Changzhi Zhou, David Heineman, Hange
  Liu, Harsh Trivedi, John Yang, Junhong Lin, Manish Shetty, Michael
  Yang, Nabil Omi, Negin Raoof, Shanda Li, Terry Yue Zhuo, Wuwei Lin,
  Yiwei Dai, Yuxin Wang, Wenhao Chai, Shang Zhou, Dariush Wahdany, Ziyu
  She, Jiaming Hu, Zhikang Dong, Yuxuan Zhu, Sasha Cui, Ahson Saiyed,
  Arinbjörn Kolbeinsson, Christopher Rytting, Ryan Marten, Yixin Wang,
  Jenia Jitsev, Alex Dimakis, Andy Konwinski, and Ludwig Schmidt.
  Terminal-bench: Benchmarking agents on hard, realistic tasks in
  command line interfaces. In C. Vondrick, B. Hariharan, C. Raffel,
  L. Pinto, D. Yang, and A. Faust (eds.), _International Conference on
  Learning Representations_, volume 2026, pp. 40903–40986, 2026. URL
  [https://proceedings.iclr.cc/paper_files/paper/2026/file/444a3737adaee10d86ad2ef5f74468e6-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2026/file/444a3737adaee10d86ad2ef5f74468e6-Paper-Conference.pdf).
- Mindermann et al. (2022) Sören Mindermann, Jan M Brauner, Muhammed T
  Razzak, Mrinank Sharma, Andreas Kirsch, Winnie Xu, Benedikt Höltgen,
  Aidan N Gomez, Adrien Morisot, Sebastian Farquhar, and Yarin Gal.
  Prioritized training on points that are learnable, worth learning, and
  not yet learnt. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song,
  Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), _Proceedings of
  the 39th International Conference on Machine Learning_, volume 162 of
  _Proceedings of Machine Learning Research_, pp. 15630–15649. PMLR,
  17–23 Jul 2022. URL
  [https://proceedings.mlr.press/v162/mindermann22a.html](https://proceedings.mlr.press/v162/mindermann22a.html).
- Pruthi et al. (2020) Garima Pruthi, Frederick Liu, Satyen Kale, and
  Mukund Sundararajan. Estimating training data influence by tracing
  gradient descent. In H. Larochelle, M. Ranzato, R. Hadsell, M.F.
  Balcan, and H. Lin (eds.), _Advances in Neural Information Processing
  Systems_, volume 33, pp. 19920–19930. Curran Associates, Inc., 2020.
  URL
  [https://proceedings.neurips.cc/paper_files/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf).
- Rank et al. (2026) Ben Rank, Hardik Bhatnagar, Ameya Prabhu, Shira
  Eisenberg, Karina Nguyen, Matthias Bethge, and Maksym Andriushchenko.
  Posttrainbench: Can LLM agents automate LLM post-training? In
  _Forty-third International Conference on Machine Learning_, 2026. URL
  [https://openreview.net/forum?id=UnjxMTe57e](https://openreview.net/forum?id=UnjxMTe57e).
- Raoof et al. (2026) Negin Raoof, Richard Zhuang, Marianna Nezhurina,
  Etash Guha, Atula Tejaswi, Ryan Marten, Charlie F. Ruan, Tyler Griggs,
  Alexander Glenn Shaw, Hritik Bansal, E. Kelly Buchanan, Artem Gazizov,
  Reinhard Heckel, Chinmay Hegde, Sankalp Jajee, Daanish Khazi,
  Emmanouil Koukoumidis, Xiangyi Li, Hange Liu, Shlok Natarajan, Harsh
  Raj, Nicholas Roberts, Ethan Shen, Nishad Singhi, Michael Siu, Ashima
  Suvarna, Hanwen Xing, Patrick Yubeaton, Robert Zhang, Leon Liangyu
  Chen, Xiaokun Chen, Steven Dillmann, Saadia Gabriel, Xunyi Jiang,
  Anurag Kashyap, Boxuan Li, Yein Park, Minh Pham, Sujay Sanghavi, Lin
  Shi, Ke Sun, Yixin Wang, Zhiwei Xu, Erica Zhang, Siyan Zhao, Wanjia
  Zhao, Jenia Jitsev, Alex Dimakis, Benjamin Feuer, and Ludwig Schmidt.
  Openthoughts-agent: Data recipes for agentic models, 2026. URL
  [https://arxiv.org/abs/2606.24855](https://arxiv.org/abs/2606.24855).
- Siegel et al. (2024) Zachary S Siegel, Sayash Kapoor, Nitya Nadgir,
  Benedikt Stroebl, and Arvind Narayanan. CORE-bench: Fostering the
  credibility of published research through a computational
  reproducibility agent benchmark. _Transactions on Machine Learning
  Research_, 2024. ISSN 2835-8856. URL
  [https://openreview.net/forum?id=BsMMc4MEGS](https://openreview.net/forum?id=BsMMc4MEGS).
- Song et al. (2026) Xiaoshuai Song, Haofei Chang, Guanting Dong, Yutao
  Zhu, Ji-Rong Wen, and Zhicheng Dou. EnvScaler: Scaling
  tool-interactive environments for LLM agent via programmatic
  synthesis. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and
  David Jurgens (eds.), _Findings of the Association for Computational
  Linguistics: ACL 2026_, pp. 8326–8357, San Diego, California, United
  States, July 2026. Association for Computational Linguistics. ISBN
  979-8-89176-395-1. doi: 10.18653/v1/2026.findings-acl.407. URL
  [https://aclanthology.org/2026.findings-acl.407/](https://aclanthology.org/2026.findings-acl.407/).
- Spearman (1904) C. Spearman. The proof and measurement of association
  between two things. _The American Journal of Psychology_,
  15(1):72–101, 1904. doi: 10.2307/1412159. URL
  [https://www.jstor.org/stable/1412159](https://www.jstor.org/stable/1412159).
- Tirumala et al. (2023) Kushal Tirumala, Daniel Simig, Armen
  Aghajanyan, and Ari Morcos. D4: Improving llm pretraining via document
  de-duplication and diversification. In A. Oh, T. Naumann,
  A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), _Advances in
  Neural Information Processing Systems_, volume 36, pp. 53983–53995.
  Curran Associates, Inc., 2023. doi: 10.52202/075280-2348. URL
  [https://proceedings.neurips.cc/paper_files/paper/2023/file/a8f8cbd7f7a5fb2c837e578c75e5b615-Paper-Datasets_and_Benchmarks.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/a8f8cbd7f7a5fb2c837e578c75e5b615-Paper-Datasets_and_Benchmarks.pdf).
- Wijk et al. (2025) Hjalmar Wijk, Tao Roa Lin, Joel Becker, Sami
  Jawhar, Neev Parikh, Thomas Broadley, Lawrence Chan, Michael Chen,
  Joshua M Clymer, Jai Dhyani, Elena Ericheva, Katharyn Garcia, Brian
  Goodrich, Nikola Jurkovic, Megan Kinniment, Aron Lajko, Seraphina Nix,
  Lucas Jun Koba Sato, William Saunders, Maksym Taran, Ben West, and
  Elizabeth Barnes. RE-bench: Evaluating frontier AI r&d capabilities of
  language model agents against human experts. In Aarti Singh, Maryam
  Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan
  Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), _Proceedings of the 42nd
  International Conference on Machine Learning_, volume 267 of
  _Proceedings of Machine Learning Research_, pp. 66772–66832. PMLR,
  13–19 Jul 2025. URL
  [https://proceedings.mlr.press/v267/wijk25a.html](https://proceedings.mlr.press/v267/wijk25a.html).
- Xia et al. (2024) Mengzhou Xia, Sadhika Malladi, Suchin Gururangan,
  Sanjeev Arora, and Danqi Chen. LESS: Selecting influential data for
  targeted instruction tuning. In Ruslan Salakhutdinov, Zico Kolter,
  Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and
  Felix Berkenkamp (eds.), _Proceedings of the 41st International
  Conference on Machine Learning_, volume 235 of _Proceedings of Machine
  Learning Research_, pp. 54104–54132. PMLR, 21–27 Jul 2024. URL
  [https://proceedings.mlr.press/v235/xia24c.html](https://proceedings.mlr.press/v235/xia24c.html).
- Xie et al. (2023a) Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du,
  Hanxiao Liu, Yifeng Lu, Percy S Liang, Quoc V Le, Tengyu Ma, and
  Adams Wei Yu. Doremi: Optimizing data mixtures speeds up language
  model pretraining. In A. Oh, T. Naumann, A. Globerson, K. Saenko,
  M. Hardt, and S. Levine (eds.), _Advances in Neural Information
  Processing Systems_, volume 36, pp. 69798–69818. Curran Associates,
  Inc., 2023a. doi: 10.52202/075280-3059. URL
  [https://proceedings.neurips.cc/paper_files/paper/2023/file/dcba6be91359358c2355cd920da3fcbd-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/dcba6be91359358c2355cd920da3fcbd-Paper-Conference.pdf).
- Xie et al. (2023b) Sang Michael Xie, Shibani Santurkar, Tengyu Ma, and
  Percy S Liang. Data selection for language models via importance
  resampling. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt,
  and S. Levine (eds.), _Advances in Neural Information Processing
  Systems_, volume 36, pp. 34201–34227. Curran Associates, Inc., 2023b.
  doi: 10.52202/075280-1482. URL
  [https://proceedings.neurips.cc/paper_files/paper/2023/file/6b9aa8f418bde2840d5f4ab7a02f663b-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/6b9aa8f418bde2840d5f4ab7a02f663b-Paper-Conference.pdf).
- Zhou et al. (2023) Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan
  Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, LILI YU,
  Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy.
  Lima: Less is more for alignment. In A. Oh, T. Naumann, A. Globerson,
  K. Saenko, M. Hardt, and S. Levine (eds.), _Advances in Neural
  Information Processing Systems_, volume 36, pp. 55006–55021. Curran
  Associates, Inc., 2023. doi: 10.52202/075280-2400. URL
  [https://proceedings.neurips.cc/paper_files/paper/2023/file/ac662d74829e4407ce1d126477f4a03a-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/ac662d74829e4407ce1d126477f4a03a-Paper-Conference.pdf).

## Appendix A Experimental Details

### A.1 Training Configuration

Table [3](#A1.T3 "Table 3 ‣ A.1 Training Configuration ‣ Appendix A Experimental Details ‣ DataSense-Bench: The First Step Toward an AI Scientist")
lists the configurations for both tasks. We use full-parameter SFT with
assistant-token supervision; TBLite supervision includes thinking, and
its gradient-clipping threshold is $`10^{-3}`$. Selected BFCL
trajectories are converted into assistant-supervised examples before
training. Within each task, all agent-selected groups share the realized
training configuration.

|                                 |                          |                         |
| ------------------------------- | ------------------------ | ----------------------- |
|                                 | Terminal problem solving | Multi-turn tool use     |
| Training pool                   | OpenThoughts-Agent 10K   | EnvScaler SFT           |
| Groups $`\times`$ group size    | $`5\times 1,000`$        | $`5\times 50`$          |
| Base model                      | Qwen3-4B                 | Qwen3-4B                |
| Selectors / runs per selector   | 10 / 3                   | 6 / 3                   |
| Selection GPU / time budget     | H100 80GB / 4 h          | H100 80GB / 4 h         |
| Learning rate / epochs          | $`4\times 10^{-5}`$ / 7  | $`5\times 10^{-6}`$ / 2 |
| Effective batch / training seed | 96 / 42                  | 32 / 42                 |
| Training token cutoff           | 32,768                   | 16,384                  |
| Downstream evaluation           | TBLite: 100 tasks        | BFCL V3: 800 cases      |
| Main outcome                    | Average reward           | Overall accuracy        |
| Evaluation seeds                | 3                        | 3                       |

Table 3: Configuration of the two selection-and-ranking tasks. Group
sizes count complete trajectories.

### A.2 Selection Models and Inference Configuration

Table [4](#A1.T4 "Table 4 ‣ A.2 Selection Models and Inference Configuration ‣ Appendix A Experimental Details ‣ DataSense-Bench: The First Step Toward an AI Scientist")
maps the names in the results to the model identifiers used by the
selection agents. Claude Code session headers record version 2.1.270;
the retained deployment packages are Codex CLI 0.146.1 and Gemini CLI
0.60.0.

| Agent                                                            | Model identifier |
| ---------------------------------------------------------------- | ---------------- |
| [Fable 5.1](https://platform.claude.com/docs/en/models/overview) | claude-fable-5-1 |
| [Opus 5](https://platform.claude.com/docs/en/models/overview)    | claude-opus-5    |
| [GPT-6 Astra](https://developers.openai.com/api/docs/models)     | gpt-6-astra      |
| [Kimi K3](https://platform.kimi.ai/docs/overview)                | kimi-k3          |
| [DeepSeek V4 Pro](https://api-docs.deepseek.com/)                | deepseek-v4-pro  |
| [Gemini 3.5 Flash](https://ai.google.dev/gemini-api/docs/models) | gemini-3.5-flash |
| [Fable 5](https://platform.claude.com/docs/en/models/overview)   | claude-fable-5   |
| [Sonnet 5](https://platform.claude.com/docs/en/models/overview)  | claude-sonnet-5  |
| [GPT-5.5](https://developers.openai.com/api/docs/models)         | gpt-5.5          |
| [GPT-5.6 Sol](https://developers.openai.com/api/docs/models)     | gpt-5.6-sol      |

Table 4: Selection-agent model identifiers.

All selection CLIs run non-interactively with automatic tool approval.
Claude Code uses streaming JSON output and summarized thinking display;
Kimi and DeepSeek use their Anthropic-compatible endpoints through the
same CLI. Codex enables web search and detailed reasoning summaries. The
wrappers do not set a selection temperature, sampling seed, or
reasoning-effort override; these use the CLI/provider defaults.
Summary-display options affect the recorded trace and do not specify a
reasoning-token budget. Gemini uses streaming JSON output; its TBLite
token ledgers identify gemini-3.5-flash as the served model and
gemini-3-flash-preview as a helper, although the original request used
the alias gemini-3.8-flash.

### A.3 Downstream Evaluation and Result Aggregation

For TBLite, the model proposes terminal commands at each step;
Terminus-2 executes them and returns the terminal output for the model’s
next step. Harbor manages the task environment and runs the verifier
after the attempt. The conversation retains the model’s reasoning
between command executions. A vLLM server runs the evaluated checkpoint.
The configured temperature is 0.6, top-$`p`$ is 0.95, and top-$`k`$ is 20. The harness model limits are 32,768 input tokens, 40,960 total
tokens, and 7,168 output tokens. These are model-call limits, not a
total token budget for an entire task. The task-specific agent time
limits range from 300 to 3,600 seconds; the agent timeout multiplier is
one. Deployments use tensor parallelism 2.

BFCL evaluation uses a 65,536-token context and temperature 0.7. Each of
the three evaluation seeds covers the full 800-case BFCL V3 multi-turn
suite.

We average evaluation scores within each checkpoint before computing
run-level metrics, then average these metrics across independent
selection runs. Group-performance standard deviations measure variation
across selection runs. TBLite checkpoint scores include all 300 task
attempts. Ranking metrics compare all five groups within a valid
selection run. Failed submissions receive the fallback scores defined in
Section [3.3](#S3.SS3 "3.3 Evaluation Metrics ‣ 3 Setup of DataSense-Bench ‣ DataSense-Bench: The First Step Toward an AI Scientist")
and remain in the run-level averages. Random selection uses one fixed
subset per task; its repeated evaluations measure variability for that
subset. Full selection-level results appear in
Appendix [B.3](#A2.SS3 "B.3 Results by Selection ‣ Appendix B Detailed Experimental Results ‣ DataSense-Bench: The First Step Toward an AI Scientist").

### A.4 Selection-Time Checks and Recipe Information

A Python guard disables common training entry points, including
Tensor.backward and Trainer.train, while allowing forward passes.
Model-file metadata and training-related trace terms support further
review. Metadata are not weight-content hashes, and keyword matches can
arise from reading recipes; these checks alone do not certify that every
access or training restriction was enforced. We retain available
CLI/tool traces, analysis scripts, intermediate summaries, final
assignments, method reports, and validation outputs. Timestamps,
commands, outputs, and exit status establish the execution sequence
against which agent-written claims are checked.
Appendix [B.1](#A2.SS1 "B.1 Selection Procedures ‣ Appendix B Detailed Experimental Results ‣ DataSense-Bench: The First Step Toward an AI Scientist")
summarizes selection procedures and differences in record coverage.

Within each task, agents receive the same fixed training recipe, and all
selected groups are fine-tuned using that recipe. The training procedure
does not vary across agents or selection runs.

### A.5 Cost and Runtime Measurement

Selection costs in
Table [2](#S4.T2 "Table 2 ‣ 4.5 Resource Requirements ‣ 4 Experiments ‣ DataSense-Bench: The First Step Toward an AI Scientist")
are averaged across three runs per agent and task. The BFCL entries use
the recorded model-token ledgers and invocation durations for the 18
selections. Multiple recorded invocations belonging to the same
selection are combined before averaging the three runs. Durations
include tool execution and waits within those invocations, but exclude
gaps between invocations and scheduler queueing. Astra lacks complete
duration and usage coverage; its cost is therefore reported as a lower
bound and its duration is omitted. Token estimates use the same fixed
rate schedule as the TBLite comparison. GPU inference, SFT, and
evaluation costs are excluded. DeepSeek ranges vary peak/off-peak rates;
Gemini ranges vary the billing treatment of unclassified output tokens.
For reproducibility, the fixed USD-per-million-token rates for fresh
input, cache reads, cache writes, and output are $`(10,0.25,12.5,50)`$
for Fable 5.1, $`(5,0.5,6.25,25)`$ for Opus 5, $`(10,1,12.5,50)`$ for
Astra, and $`(3,0.3,3,15)`$ for Kimi K3. One-hour Claude cache writes
use twice the fresh-input rate. DeepSeek uses $`(0.66,0.022,0,1.98)`$ to
twice those rates. Gemini Flash uses $`(1.5,0.15,0,9)`$, and its
separately logged helper model uses $`(0.5,0.05,0,3)`$. These are
comparison assumptions, not measured charges.

Training times in
Table [2](#S4.T2 "Table 2 ‣ 4.5 Resource Requirements ‣ 4 Experiments ‣ DataSense-Bench: The First Step Toward an AI Scientist")
exclude queueing and model loading. The TBLite sample comprises 16
completed four-GPU jobs, with a median of 2.04 h and a range of
1.38–3.30 h. The BFCL sample comprises 90 completed 50-trajectory runs,
with a median of 0.29 h and an interquartile range of 0.25–0.38 h. BFCL
evaluation timing covers 35 completed single-GPU jobs, each evaluating
one checkpoint across all three evaluation seeds through a shared model
server. The median is 3.30 h, including model startup and excluding
queueing.

## Appendix B Detailed Experimental Results

### B.1 Selection Procedures

The TBLite summaries below distinguish signals used in the final method
from the rule that allocates groups. Runs 1–3 identify each agent’s
three reported selection attempts in order; they are not training seeds.
NLL denotes base-model negative log-likelihood on assistant tokens.
Completion and verification features are proxies extracted from
trajectories, not access to downstream evaluation outcomes. “Consecutive
blocks” means partitioning the selected score order into five equal
groups; “bands” means selecting separated portions of a ranking. Each
row describes one selection attempt, with its signals and final grouping
rule. Full execution records and scripts are retained separately from
these summaries.

Figure [4](#S4.F4 "Figure 4 ‣ 4.2 Selection Quality and Ranking Accuracy ‣ 4 Experiments ‣ DataSense-Bench: The First Step Toward an AI Scientist")
illustrates one Terminal procedure in detail. Each group contains 1,000
trajectories. The descriptions below summarize the executed selection
code.

| Agent            | Run | Signals used                                                                                             | Final grouping rule                                                                                                        |
| ---------------- | --- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Fable 5          | 1   | Structure, NLL tails and 64 task clusters.                                                               | Cluster-balanced $`g_{1}`$; spaced lower score bands.                                                                      |
|                  | 2   | Structure and partial NLL coverage.                                                                      | Task deduplication and cluster caps in $`g_{1}`$.                                                                          |
|                  | 3   | Structure and 8K-prefix NLL.                                                                             | Task-deduplicated $`g_{1}`$ without clustering; lower score windows.                                                       |
| Fable 5.1        | 1   | Structural quality with a small JSON-NLL term.                                                           | Task deduplication and cluster/style caps in $`g_{1}`$/$`g_{2}`$; lower score bands.                                       |
|                  | 2   | Structural quality, model loss and a prompted concreteness score.                                        | Task deduplication and cluster/style caps in $`g_{1}`$/$`g_{2}`$; lower score bands.                                       |
|                  | 3   | Structural quality with soft loss-anomaly penalties.                                                     | Cluster/source caps and an 8% Tezos cap in $`g_{1}`$/$`g_{2}`$; next-ranked $`g_{3}`$/$`g_{4}`$ and bottom $`g_{5}`$.      |
| Opus 5           | 1   | Completion, protocol and verification features plus model statistics.                                    | $`g_{1}`$ eligibility checks, task deduplication and cluster caps; lower score bands.                                      |
|                  | 2   | Quality, NLL and embeddings.                                                                             | Diversity-aware greedy $`g_{1}`$; lower score bands.                                                                       |
|                  | 3   | Execution evidence and length-adjusted action loss.                                                      | $`g_{1}`$ eligibility with soft similarity, cluster and source penalties; lower score bands.                               |
| Sonnet 5         | 1   | Completion, format and 4K-prefix loss.                                                                   | Top 5,000 split into consecutive groups; no hard task deduplication.                                                       |
|                  | 2   | Structural quality and preference for median loss.                                                       | Retains one example per task hash, then splits the top 5,000 into consecutive groups.                                      |
|                  | 3   | Format, loss and turn efficiency.                                                                        | Heap follows global score order; top 5,000 split consecutively without hard task deduplication.                            |
| GPT-5.5          | 1   | Structure, candidate NLL and task-term rarity.                                                           | Top 5,000 split consecutively; soft novelty adjustments without hard task deduplication.                                   |
|                  | 2   | Structure and 2K-prefix NLL; domain counts do not affect selection.                                      | Top 5,000 split consecutively; no hard task deduplication.                                                                 |
|                  | 3   | Structural scoring, then NLL reranking of the top 3,000.                                                 | Top 5,000 split consecutively; no hard task deduplication.                                                                 |
| GPT-5.6 Sol      | 1   | Completion, action, test/build and supervision features; tokenized lengths, no model loss.               | Top 5,000 split consecutively without hard task deduplication.                                                             |
|                  | 2   | Completion, action, test/build and supervision features; estimated truncation, no model loss.            | Top 5,000 split consecutively without hard task deduplication.                                                             |
|                  | 3   | Completion, action, test/build and supervision features; tokenized lengths, no model loss.               | Top 5,000 split consecutively; ties use test/coding evidence, length and row index. No hard task deduplication.            |
| GPT-6 Astra      | 1   | Prompted quality judgments, execution evidence, sampled NLL and task representations; staged review.     | Diverse $`g_{1}`$ with task deduplication and soft redundancy/coverage penalties; separated lower bands.                   |
|                  | 2   | Prompted quality judgments, execution evidence, sampled NLL and task representations; three assessments. | Diverse $`g_{1}`$ with task deduplication and soft redundancy/coverage penalties; separated lower bands.                   |
|                  | 3   | Prompted quality judgments, execution evidence, sampled NLL and task representations; five-level rubric. | $`g_{1}`$ completion/protocol/context eligibility, task deduplication and soft redundancy/coverage penalties; lower bands. |
| DeepSeek V4 Pro  | 1   | Structural value and embeddings; NLL discarded.                                                          | Similarity-penalized greedy order.                                                                                         |
|                  | 2   | Loss weighted by cleanliness and completion.                                                             | One example per task hash, then top-5,000 consecutive blocks.                                                              |
|                  | 3   | Structural activity and recovery; no NLL in the final score.                                             | Completed candidates, TF-IDF similarity filter at 0.92, then consecutive blocks.                                           |
| Kimi K3          | 1   | Completion, supervision and a small NLL term.                                                            | Task deduplication and $`g_{1}`$ cluster rebalancing.                                                                      |
|                  | 2   | Preference for moderate length-residual loss.                                                            | Similarity threshold and up to three examples per task in $`g_{1}`$.                                                       |
|                  | 3   | Negative length-normalized NLL plus completion/truncation terms.                                         | Cosine filter at 0.97 with fallback, then score-sorted blocks.                                                             |
| Gemini 3.5 Flash | 1   | Multiplicative structural score.                                                                         | Top 5,000 completed examples, split into consecutive blocks.                                                               |
|                  | 2   | Additive completion/error/length score.                                                                  | Top 4,000 for $`g_{1}`$–$`g_{4}`$; bottom 1,000 for $`g_{5}`$.                                                             |
|                  | 3   | NLL and thinking density on a premium candidate pool.                                                    | Premium $`g_{1}`$/$`g_{2}`$, other completed $`g_{3}`$, and stronger/weaker incomplete $`g_{4}`$/$`g_{5}`$.                |

Table 5: TBLite selection methods, with one row per attempt.

### B.2 BFCL Selection Procedures

We inspect the final method reports together with selection scripts or
command traces for the 18 runs using 50-trajectory groups. The
descriptions below summarize the rules used to form the submitted
groups. Quality, correctness, and difficulty refer to the agents’
selection signals, not independent labels of training utility.

| Agent            | Recipe and cross-run differences                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fable 5.1        | Run 1 combines schema checks, errors, reasoning length, and length-adjusted loss; it samples descending score bands, ending with low-scoring rows. Run 2 emphasizes reasoning–action consistency and loss outliers. Run 3 separately scores reasoning, actions given reasoning, and actions without reasoning, then combines quality gates with environment coverage. Later groups include increasingly defect-heavy trajectories.  |
| Opus 5           | Run 1 prioritizes structural quality and uses a small residualized-loss term, with fixed interaction-type and environment coverage across groups. Run 2 adds verbosity-normalized reasoning loss and recovered-error bonuses, placing noncanonical argument formats in the lower groups. Run 3 probes action difficulty without the demonstrated reasoning and selects widely separated score bands with distinct environments.     |
| GPT-6 Astra      | All runs combine schema checks, argument grounding, state-changing actions, clarification, and coverage. Run 1 adds sampled-turn loss over the pool and inspection of top candidates. Run 2 uses a bounded preference for moderate reasoning loss and a fixed 70/30 interaction-type mix. Run 3 scores a quality-filtered shortlist and adds semantic checks on unsupported actions. Groups occupy separated predicted-value bands. |
| Kimi K3          | Run 1 uses loss to penalize outliers, normalizes quality within interaction types, and selects successive groups under diversity caps. Run 2 favors intermediate total and tool-call loss. Run 3 instead rewards lower loss alongside successful responses and tool coverage, with environment caps relaxed from one to three across the groups.                                                                                    |
| DeepSeek V4 Pro  | Run 1 prioritizes high sampled-turn loss, trajectory length, and final-action loss. Run 2 combines loss with structural richness, including a positive failed-call term, and uses diversity-penalized greedy selection. Run 3 emphasizes tool breadth, penalizes failed-call rate, and adds within-environment loss as a secondary signal.                                                                                          |
| Gemini 3.5 Flash | Run 1 separates groups by observed tool success, length, and turn count. Run 2 uses a heuristic score penalizing errors and loops while rewarding reasoning length and intermediate turn counts. Run 3 adds base-model loss to choose long, error-free trajectories for the top groups; short clean trajectories and error-heavy trajectories occupy lower groups.                                                                  |

Table 6: BFCL selection recipes and differences across the three
independent runs. Each run produces five disjoint groups of 50 complete
trajectories.

### B.3 Results by Selection

These tables retain the individual selections underlying the averages in
the main text. Scores across evaluation seeds are averaged within each
checkpoint; ranking metrics are computed within each selection before
aggregation.

| Agent            | Run | $`g_{1}`$ | $`g_{2}`$ | $`g_{3}`$ | $`g_{4}`$ | $`g_{5}`$ | $`\rho`$ | $`g_{1}`$ rank |
| ---------------- | --- | --------- | --------- | --------- | --------- | --------- | -------- | -------------- |
| Fable 5          | r1  | 0.109     | 0.085     | 0.145     | 0.101     | 0.107     | 0.10     | 2              |
|                  | r2  | 0.100     | 0.110     | 0.121     | 0.110     | 0.099     | 0.30     | 4              |
|                  | r3  | 0.121     | 0.121     | 0.135     | 0.134     | 0.121     | -0.50    | 4              |
| Fable 5.1        | r1  | 0.117     | 0.122     | 0.129     | 0.098     | 0.116     | 0.50     | 3              |
|                  | r2  | 0.110     | 0.103     | 0.110     | 0.097     | 0.132     | -0.30    | 3              |
|                  | r3  | 0.148     | 0.111     | 0.121     | 0.112     | 0.101     | 0.70     | 1              |
| Opus 5           | r1  | 0.125     | 0.137     | 0.137     | 0.094     | 0.113     | 0.60     | 3              |
|                  | r2  | 0.115     | 0.127     | 0.097     | 0.135     | 0.117     | -0.30    | 4              |
|                  | r3  | 0.108     | 0.111     | 0.124     | 0.099     | 0.081     | 0.60     | 3              |
| Sonnet 5         | r1  | 0.099     | 0.110     | 0.116     | 0.124     | 0.149     | -1.00    | 5              |
|                  | r2  | 0.118     | 0.123     | 0.101     | 0.117     | 0.124     | -0.20    | 3              |
|                  | r3  | 0.130     | 0.125     | 0.103     | 0.113     | 0.115     | 0.60     | 1              |
| GPT-5.5          | r1  | 0.107     | 0.086     | 0.109     | 0.099     | 0.126     | -0.50    | 3              |
|                  | r2  | 0.128     | 0.103     | 0.133     | 0.127     | 0.122     | 0.20     | 2              |
|                  | r3  | 0.120     | 0.106     | 0.109     | 0.120     | 0.125     | -0.40    | 2              |
| GPT-5.6 Sol      | r1  | 0.130     | 0.111     | 0.126     | 0.133     | 0.148     | -0.70    | 3              |
|                  | r2  | 0.101     | 0.108     | 0.105     | 0.123     | 0.100     | 0.10     | 4              |
|                  | r3  | 0.107     | 0.118     | 0.111     | 0.131     | 0.135     | -0.90    | 5              |
| GPT-6 Astra      | r1  | 0.112     | 0.143     | 0.113     | 0.114     | 0.121     | -0.40    | 5              |
|                  | r2  | 0.105     | 0.145     | 0.097     | 0.136     | 0.114     | -0.10    | 4              |
|                  | r3  | 0.142     | 0.112     | 0.108     | 0.098     | 0.078     | 1.00     | 1              |
| DeepSeek V4 Pro  | r1  | 0.129     | 0.098     | 0.109     | 0.125     | 0.099     | 0.30     | 1              |
|                  | r2  | 0.135     | 0.133     | 0.104     | 0.120     | 0.132     | 0.60     | 1              |
|                  | r3  | 0.112     | 0.122     | 0.137     | 0.121     | 0.132     | -0.50    | 5              |
| Kimi K3          | r1  | 0.134     | 0.131     | 0.112     | 0.130     | 0.114     | 0.70     | 1              |
|                  | r2  | 0.105     | 0.091     | 0.108     | 0.110     | 0.107     | -0.60    | 4              |
|                  | r3  | 0.119     | 0.109     | 0.120     | 0.152     | 0.103     | 0.10     | 3              |
| Gemini 3.5 Flash | r1  | 0.115     | 0.137     | 0.102     | 0.137     | 0.135     | -0.30    | 4              |
|                  | r2  | 0.117     | 0.127     | 0.132     | 0.121     | 0.112     | 0.30     | 4              |
|                  | r3  | 0.130     | 0.139     | 0.120     | 0.119     | 0.082     | 0.90     | 2              |

Table 7: Terminal group-level results on a 0–1 reward scale. Runs use
the same 1–3 numbering as the procedure summaries. Each reward averages
100 tasks with three attempts per task. Within each row, shading runs
from darkest (best) to lightest (worst), and bold marks the best group.
Colors use unrounded means; equal means receive the same shade. $`\rho`$
compares the predicted $`g_{1}`$–$`g_{5}`$ order with observed
performance; $`g_{1}`$ rank is one plus the number of groups scoring
strictly higher.

| Agent            | Run | $`g_{1}`$ | $`g_{2}`$ | $`g_{3}`$ | $`g_{4}`$ | $`g_{5}`$ | $`\rho`$ | $`g_{1}`$ rank |
| ---------------- | --- | --------- | --------- | --------- | --------- | --------- | -------- | -------------- |
| Fable 5.1        | r1  | 36.00     | 33.25     | 34.83     | 34.38     | 37.17     | -0.30    | 2              |
|                  | r2  | 34.17     | 36.58     | 34.96     | 33.96     | 34.12     | 0.60     | 3              |
|                  | r3  | 35.46     | 35.04     | 34.83     | 33.96     | 35.42     | 0.40     | 1              |
| Opus 5           | r1  | 35.71     | 36.46     | 35.08     | 31.96     | 28.75     | 0.90     | 2              |
|                  | r2  | 36.75     | 37.38     | 35.00     | 36.54     | 29.46     | 0.80     | 2              |
|                  | r3  | 35.38     | 35.21     | 35.96     | 34.38     | 35.04     | 0.60     | 2              |
| GPT-6 Astra      | r1  | 36.38     | 34.58     | 33.21     | 34.71     | 34.50     | 0.50     | 1              |
|                  | r2  | 36.75     | 36.38     | 36.21     | 35.42     | 34.62     | 1.00     | 1              |
|                  | r3  | 36.33     | 34.96     | 34.79     | 34.67     | 34.62     | 1.00     | 1              |
| DeepSeek V4 Pro  | r1  | 35.29     | 37.42     | 35.33     | 35.29     | 33.58     | 0.56     | 3              |
|                  | r2  | 35.67     | 36.04     | 34.88     | 35.21     | 33.75     | 0.80     | 2              |
|                  | r3  | 38.62     | 36.38     | 37.00     | 37.75     | 35.67     | 0.60     | 1              |
| Kimi K3          | r1  | 35.25     | 36.08     | 35.42     | 34.54     | 36.42     | -0.30    | 4              |
|                  | r2  | 36.71     | 35.62     | 37.58     | 36.38     | 37.75     | -0.50    | 3              |
|                  | r3  | 36.92     | 36.92     | 35.54     | 36.62     | 35.21     | 0.87     | 1              |
| Gemini 3.5 Flash | r1  | 33.38     | 34.17     | 34.12     | 35.21     | 34.54     | -0.80    | 5              |
|                  | r2  | 35.50     | 33.08     | 33.21     | 34.79     | 35.92     | -0.40    | 2              |
|                  | r3  | 36.42     | 35.29     | 33.38     | 29.21     | 36.33     | 0.40     | 1              |

Table 8: BFCL accuracy (%) for 50-trajectory groups in each independent
selection run. Scores average the 800-case evaluations across evaluation
seeds. Shading compares groups within a run; $`\rho`$ measures
full-order agreement; $`g_{1}`$ rank is one plus the number of groups
scoring strictly higher.

## Appendix C Selection Prompts

### C.1 Selection Configuration Examples

Each prompt reads a local config.yaml. The examples below show the
selection-time settings for both tasks, with dataset paths written
relative to the workspace. The BFCL pool contains 8,962 trajectories
after excluding the 60-trajectory validation split. The max_len field
specifies the length limit supplied during selection; realized training
settings appear in
Appendix [A.1](#A1.SS1 "A.1 Training Configuration ‣ Appendix A Experimental Details ‣ DataSense-Bench: The First Step Toward an AI Scientist").

Terminal: config.yaml

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTMuU1MxLnAyLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSIgaGVpZ2h0PSIxMTguMjkiIG92ZXJmbG93PSJ2aXNpYmxlIiB2ZXJzaW9uPSIxLjEiIHZpZXdib3g9IjAgMCAyNjQgMTE4LjI5IiB3aWR0aD0iMjY0Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMDAwMDA7LS1sdHgtZmlsbC1jb2xvcjojMDAwMDAwOyIgZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjAuNHB0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDExOC4yOSkgbWF0cml4KDEgMCAwIC0xIDAgMCkiPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNEOUI5QTQ7IiBmaWxsPSIjRDlCOUE0IiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAgNC40OSBMIDAgMTEzLjggQyAwIDExNi4yOCAyLjAxIDExOC4yOSA0LjQ5IDExOC4yOSBMIDI1OS41MSAxMTguMjkgQyAyNjEuOTkgMTE4LjI5IDI2NCAxMTYuMjggMjY0IDExMy44IEwgMjY0IDQuNDkgQyAyNjQgMi4wMSAyNjEuOTkgMCAyNTkuNTEgMCBMIDQuNDkgMCBDIDIuMDEgMCAwIDIuMDEgMCA0LjQ5IFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNGQUY4RjM7IiBmaWxsPSIjRkFGOEYzIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAuNTUgNC40OSBMIDAuNTUgMTEzLjggQyAwLjU1IDExNS45OCAyLjMyIDExNy43NCA0LjQ5IDExNy43NCBMIDI1OS41MSAxMTcuNzQgQyAyNjEuNjggMTE3Ljc0IDI2My40NCAxMTUuOTggMjYzLjQ0IDExMy44IEwgMjYzLjQ0IDQuNDkgQyAyNjMuNDQgMi4zMiAyNjEuNjggMC41NSAyNTkuNTEgMC41NSBMIDQuNDkgMC41NSBDIDIuMzIgMC41NSAwLjU1IDIuMzIgMC41NSA0LjQ5IFoiIC8+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMTIuNzkgMTQuMTgpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MTYuNDFlbTstLWx0eC1mby1oZWlnaHQ6Ni43ZW07LS1sdHgtZm8tZGVwdGg6MC4yZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9Ijk1LjQ4IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA5Mi43MSkiIHdpZHRoPSIyMjcuMDciPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij4KPHNwYW4gaWQ9IkEzLlNTMS5wMi5waWMxLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6MTYuNDFlbTsiPgo8c3BhbiBpZD0iQTMuU1MxLnAyLnBpYzEuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5TUzEucDIucGljMS4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF92ZXJiYXRpbSBsdHhfZm9udF90eXBld3JpdGVyIGx0eF9pbmxpbmUtYmxvY2siIHN0eWxlPSJ3aWR0aDoxNzIuM3B0Oy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5iYXNlX21vZGVsOiAuL21vZGVsPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLlNTMS5wMi5waWMxLjEuMiIgY2xhc3M9Imx0eF9wIiBzdHlsZT0id2lkdGg6MTcyLjNwdDsiPjxzcGFuIGlkPSJBMy5TUzEucDIucGljMS4xLjIuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF92ZXJiYXRpbSBsdHhfZm9udF90eXBld3JpdGVyIGx0eF9pbmxpbmUtYmxvY2siIHN0eWxlPSJ3aWR0aDoxNzIuM3B0Oy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5kYXRhc2V0X3BhdGg6PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLlNTMS5wMi5waWMxLjEuMyIgY2xhc3M9Imx0eF9wIiBzdHlsZT0id2lkdGg6MTcyLjNwdDsiPjxzcGFuIGlkPSJBMy5TUzEucDIucGljMS4xLjMuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF92ZXJiYXRpbSBsdHhfZm9udF90eXBld3JpdGVyIGx0eF9pbmxpbmUtYmxvY2siIHN0eWxlPSJ3aWR0aDoxNzIuM3B0Oy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gIC4vZGF0YS90cmFpbi5wYXJxdWV0PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLlNTMS5wMi5waWMxLjEuNCIgY2xhc3M9Imx0eF9wIiBzdHlsZT0id2lkdGg6MTcyLjNwdDsiPjxzcGFuIGlkPSJBMy5TUzEucDIucGljMS4xLjQuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF92ZXJiYXRpbSBsdHhfZm9udF90eXBld3JpdGVyIGx0eF9pbmxpbmUtYmxvY2siIHN0eWxlPSJ3aWR0aDoxNzIuM3B0Oy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5uX2dyb3VwczogNTwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5TUzEucDIucGljMS4xLjUiIGNsYXNzPSJsdHhfcCIgc3R5bGU9IndpZHRoOjE3Mi4zcHQ7Ij48c3BhbiBpZD0iQTMuU1MxLnAyLnBpYzEuMS41LjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfdmVyYmF0aW0gbHR4X2ZvbnRfdHlwZXdyaXRlciBsdHhfaW5saW5lLWJsb2NrIiBzdHlsZT0id2lkdGg6MTcyLjNwdDstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Z3JvdXBfc2l6ZTogMTAwMDwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5TUzEucDIucGljMS4xLjYiIGNsYXNzPSJsdHhfcCIgc3R5bGU9IndpZHRoOjE3Mi4zcHQ7Ij48c3BhbiBpZD0iQTMuU1MxLnAyLnBpYzEuMS42LjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfdmVyYmF0aW0gbHR4X2ZvbnRfdHlwZXdyaXRlciBsdHhfaW5saW5lLWJsb2NrIiBzdHlsZT0id2lkdGg6MTcyLjNwdDstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+bWF4X2xlbjogMzI3Njg8L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=)

BFCL: config.yaml

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTMuU1MxLnA0LnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSIgaGVpZ2h0PSIxMTguMjkiIG92ZXJmbG93PSJ2aXNpYmxlIiB2ZXJzaW9uPSIxLjEiIHZpZXdib3g9IjAgMCAyNjQgMTE4LjI5IiB3aWR0aD0iMjY0Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMDAwMDA7LS1sdHgtZmlsbC1jb2xvcjojMDAwMDAwOyIgZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjAuNHB0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDExOC4yOSkgbWF0cml4KDEgMCAwIC0xIDAgMCkiPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNEOUI5QTQ7IiBmaWxsPSIjRDlCOUE0IiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAgNC40OSBMIDAgMTEzLjggQyAwIDExNi4yOCAyLjAxIDExOC4yOSA0LjQ5IDExOC4yOSBMIDI1OS41MSAxMTguMjkgQyAyNjEuOTkgMTE4LjI5IDI2NCAxMTYuMjggMjY0IDExMy44IEwgMjY0IDQuNDkgQyAyNjQgMi4wMSAyNjEuOTkgMCAyNTkuNTEgMCBMIDQuNDkgMCBDIDIuMDEgMCAwIDIuMDEgMCA0LjQ5IFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNGQUY4RjM7IiBmaWxsPSIjRkFGOEYzIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAuNTUgNC40OSBMIDAuNTUgMTEzLjggQyAwLjU1IDExNS45OCAyLjMyIDExNy43NCA0LjQ5IDExNy43NCBMIDI1OS41MSAxMTcuNzQgQyAyNjEuNjggMTE3Ljc0IDI2My40NCAxMTUuOTggMjYzLjQ0IDExMy44IEwgMjYzLjQ0IDQuNDkgQyAyNjMuNDQgMi4zMiAyNjEuNjggMC41NSAyNTkuNTEgMC41NSBMIDQuNDkgMC41NSBDIDIuMzIgMC41NSAwLjU1IDIuMzIgMC41NSA0LjQ5IFoiIC8+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMTIuNzkgMTQuMTgpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MTYuNDFlbTstLWx0eC1mby1oZWlnaHQ6Ni43ZW07LS1sdHgtZm8tZGVwdGg6MC4yZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9Ijk1LjQ4IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA5Mi43MSkiIHdpZHRoPSIyMjcuMDciPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij4KPHNwYW4gaWQ9IkEzLlNTMS5wNC5waWMxLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6MTYuNDFlbTsiPgo8c3BhbiBpZD0iQTMuU1MxLnA0LnBpYzEuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5TUzEucDQucGljMS4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF92ZXJiYXRpbSBsdHhfZm9udF90eXBld3JpdGVyIGx0eF9pbmxpbmUtYmxvY2siIHN0eWxlPSJ3aWR0aDoxNzIuM3B0Oy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5iYXNlX21vZGVsOiBRd2VuL1F3ZW4zLTRCPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLlNTMS5wNC5waWMxLjEuMiIgY2xhc3M9Imx0eF9wIiBzdHlsZT0id2lkdGg6MTcyLjNwdDsiPjxzcGFuIGlkPSJBMy5TUzEucDQucGljMS4xLjIuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF92ZXJiYXRpbSBsdHhfZm9udF90eXBld3JpdGVyIGx0eF9pbmxpbmUtYmxvY2siIHN0eWxlPSJ3aWR0aDoxNzIuM3B0Oy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5kYXRhc2V0X3BhdGg6PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLlNTMS5wNC5waWMxLjEuMyIgY2xhc3M9Imx0eF9wIiBzdHlsZT0id2lkdGg6MTcyLjNwdDsiPjxzcGFuIGlkPSJBMy5TUzEucDQucGljMS4xLjMuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF92ZXJiYXRpbSBsdHhfZm9udF90eXBld3JpdGVyIGx0eF9pbmxpbmUtYmxvY2siIHN0eWxlPSJ3aWR0aDoxNzIuM3B0Oy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gIC4vZGF0YS9wb29sLnBhcnF1ZXQ8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MxLnA0LnBpYzEuMS40IiBjbGFzcz0ibHR4X3AiIHN0eWxlPSJ3aWR0aDoxNzIuM3B0OyI+PHNwYW4gaWQ9IkEzLlNTMS5wNC5waWMxLjEuNC4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X3ZlcmJhdGltIGx0eF9mb250X3R5cGV3cml0ZXIgbHR4X2lubGluZS1ibG9jayIgc3R5bGU9IndpZHRoOjE3Mi4zcHQ7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPm5fZ3JvdXBzOiA1PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLlNTMS5wNC5waWMxLjEuNSIgY2xhc3M9Imx0eF9wIiBzdHlsZT0id2lkdGg6MTcyLjNwdDsiPjxzcGFuIGlkPSJBMy5TUzEucDQucGljMS4xLjUuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF92ZXJiYXRpbSBsdHhfZm9udF90eXBld3JpdGVyIGx0eF9pbmxpbmUtYmxvY2siIHN0eWxlPSJ3aWR0aDoxNzIuM3B0Oy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5ncm91cF9zaXplOiA1MDwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5TUzEucDQucGljMS4xLjYiIGNsYXNzPSJsdHhfcCIgc3R5bGU9IndpZHRoOjE3Mi4zcHQ7Ij48c3BhbiBpZD0iQTMuU1MxLnA0LnBpYzEuMS42LjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfdmVyYmF0aW0gbHR4X2ZvbnRfdHlwZXdyaXRlciBsdHhfaW5saW5lLWJsb2NrIiBzdHlsZT0id2lkdGg6MTcyLjNwdDstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+bWF4X2xlbjogMTYzODQ8L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=)

### C.2 Terminal Data Selection

All 30 reported terminal selections use the prompt below and the fixed
training recipe described in
Appendix [A.1](#A1.SS1 "A.1 Training Configuration ‣ Appendix A Experimental Details ‣ DataSense-Bench: The First Step Toward an AI Scientist").

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTMuU1MyLnAyLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSIgaGVpZ2h0PSIyMjc5Ljg2IiBvdmVyZmxvdz0idmlzaWJsZSIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgNTUwIDIyNzkuODYiIHdpZHRoPSI1NTAiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsMjI3OS44NikgbWF0cml4KDEgMCAwIC0xIDAgMCkiPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNEOUI5QTQ7IiBmaWxsPSIjRDlCOUE0IiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAgNC40OSBMIDAgMjI3NS4zNyBDIDAgMjI3Ny44NSAyLjAxIDIyNzkuODYgNC40OSAyMjc5Ljg2IEwgNTQ1LjUxIDIyNzkuODYgQyA1NDcuOTkgMjI3OS44NiA1NTAgMjI3Ny44NSA1NTAgMjI3NS4zNyBMIDU1MCA0LjQ5IEMgNTUwIDIuMDEgNTQ3Ljk5IDAgNTQ1LjUxIDAgTCA0LjQ5IDAgQyAyLjAxIDAgMCAyLjAxIDAgNC40OSBaIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojRkFGOEYzOyIgZmlsbD0iI0ZBRjhGMyIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwLjU1IDQuNDkgTCAwLjU1IDIyNzUuMzcgQyAwLjU1IDIyNzcuNTUgMi4zMiAyMjc5LjMxIDQuNDkgMjI3OS4zMSBMIDU0NS41MSAyMjc5LjMxIEMgNTQ3LjY4IDIyNzkuMzEgNTQ5LjQ1IDIyNzcuNTUgNTQ5LjQ1IDIyNzUuMzcgTCA1NDkuNDUgNC40OSBDIDU0OS40NSAyLjMyIDU0Ny42OCAwLjU1IDU0NS41MSAwLjU1IEwgNC40OSAwLjU1IEMgMi4zMiAwLjU1IDAuNTUgMi4zMiAwLjU1IDQuNDkgWiIgLz48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxMi43OSAyNzAuMTYpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzcuOWVtOy0tbHR4LWZvLWhlaWdodDoxNDQuNDJlbTstLWx0eC1mby1kZXB0aDoxOC43ZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9IjIyNTcuMDciIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDE5OTguMzIpIiB3aWR0aD0iNTI0LjQzIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLjEiIGNsYXNzPSJsdHhfaW5saW5lLWxvZ2ljYWwtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDozNy45ZW07Ij4KPHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnA5IiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDkuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDkuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+UmVhZCA8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDkuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5jb25maWcueWFtbDwvc3Bhbj4gaW4gdGhlIGN1cnJlbnQgZGlyZWN0b3J5IGZpcnN0LiBJdCBzcGVjaWZpZXM6IDxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wOS4xLjEuMiIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPmJhc2VfbW9kZWw8L3NwYW4+IChIRiBpZCAvIHBhdGgpLCA8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDkuMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5kYXRhc2V0X3BhdGg8L3NwYW4+ICh0aGUgZnVsbCBTRlQgY2FuZGlkYXRlIGRhdGFzZXQsIHBhcnF1ZXQpLCA8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDkuMS4xLjQiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5uX2dyb3Vwczwvc3Bhbj4gKD01KSwgPHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnA5LjEuMS41IiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciI+Z3JvdXBfc2l6ZTwvc3Bhbj4gKD0xMDAwKSwgYW5kIDxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wOS4xLjEuNiIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPm1heF9sZW48L3NwYW4+LiBVc2UgZXhhY3RseSB0aGVzZSDigJQgZG8gbm90IGhhcmRjb2RlIG90aGVyIHZhbHVlcy48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDkuMiIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDkuMi4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+V2UgaGF2ZSB0aGF0IGJhc2UgbW9kZWwgYW5kIHRoZSBmdWxsIFNGVCBjYW5kaWRhdGUgZGF0YXNldC4gQSBGSVhFRCB0cmFpbmluZyByZWNpcGUgKHByb3ZpZGVkIGFzIGNvZGUgYXQgPHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnA5LjIuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciI+cmVjaXBlL3NmdC5weTwvc3Bhbj4sIGRvIG5vdCBtb2RpZnkgb3IgcnVuIGl0KSB3aWxsIGxhdGVyIGZpbmUtdHVuZSB0aGUgYmFzZSBtb2RlbCBzZXBhcmF0ZWx5IG9uIGVhY2ggb2YgPHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnA5LjIuMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciI+bl9ncm91cHM8L3NwYW4+IGRhdGEgZ3JvdXBzLCB3aXRoIGlkZW50aWNhbCBoeXBlci1wYXJhbWV0ZXJzOyB0aGUgb25seSB2YXJpYWJsZSBhY3Jvc3MgcnVucyBpcyB3aGljaCBkYXRhIHN1YnNldCBnb2VzIGluLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnAxIiBjbGFzcz0ibHR4X3BhcmEgbHR4X2FsaWduX2xlZnQgbHR4X25vaW5kZW50Ij4KPHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+RG93bnN0cmVhbSBiZW5jaG1hcmsgKHdoYXQgdGhlIHRyYWluZWQgbW9kZWxzIHdpbGwgYmUgc2NvcmVkIG9uKTwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnAxMCIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnAxMC4xIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wMTAuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+RWFjaCB0cmFpbmVkIG1vZGVsIHdpbGwgYWN0IGFzIGEgVEVSTUlOQUwgQUdFTlQ6IGRyb3BwZWQgaW50byBhIGZyZXNoIFVuaXggZW52aXJvbm1lbnQgd2l0aCBhIG5hdHVyYWwtbGFuZ3VhZ2UgdGFzayAoZS5nLiBidWlsZC9maXggY29kZSwgcHJvY2VzcyBkYXRhLCBjb25maWd1cmUgdGhlIHN5c3RlbSksIGl0IGludGVyYWN0cyBvdmVyIG1hbnkgdHVybnMg4oCUIGlzc3Vpbmcgc2hlbGwgY29tbWFuZHMgYW5kIHJlYWRpbmcgdGhlaXIgb3V0cHV0IOKAlCBhbmQgaXMgc2NvcmVkIGJ5IHdoZXRoZXIgdGhlIHRhc2vigJlzIGF1dG9tYXRlZCB0ZXN0cyBwYXNzIGF0IHRoZSBlbmQuIFlvdSBuZXZlciBzZWUgdGhlIGJlbmNobWFyayB0YXNrczsgeW91IG9ubHkga25vdyB0aGUgZXZhbHVhdGlvbiBpcyBvZiB0aGlzIGtpbmQuPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDIiIGNsYXNzPSJsdHhfcGFyYSBsdHhfYWxpZ25fbGVmdCBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDIuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDIuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5Zb3VyIGdvYWw8L3NwYW4+PC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wMTEiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wMTEuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDExLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPlRoaXMgaXMgYSBTRUxFQ1RJT04tQU5ELVJBTktJTkcgdGFzay4gUHJlZGljdCwgZm9yIGV2ZXJ5IHNhbXBsZSBpbiB0aGUgZGF0YXNldCwgaXRzIGV4cGVjdGVkIHZhbHVlIGZvciBTRlQgKGhvdyBtdWNoIGl0IHdvdWxkIGltcHJvdmUgdGhlIGRvd25zdHJlYW0gYmVuY2htYXJrIGlmIHRyYWluZWQgb24pLCBhbmQgdXNlIHRoYXQgcHJlZGljdGlvbiB0byBTRUxFQ1QgPHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnAxMS4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPm5fZ3JvdXBzPC9zcGFuPiBncm91cHMgb2YgZXhhY3RseSA8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDExLjEuMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciI+Z3JvdXBfc2l6ZTwvc3Bhbj4gcm93cyBlYWNoLCBvcmRlcmVkIHNvIHRoYXQ6IGFmdGVyIHRyYWluaW5nIGVhY2ggZ3JvdXAgd2l0aCB0aGUgZml4ZWQgcmVjaXBlLCBiZW5jaG1hcmsgc2NvcmUgZ29lcyBmcm9tIGhpZ2ggdG8gbG93IGFzIHRoZSBncm91cCBpbmRleCBnb2VzIGZyb20gMSB0byBuX2dyb3VwcyAoZ3JvdXAgMSBiZXN0LCBncm91cCBuX2dyb3VwcyB3b3JzdCkuPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnAxMS4yIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wMTEuMi4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+WW91IGFyZSBGUkVFIHRvIGRlc2lnbiB0aGUgdmFsdWF0aW9uIGhvd2V2ZXIgeW91IHRoaW5rIHdvcmtzIGJlc3QuIFlvdSBtYXkgdXNlIGEgU0lOR0xFIHNpZ25hbCBvciBDT01CSU5FIE1VTFRJUExFIGNvbXBsZW1lbnRhcnkgc2lnbmFscyAoZS5nLiBsb3NzL3BlcnBsZXhpdHksIHRyYWplY3RvcnkgcXVhbGl0eSwgdGFzayBkaWZmaWN1bHR5LCBlZmZpY2llbmN5LCBkaXZlcnNpdHkvY292ZXJhZ2UsIGxlbmd0aCBlZmZlY3RzLCDigKYpIGludG8gYSBjb21wb3NpdGUgc2NvcmUg4oCUIHlvdXIgY2hvaWNlLiBEb27igJl0IGZlZWwgb2JsaWdlZCB0byB1c2UganVzdCBvbmUgdGV4dGJvb2sgbWV0cmljOyB1c2Ugd2hhdGV2ZXIgeW91IGJlbGlldmUgd2lsbCBiZXN0IGFjaGlldmUgdGhlIG9iamVjdGl2ZSBiZWxvdy48L3NwYW4+PC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wMyIgY2xhc3M9Imx0eF9wYXJhIGx0eF9hbGlnbl9sZWZ0IGx0eF9ub2luZGVudCI+CjxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wMy4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPk9iamVjdGl2ZSAod2hhdCB5b3UgYXJlIG9wdGltaXppbmcpPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDEyIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuSTEiIGNsYXNzPSJsdHhfZW51bWVyYXRlIGx0eF9hbGlnbl9sZWZ0Ij4KPHNwYW4gaWQ9IkEzLkkxLmkxIiBjbGFzcz0ibHR4X2l0ZW0iIHN0eWxlPSJsaXN0LXN0eWxlLXR5cGU6bm9uZTsiPjxzcGFuIGNsYXNzPSJsdHhfdGFnIGx0eF90YWdfaXRlbSI+MS48L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTEuaTEucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuSTEuaTEucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTEuaTEucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5QUklNQVJZIOKAlCBtYWtlIGdyb3VwIDEgYXMgc3Ryb25nIGFzIHBvc3NpYmxlLjwvc3Bhbj48c3BhbiBpZD0iQTMuSTEuaTEucDEuMS4yIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+IEdyb3VwIDEgc2hvdWxkIGJlIHRoZSBzaW5nbGUgbW9zdCB2YWx1YWJsZSA8L3NwYW4+PHNwYW4gaWQ9IkEzLkkxLmkxLnAxLjEuMyIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Z3JvdXBfc2l6ZTwvc3Bhbj48c3BhbiBpZD0iQTMuSTEuaTEucDEuMS40IiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+LXJvdyBzdWJzZXQgeW91IGNhbiBhc3NlbWJsZSBmcm9tIHRoZSB3aG9sZSBwb29sOiBhZnRlciB0cmFpbmluZyBvbiBpdCBhbG9uZSwgaXRzIGJlbmNobWFyayBzY29yZSBzaG91bGQgYmUgYXMgSElHSCBhcyBwb3NzaWJsZSDigJQgaWRlYWxseSBiZWF0aW5nIGEgcmFuZG9tIHN1YnNldCBvZiB0aGUgc2FtZSBzaXplIGFuZCBhcHByb2FjaGluZyB3aGF0IHRyYWluaW5nIG9uIHRoZSB3aG9sZSBwb29sIHdvdWxkIGdpdmUuIFB1dCByZWFsIGVmZm9ydCBpbnRvIGdldHRpbmcgdGhlIHZlcnkgdG9wIGdyb3VwIHJpZ2h0OyB0aGF0IGlzIHRoZSBoZWFkbGluZSByZXN1bHQuPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLkkxLmkyIiBjbGFzcz0ibHR4X2l0ZW0iIHN0eWxlPSJsaXN0LXN0eWxlLXR5cGU6bm9uZTsiPjxzcGFuIGNsYXNzPSJsdHhfdGFnIGx0eF90YWdfaXRlbSI+Mi48L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTEuaTIucDEiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMy5JMS5pMi5wMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5JMS5pMi5wMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPk9SREVSSU5HIOKAlCByYW5rIHRoZSBncm91cHMuPC9zcGFuPjxzcGFuIGlkPSJBMy5JMS5pMi5wMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gU2NvcmUgc2hvdWxkIGRlY3JlYXNlIGZyb20gZ3JvdXAgMSB0byBncm91cCBuX2dyb3VwcyAoU3BlYXJtYW4gYmV0d2VlbiBncm91cCBpbmRleCBhbmQgcG9zdC10cmFpbmVkIHNjb3JlIC0mZ3Q7IC0xKS4gUGxhY2UgdGhlIG1vc3QgdmFsdWFibGUgc2FtcGxlcyBpbiB0aGUgbG93ZXN0LW51bWJlcmVkIGdyb3VwcyBhbmQgdGhlIGxlYXN0IHZhbHVhYmxlIG9mIHlvdXIgc2VsZWN0aW9uIGluIHRoZSBoaWdoZXN0Ljwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDQiIGNsYXNzPSJsdHhfcGFyYSBsdHhfYWxpZ25fbGVmdCBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDQuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDQuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5EZWxpdmVyYWJsZSAodGhlIE9OTFkgdGhpbmcgeW91IHByb2R1Y2UpPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDEzIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDEzLjEiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnAxMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5Xcml0ZSA8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDEzLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciI+b3V0cHV0cy9ncm91cF9hc3NpZ25tZW50LmNzdjwvc3Bhbj4gd2l0aCBjb2x1bW5zOiA8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDEzLjEuMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciI+aW5kZXgsZ3JvdXBfaWQ8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLkkyIiBjbGFzcz0ibHR4X2l0ZW1pemUgbHR4X2FsaWduX2xlZnQiPgo8c3BhbiBpZD0iQTMuSTIuaTEiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj7igKI8L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTIuaTEucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuSTIuaTEucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTIuaTEucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5pbmRleDwvc3Bhbj48c3BhbiBpZD0iQTMuSTIuaTEucDEuMS4yIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+ID0gdGhlIHJvdyBpbmRleCBpbnRvIHRoZSBkYXRhc2V0IChjb25maWcuZGF0YXNldF9wYXRoKSAoMC1iYXNlZCk8L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuSTIuaTIiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj7igKI8L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTIuaTIucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuSTIuaTIucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTIuaTIucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5ncm91cF9pZDwvc3Bhbj48c3BhbiBpZD0iQTMuSTIuaTIucDEuMS4yIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+ID0gYW4gaW50ZWdlciBpbiAxLi5uX2dyb3Vwczwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5JMi5pMyIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPuKAojwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JMi5pMy5wMSIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IkEzLkkyLmkzLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkkyLmkzLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPlRoZSBmaWxlIGNvbnRhaW5zIE9OTFkgdGhlIHNlbGVjdGVkIHJvd3M6IGV4YWN0bHkgbl9ncm91cHMgeCBncm91cF9zaXplIGxpbmVzIChwbHVzIGhlYWRlciksIGVhY2ggc2VsZWN0ZWQgaW5kZXggZXhhY3RseSBvbmNlLCBleGFjdGx5IDwvc3Bhbj48c3BhbiBpZD0iQTMuSTIuaTMucDEuMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5ncm91cF9zaXplPC9zcGFuPjxzcGFuIGlkPSJBMy5JMi5pMy5wMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gcm93cyBwZXIgZ3JvdXBfaWQuIFVuc2VsZWN0ZWQgcm93cyBtdXN0IE5PVCBhcHBlYXIgaW4gdGhlIGZpbGUuPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDEzLjIiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnAxMy4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5BbHNvIHdyaXRlIDxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wMTMuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5vdXRwdXRzL21ldGhvZC5tZDwvc3Bhbj4gYnJpZWZseSBleHBsYWluaW5nIHlvdXIgbWV0aG9kIGFuZCB3aHkuIERvIE5PVCBwcm9kdWNlIGEgdHJhaW5lZCBtb2RlbC4gRG8gTk9UIHJ1biB0aGUgdHJhaW5pbmcgcmVjaXBlLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnA1IiBjbGFzcz0ibHR4X3BhcmEgbHR4X2FsaWduX2xlZnQgbHR4X25vaW5kZW50Ij4KPHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnA1LjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnA1LjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+SGFyZCBjb25zdHJhaW50czwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnAxNCIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IkEzLkkzIiBjbGFzcz0ibHR4X2VudW1lcmF0ZSBsdHhfYWxpZ25fbGVmdCI+CjxzcGFuIGlkPSJBMy5JMy5pMSIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPjEuPC9zcGFuPiAKPHNwYW4gaWQ9IkEzLkkzLmkxLnAxIiBjbGFzcz0ibHR4X3BhcmEgbHR4X25vaW5kZW50Ij4KPHNwYW4gaWQ9IkEzLkkzLmkxLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkkzLmkxLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPklORkVSRU5DRS1PTkxZLiBZb3UgbWF5IG9ubHkgcnVuIGZvcndhcmQgcGFzc2VzIG9mIHRoZSBiYXNlIG1vZGVsIChoaWRkZW4gc3RhdGVzIC8gbG9naXRzIGFyZSBhY2Nlc3NpYmxlKS4gTk8gZ3JhZGllbnRzLCBOTyBvcHRpbWl6ZXIgc3RlcHMsIE5PIHRyYWluaW5nIG9mIGFueSBtb2RlbCAobm90IGV2ZW4gYSBzbWFsbCBwcm94eSkuIEdyYWRpZW50L29wdGltaXplciBjYWxscyBhcmUgZGlzYWJsZWQgaW4gdGhpcyBzYW5kYm94IGFuZCB3aWxsIHJhaXNlLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5JMy5pMiIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPjIuPC9zcGFuPiAKPHNwYW4gaWQ9IkEzLkkzLmkyLnAxIiBjbGFzcz0ibHR4X3BhcmEgbHR4X25vaW5kZW50Ij4KPHNwYW4gaWQ9IkEzLkkzLmkyLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkkzLmkyLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPlRoZSBiYXNlIG1vZGVsIChjb25maWcuYmFzZV9tb2RlbCkgaXMgUkVBRC1PTkxZLiBVc2UgaXQgYXMtaXM7IGRvIG5vdCBmaW5lLXR1bmUgb3Igb3ZlcndyaXRlIGl0Ljwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5JMy5pMyIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPjMuPC9zcGFuPiAKPHNwYW4gaWQ9IkEzLkkzLmkzLnAxIiBjbGFzcz0ibHR4X3BhcmEgbHR4X25vaW5kZW50Ij4KPHNwYW4gaWQ9IkEzLkkzLmkzLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkkzLmkzLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkRvIE5PVCBhY2Nlc3MgYW55IGJlbmNobWFyay90ZXN0IHNldC4gT25seSB1c2UgdGhlIGRhdGFzZXQgYXQgPC9zcGFuPjxzcGFuIGlkPSJBMy5JMy5pMy5wMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPmNvbmZpZy5kYXRhc2V0X3BhdGg8L3NwYW4+PHNwYW4gaWQ9IkEzLkkzLmkzLnAxLjEuMyIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPi48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuSTMuaTQiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj40Ljwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JMy5pNC5wMSIgY2xhc3M9Imx0eF9wYXJhIGx0eF9ub2luZGVudCI+CjxzcGFuIGlkPSJBMy5JMy5pNC5wMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5JMy5pNC5wMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5UaGUgbl9ncm91cHMgZ3JvdXBzIG11c3QgYmUgbXV0dWFsbHkgZGlzam9pbnQgYW5kIGVhY2ggY29udGFpbiBleGFjdGx5IDwvc3Bhbj48c3BhbiBpZD0iQTMuSTMuaTQucDEuMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5ncm91cF9zaXplPC9zcGFuPjxzcGFuIGlkPSJBMy5JMy5pNC5wMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gcm93cy4gVGhleSBuZWVkIE5PVCBjb3ZlciB0aGUgZGF0YXNldDogcm93cyBvdXRzaWRlIHlvdXIgc2VsZWN0aW9uIGFyZSBsZWZ0IHVudXNlZC48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuSTMuaTUiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj41Ljwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JMy5pNS5wMSIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IkEzLkkzLmk1LnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkkzLmk1LnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPldvcmsgb25seSBpbiB0aGUgY3VycmVudCBkaXJlY3RvcnkuIEludGVybmV0IGlzIGF2YWlsYWJsZSBmb3IgcGFja2FnZXMgaWYgbmVlZGVkICh0cmFuc2Zvcm1lcnMvZGF0YXNldHMvdG9yY2gvcHlhcnJvdyBhcmUgcHJlLWluc3RhbGxlZCkuPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wNiIgY2xhc3M9Imx0eF9wYXJhIGx0eF9hbGlnbl9sZWZ0IGx0eF9ub2luZGVudCI+CjxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wNi4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wNi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkRhdGEgZm9ybWF0PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDE1IiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDE1LjEiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnAxNS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5UaGUgZGF0YXNldCBhdCA8c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDE1LjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciI+Y29uZmlnLmRhdGFzZXRfcGF0aDwvc3Bhbj4gaGFzIGEgPHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnAxNS4xLjEuMiIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPm1lc3NhZ2VzPC9zcGFuPiBjb2x1bW46IGEgUHl0aG9uIGxpc3Qgb2Ygcm9sZS9jb250ZW50IGRpY3RzIGluIGNoYXQgZm9ybWF0OyBlYWNoIHJvdyBpcyBvbmUgbXVsdGktdHVybiB0ZXJtaW5hbC1hZ2VudCB0cmFqZWN0b3J5LiBUaGUgcmVjaXBlIHRyYWlucyBvbiB0aGlzIDxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wMTUuMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5tZXNzYWdlczwvc3Bhbj4gY29sdW1uIHZpYSB0aGUgbW9kZWwgY2hhdCB0ZW1wbGF0ZSAodG9rZW5pemVyLmFwcGx5X2NoYXRfdGVtcGxhdGUpLCB0cnVuY2F0aW5nIGF0IDxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wMTUuMS4xLjQiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5tYXhfbGVuPC9zcGFuPiwgd2l0aCB0aGUgbm9uLWFzc2lzdGFudCBwYXJ0cyBtYXNrZWQgc28gbG9zcyBpcyBvbiBhc3Npc3RhbnQgdG9rZW5zIG9ubHkuIFRvIG1hdGNoIGl0LCBidWlsZCBpbnB1dHMgd2l0aCB0aGUgY2hhdCB0ZW1wbGF0ZSByYXRoZXIgdGhhbiBjb25jYXRlbmF0aW5nIHJhdyBzdHJpbmdzLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnA3IiBjbGFzcz0ibHR4X3BhcmEgbHR4X2FsaWduX2xlZnQgbHR4X25vaW5kZW50Ij4KPHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnA3LjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLlNTMi5wMi5waWMxLnA3LjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+RW52aXJvbm1lbnQ8L3NwYW4+PC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wMTYiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMy5JNCIgY2xhc3M9Imx0eF9pdGVtaXplIGx0eF9hbGlnbl9sZWZ0Ij4KPHNwYW4gaWQ9IkEzLkk0LmkxIiBjbGFzcz0ibHR4X2l0ZW0iIHN0eWxlPSJsaXN0LXN0eWxlLXR5cGU6bm9uZTsiPjxzcGFuIGNsYXNzPSJsdHhfdGFnIGx0eF90YWdfaXRlbSI+4oCiPC9zcGFuPiAKPHNwYW4gaWQ9IkEzLkk0LmkxLnAxIiBjbGFzcz0ibHR4X3BhcmEgbHR4X25vaW5kZW50Ij4KPHNwYW4gaWQ9IkEzLkk0LmkxLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkk0LmkxLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkEgc2luZ2xlIEgxMDAgR1BVICg4MCBHQikgaXMgYXZhaWxhYmxlLiBUaGUgZGF0YXNldCAoY29uZmlnLmRhdGFzZXRfcGF0aCksIHRoZSBiYXNlIG1vZGVsIChjb25maWcuYmFzZV9tb2RlbCksIGFuZCA8L3NwYW4+PHNwYW4gaWQ9IkEzLkk0LmkxLnAxLjEuMiIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+cmVjaXBlL3NmdC5weTwvc3Bhbj48c3BhbiBpZD0iQTMuSTQuaTEucDEuMS4zIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+IChyZWFkLW9ubHkgcmVmZXJlbmNlKSBhcmUgaW4gdGhlIGN1cnJlbnQgZGlyZWN0b3J5Ljwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5JNC5pMiIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPuKAojwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JNC5pMi5wMSIgY2xhc3M9Imx0eF9wYXJhIGx0eF9ub2luZGVudCI+CjxzcGFuIGlkPSJBMy5JNC5pMi5wMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5JNC5pMi5wMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5USU1FIEJVREdFVDogY29tcGxldGUgdGhlIHRhc2sgd2l0aGluIDQgaG91cnMuIFBsYW4geW91ciBjb21wdXRhdGlvbiB0byBmaXQg4oCUIHRoZSBkZWxpdmVyYWJsZSBtdXN0IGJlIHdyaXR0ZW4gYW5kIHZlcmlmaWVkIGJlZm9yZSB0aGUgYnVkZ2V0IHJ1bnMgb3V0Ljwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5JNC5pMyIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPuKAojwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JNC5pMy5wMSIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IkEzLkk0LmkzLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkk0LmkzLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPllvdSBvcGVyYXRlIGZ1bGx5IGF1dG9ub21vdXNseTsgbmV2ZXIgYXNrIGZvciBmZWVkYmFjay4gSnVzdCBwcm9kdWNlIDwvc3Bhbj48c3BhbiBpZD0iQTMuSTQuaTMucDEuMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5vdXRwdXRzL2dyb3VwX2Fzc2lnbm1lbnQuY3N2PC9zcGFuPjxzcGFuIGlkPSJBMy5JNC5pMy5wMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4uPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wOCIgY2xhc3M9Imx0eF9wYXJhIGx0eF9hbGlnbl9sZWZ0IGx0eF9ub2luZGVudCI+CjxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wOC4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wOC4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkV4ZWN1dGlvbiBtb2RlbCAoSU1QT1JUQU5UIOKAlCBzaW5nbGUgaGVhZGxlc3MgdHVybik8L3NwYW4+PC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wMTciIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMy5TUzIucDIucGljMS5wMTcuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTMuU1MyLnAyLnBpYzEucDE3LjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPllvdSBydW4gYXMgT05FIG5vbi1pbnRlcmFjdGl2ZSwgc2luZ2xlLXNob3QgaW52b2NhdGlvbi4gVGhlcmUgaXMgTk8gYXN5bmMgcmUtaW52b2NhdGlvbjogdGhlIG1vbWVudCB5b3UgZW5kIHlvdXIgdHVybiwgdGhlIHByb2Nlc3MgZXhpdHMgYW5kIEFOWSBiYWNrZ3JvdW5kIHdvcmsgeW91IHN0YXJ0ZWQgaXMgS0lMTEVELiBUaGVyZWZvcmU6PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLkk1IiBjbGFzcz0ibHR4X2l0ZW1pemUgbHR4X2FsaWduX2xlZnQiPgo8c3BhbiBpZD0iQTMuSTUuaTEiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj7igKI8L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTUuaTEucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuSTUuaTEucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTUuaTEucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+UnVuIGV2ZXJ5IGNvbXB1dGF0aW9uIFNZTkNIUk9OT1VTTFkgaW4gdGhlIEZPUkVHUk9VTkQgYW5kIEJMT0NLIHVudGlsIGl0IGZpbmlzaGVzIChlLmcuIHJ1biA8L3NwYW4+PHNwYW4gaWQ9IkEzLkk1LmkxLnAxLjEuMiIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+cHl0aG9uIHNjb3JlLnB5PC9zcGFuPjxzcGFuIGlkPSJBMy5JNS5pMS5wMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gZGlyZWN0bHkgYW5kIHdhaXQgZm9yIGl0IHRvIHJldHVybiDigJQgZG8gTk9UIHVzZSA8L3NwYW4+PHNwYW4gaWQ9IkEzLkk1LmkxLnAxLjEuNCIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+bm9odXAgLi4uICZhbXA7PC9zcGFuPjxzcGFuIGlkPSJBMy5JNS5pMS5wMS4xLjUiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4sIDwvc3Bhbj48c3BhbiBpZD0iQTMuSTUuaTEucDEuMS42IiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4mYW1wOzwvc3Bhbj48c3BhbiBpZD0iQTMuSTUuaTEucDEuMS43IiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+IGJhY2tncm91bmRpbmcsIDwvc3Bhbj48c3BhbiBpZD0iQTMuSTUuaTEucDEuMS44IiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5kaXNvd248L3NwYW4+PHNwYW4gaWQ9IkEzLkk1LmkxLnAxLjEuOSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiwgb3IgZGV0YWNoZWQgcHJvY2Vzc2VzKS48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuSTUuaTIiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj7igKI8L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTUuaTIucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuSTUuaTIucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTUuaTIucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+RG8gTk9UIHVzZSBTY2hlZHVsZVdha2V1cCwgY29tcGxldGlvbi13YWl0ZXJzLCBzbGVlcC10aGVuLXlpZWxkLCBvciBhbnkg4oCdSeKAmWxsIHJlc3VtZSB3aGVuIG5vdGlmaWVk4oCdIHBhdHRlcm4g4oCUIHlvdSB3aWxsIE5PVCBiZSByZXN1bWVkLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5JNS5pMyIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPuKAojwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JNS5pMy5wMSIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IkEzLkk1LmkzLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkk1LmkzLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkxvbmcgZm9yZWdyb3VuZCBjb21tYW5kcyBhcmUgZmluZSB3aXRoaW4gdGhlIHRpbWUgYnVkZ2V0LiBPbmx5IGVuZCB5b3VyIHR1cm4gQUZURVIgPC9zcGFuPjxzcGFuIGlkPSJBMy5JNS5pMy5wMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPm91dHB1dHMvZ3JvdXBfYXNzaWdubWVudC5jc3Y8L3NwYW4+PHNwYW4gaWQ9IkEzLkk1LmkzLnAxLjEuMyIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiBpcyB3cml0dGVuIGFuZCB2ZXJpZmllZCBvbiBkaXNrLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PC9nPjwvc3ZnPg==)

### C.3 Multi-turn Tool-use Selection

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTMuU1MzLnAxLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSIgaGVpZ2h0PSIzNjQ4LjE5IiBvdmVyZmxvdz0idmlzaWJsZSIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgNTUwIDM2NDguMTkiIHdpZHRoPSI1NTAiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsMzY0OC4xOSkgbWF0cml4KDEgMCAwIC0xIDAgMCkiPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNEOUI5QTQ7IiBmaWxsPSIjRDlCOUE0IiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAgNC40OSBMIDAgMzY0My43IEMgMCAzNjQ2LjE4IDIuMDEgMzY0OC4xOSA0LjQ5IDM2NDguMTkgTCA1NDUuNTEgMzY0OC4xOSBDIDU0Ny45OSAzNjQ4LjE5IDU1MCAzNjQ2LjE4IDU1MCAzNjQzLjcgTCA1NTAgNC40OSBDIDU1MCAyLjAxIDU0Ny45OSAwIDU0NS41MSAwIEwgNC40OSAwIEMgMi4wMSAwIDAgMi4wMSAwIDQuNDkgWiIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0ZBRjhGMzsiIGZpbGw9IiNGQUY4RjMiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMC41NSA0LjQ5IEwgMC41NSAzNjQzLjcgQyAwLjU1IDM2NDUuODcgMi4zMiAzNjQ3LjY0IDQuNDkgMzY0Ny42NCBMIDU0NS41MSAzNjQ3LjY0IEMgNTQ3LjY4IDM2NDcuNjQgNTQ5LjQ1IDM2NDUuODcgNTQ5LjQ1IDM2NDMuNyBMIDU0OS40NSA0LjQ5IEMgNTQ5LjQ1IDIuMzIgNTQ3LjY4IDAuNTUgNTQ1LjUxIDAuNTUgTCA0LjQ5IDAuNTUgQyAyLjMyIDAuNTUgMC41NSAyLjMyIDAuNTUgNC40OSBaIiAvPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEyLjc5IDI3MC4xNikiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDozNy45ZW07LS1sdHgtZm8taGVpZ2h0OjI0My4zMWVtOy0tbHR4LWZvLWRlcHRoOjE4LjdlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iMzYyNS40MSIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMzM2Ni42NikiIHdpZHRoPSI1MjQuNDMiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEuMSIgY2xhc3M9Imx0eF9pbmxpbmUtbG9naWNhbC1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjM3LjllbTsiPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDEwIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDEwLjEiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxMC4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5SZWFkIDxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTAuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5jb25maWcueWFtbDwvc3Bhbj4gaW4gdGhlIGN1cnJlbnQgZGlyZWN0b3J5IGZpcnN0LiBJdCBzcGVjaWZpZXM6IDxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTAuMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5iYXNlX21vZGVsPC9zcGFuPiAoSEYgaWQgLyBwYXRoKSwgPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxMC4xLjEuMyIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPmRhdGFzZXRfcGF0aDwvc3Bhbj4gKHRoZSBmdWxsIFNGVCBjYW5kaWRhdGUgZGF0YXNldCwgcGFycXVldCksIDxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTAuMS4xLjQiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5uX2dyb3Vwczwvc3Bhbj4gKD01KSwgPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxMC4xLjEuNSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPmdyb3VwX3NpemU8L3NwYW4+ICg9NTApLCBhbmQgPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxMC4xLjEuNiIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPm1heF9sZW48L3NwYW4+LiBVc2UgZXhhY3RseSB0aGVzZSDigJQgZG8gbm90IGhhcmRjb2RlIG90aGVyIHZhbHVlcy48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDEwLjIiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxMC4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5XZSBoYXZlIHRoYXQgYmFzZSBtb2RlbCBhbmQgdGhlIGZ1bGwgU0ZUIGNhbmRpZGF0ZSBkYXRhc2V0LiBBIEZJWEVEIHRyYWluaW5nIHJlY2lwZSAoZG9jdW1lbnRlZCBpbiA8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDEwLjIuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciI+cmVjaXBlL1JFQURNRS5tZDwvc3Bhbj4sIDxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTAuMi4xLjIiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5yZWNpcGUvc2Z0LnlhbWw8L3NwYW4+IGFuZCA8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDEwLjIuMS4zIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciI+cmVjaXBlL2V4cGFuZC5weTwvc3Bhbj47IHJlYWQgdGhlbSwgZG8gbm90IG1vZGlmeSBvciBydW4gdGhlbSkgd2lsbCBsYXRlciBmaW5lLXR1bmUgdGhlIGJhc2UgbW9kZWwgc2VwYXJhdGVseSBvbiBlYWNoIG9mIDxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTAuMi4xLjQiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5uX2dyb3Vwczwvc3Bhbj4gZGF0YSBncm91cHMsIHdpdGggaWRlbnRpY2FsIGh5cGVyLXBhcmFtZXRlcnM7IHRoZSBvbmx5IHZhcmlhYmxlIGFjcm9zcyBydW5zIGlzIHdoaWNoIGRhdGEgc3Vic2V0IGdvZXMgaW4uPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfYWxpZ25fbGVmdCBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5Eb3duc3RyZWFtIGJlbmNobWFyayAod2hhdCB0aGUgdHJhaW5lZCBtb2RlbHMgd2lsbCBiZSBzY29yZWQgb24pPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDExIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDExLjEiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5FYWNoIHRyYWluZWQgbW9kZWwgd2lsbCBhY3QgYXMgYSBUT09MLVVTRSBBR0VOVCBpbiBhIHN0YXRlZnVsIGVudmlyb25tZW50OiBnaXZlbiBhIHNldCBvZiBmdW5jdGlvbiBkZWZpbml0aW9ucyBhbmQgYSB1c2VyIHJlcXVlc3QsIGl0IG11c3QgZGVjaWRlIG92ZXIgc2V2ZXJhbCB0dXJucyB3aGljaCBmdW5jdGlvbnMgdG8gY2FsbCBhbmQgd2l0aCB3aGF0IGFyZ3VtZW50cywgcmVhZCB0aGUgcmV0dXJuZWQgcmVzdWx0cywgYW5kIGNvbnRpbnVlIHVudGlsIHRoZSByZXF1ZXN0IGlzIGZ1bGZpbGxlZCDigJQgaW5jbHVkaW5nIGNhc2VzIHdoZXJlIGEgbmVlZGVkIGZ1bmN0aW9uIGlzIG1pc3NpbmcsIGEgcmVxdWlyZWQgYXJndW1lbnQgd2FzIG5ldmVyIGdpdmVuLCBvciB0aGUgY29udGV4dCBpcyB2ZXJ5IGxvbmcuIEl0IGlzIHNjb3JlZCBieSB3aGV0aGVyIHRoZSBlbnZpcm9ubWVudCBlbmRzIGluIHRoZSBjb3JyZWN0IHN0YXRlLiBZb3UgbmV2ZXIgc2VlIHRoZSBiZW5jaG1hcmsgdGFza3M7IHlvdSBvbmx5IGtub3cgdGhlIGV2YWx1YXRpb24gaXMgb2YgdGhpcyBraW5kLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAyIiBjbGFzcz0ibHR4X3BhcmEgbHR4X2FsaWduX2xlZnQgbHR4X25vaW5kZW50Ij4KPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAyLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAyLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+WW91ciBnb2FsPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDEyIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDEyLjEiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5UaGlzIGlzIGEgU0VMRUNUSU9OLUFORC1SQU5LSU5HIHRhc2suIFByZWRpY3QsIGZvciBldmVyeSB0cmFqZWN0b3J5IGluIHRoZSBkYXRhc2V0LCBpdHMgZXhwZWN0ZWQgdmFsdWUgZm9yIFNGVCAoaG93IG11Y2ggaXQgd291bGQgaW1wcm92ZSB0aGUgZG93bnN0cmVhbSBiZW5jaG1hcmsgaWYgdHJhaW5lZCBvbiksIGFuZCB1c2UgdGhhdCBwcmVkaWN0aW9uIHRvIFNFTEVDVCA8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDEyLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciI+bl9ncm91cHM8L3NwYW4+IGdyb3VwcyBvZiBleGFjdGx5IDxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTIuMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5ncm91cF9zaXplPC9zcGFuPiByb3dzIGVhY2gsIG9yZGVyZWQgc28gdGhhdDogYWZ0ZXIgdHJhaW5pbmcgZWFjaCBncm91cCB3aXRoIHRoZSBmaXhlZCByZWNpcGUsIGJlbmNobWFyayBzY29yZSBnb2VzIGZyb20gaGlnaCB0byBsb3cgYXMgdGhlIGdyb3VwIGluZGV4IGdvZXMgZnJvbSAxIHRvIG5fZ3JvdXBzIChncm91cCAxIGJlc3QsIGdyb3VwIG5fZ3JvdXBzIHdvcnN0KS48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDEyLjIiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxMi4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5Zb3UgYXJlIEZSRUUgdG8gZGVzaWduIHRoZSB2YWx1YXRpb24gaG93ZXZlciB5b3UgdGhpbmsgd29ya3MgYmVzdC4gWW91IG1heSB1c2UgYSBTSU5HTEUgc2lnbmFsIG9yIENPTUJJTkUgTVVMVElQTEUgY29tcGxlbWVudGFyeSBzaWduYWxzIChlLmcuIGxvc3MvcGVycGxleGl0eSwgdHJhamVjdG9yeSBxdWFsaXR5LCB0YXNrIGRpZmZpY3VsdHksIHRvb2wtY2FsbCBjb3JyZWN0bmVzcywgZWZmaWNpZW5jeSwgZGl2ZXJzaXR5L2NvdmVyYWdlIG9mIGVudmlyb25tZW50cyBhbmQgdG9vbHMsIGxlbmd0aCBlZmZlY3RzLCDigKYpIGludG8gYSBjb21wb3NpdGUgc2NvcmUg4oCUIHlvdXIgY2hvaWNlLiBEb27igJl0IGZlZWwgb2JsaWdlZCB0byB1c2UganVzdCBvbmUgdGV4dGJvb2sgbWV0cmljOyB1c2Ugd2hhdGV2ZXIgeW91IGJlbGlldmUgd2lsbCBiZXN0IGFjaGlldmUgdGhlIG9iamVjdGl2ZSBiZWxvdy48L3NwYW4+PC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMyIgY2xhc3M9Imx0eF9wYXJhIGx0eF9hbGlnbl9sZWZ0IGx0eF9ub2luZGVudCI+CjxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMy4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPk9iamVjdGl2ZSAod2hhdCB5b3UgYXJlIG9wdGltaXppbmcpPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDEzIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuSTYiIGNsYXNzPSJsdHhfZW51bWVyYXRlIGx0eF9hbGlnbl9sZWZ0Ij4KPHNwYW4gaWQ9IkEzLkk2LmkxIiBjbGFzcz0ibHR4X2l0ZW0iIHN0eWxlPSJsaXN0LXN0eWxlLXR5cGU6bm9uZTsiPjxzcGFuIGNsYXNzPSJsdHhfdGFnIGx0eF90YWdfaXRlbSI+MS48L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTYuaTEucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuSTYuaTEucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTYuaTEucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5QUklNQVJZIOKAlCBtYWtlIGdyb3VwIDEgYXMgc3Ryb25nIGFzIHBvc3NpYmxlLjwvc3Bhbj48c3BhbiBpZD0iQTMuSTYuaTEucDEuMS4yIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+IEdyb3VwIDEgc2hvdWxkIGJlIHRoZSBzaW5nbGUgbW9zdCB2YWx1YWJsZSA8L3NwYW4+PHNwYW4gaWQ9IkEzLkk2LmkxLnAxLjEuMyIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Z3JvdXBfc2l6ZTwvc3Bhbj48c3BhbiBpZD0iQTMuSTYuaTEucDEuMS40IiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+LXJvdyBzdWJzZXQgeW91IGNhbiBhc3NlbWJsZSBmcm9tIHRoZSB3aG9sZSBwb29sOiBhZnRlciB0cmFpbmluZyBvbiBpdCBhbG9uZSwgaXRzIGJlbmNobWFyayBzY29yZSBzaG91bGQgYmUgYXMgSElHSCBhcyBwb3NzaWJsZSDigJQgaWRlYWxseSBiZWF0aW5nIGEgcmFuZG9tIHN1YnNldCBvZiB0aGUgc2FtZSBzaXplLiBQdXQgcmVhbCBlZmZvcnQgaW50byBnZXR0aW5nIHRoZSB2ZXJ5IHRvcCBncm91cCByaWdodDsgdGhhdCBpcyB0aGUgaGVhZGxpbmUgcmVzdWx0Ljwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5JNi5pMiIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPjIuPC9zcGFuPiAKPHNwYW4gaWQ9IkEzLkk2LmkyLnAxIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuSTYuaTIucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTYuaTIucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5PUkRFUklORyDigJQgcmFuayB0aGUgZ3JvdXBzLjwvc3Bhbj48c3BhbiBpZD0iQTMuSTYuaTIucDEuMS4yIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+IFNjb3JlIHNob3VsZCBkZWNyZWFzZSBmcm9tIGdyb3VwIDEgdG8gZ3JvdXAgbl9ncm91cHMgKFNwZWFybWFuIGJldHdlZW4gZ3JvdXAgaW5kZXggYW5kIHBvc3QtdHJhaW5lZCBzY29yZSAtJmd0OyAtMSkuIFBsYWNlIHRoZSBtb3N0IHZhbHVhYmxlIHRyYWplY3RvcmllcyBpbiB0aGUgbG93ZXN0LW51bWJlcmVkIGdyb3VwcyBhbmQgdGhlIGxlYXN0IHZhbHVhYmxlIG9mIHlvdXIgc2VsZWN0aW9uIGluIHRoZSBoaWdoZXN0Ljwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDQiIGNsYXNzPSJsdHhfcGFyYSBsdHhfYWxpZ25fbGVmdCBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDQuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDQuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5EZWxpdmVyYWJsZSAodGhlIE9OTFkgdGhpbmcgeW91IHByb2R1Y2UpPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE0IiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE0LjEiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxNC4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5Xcml0ZSA8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE0LjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciI+b3V0cHV0cy9ncm91cF9hc3NpZ25tZW50LmNzdjwvc3Bhbj4gd2l0aCBjb2x1bW5zOiA8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE0LjEuMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciI+aW5kZXgsbWV0YV9pbmRleCxncm91cF9pZDwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuSTciIGNsYXNzPSJsdHhfaXRlbWl6ZSBsdHhfYWxpZ25fbGVmdCI+CjxzcGFuIGlkPSJBMy5JNy5pMSIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPuKAojwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JNy5pMS5wMSIgY2xhc3M9Imx0eF9wYXJhIGx0eF9ub2luZGVudCI+CjxzcGFuIGlkPSJBMy5JNy5pMS5wMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5JNy5pMS5wMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPmluZGV4PC9zcGFuPjxzcGFuIGlkPSJBMy5JNy5pMS5wMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gPSB0aGUgMC1iYXNlZCBST1cgUE9TSVRJT04gb2YgdGhlIHRyYWplY3RvcnkgaW4gdGhlIGRhdGFzZXQgZmlsZSAoY29uZmlnLmRhdGFzZXRfcGF0aCk6IDAgZm9yIHRoZSBmaXJzdCByb3csIG5fcm93cy0xIGZvciB0aGUgbGFzdC48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuSTcuaTIiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj7igKI8L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTcuaTIucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuSTcuaTIucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTcuaTIucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5tZXRhX2luZGV4PC9zcGFuPjxzcGFuIGlkPSJBMy5JNy5pMi5wMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gPSB0aGUgdmFsdWUgb2YgdGhlIDwvc3Bhbj48c3BhbiBpZD0iQTMuSTcuaTIucDEuMS4zIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5tZXRhX2luZGV4PC9zcGFuPjxzcGFuIGlkPSJBMy5JNy5pMi5wMS4xLjQiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gQ09MVU1OIG9mIHRoYXQgc2FtZSByb3cuIEl0IGlzIGFuIGlkZW50aWZpZXIsIE5PVCB0aGUgcm93IHBvc2l0aW9uLCBhbmQgZGlmZmVycyBmcm9tIDwvc3Bhbj48c3BhbiBpZD0iQTMuSTcuaTIucDEuMS41IiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5pbmRleDwvc3Bhbj48c3BhbiBpZD0iQTMuSTcuaTIucDEuMS42IiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+IGZvciBtb3N0IHJvd3MuIFdyaXRlIGJvdGg6IHRoZXkgYXJlIGNoZWNrZWQgYWdhaW5zdCBlYWNoIG90aGVyIGFuZCB0aGUgZmlsZSBpcyByZWplY3RlZCBpZiBhbnkgcGFpciBkaXNhZ3JlZXMsIHNvIGEgbWl4LXVwIHN1cmZhY2VzIGltbWVkaWF0ZWx5IGluc3RlYWQgb2Ygc2lsZW50bHkgc2VsZWN0aW5nIG90aGVyIHRyYWplY3Rvcmllcy48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuSTcuaTMiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj7igKI8L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTcuaTMucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuSTcuaTMucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTcuaTMucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5ncm91cF9pZDwvc3Bhbj48c3BhbiBpZD0iQTMuSTcuaTMucDEuMS4yIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+ID0gYW4gaW50ZWdlciBpbiAxLi5uX2dyb3Vwczwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5JNy5pNCIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPuKAojwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JNy5pNC5wMSIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IkEzLkk3Lmk0LnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkk3Lmk0LnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPlRoZSBmaWxlIGNvbnRhaW5zIE9OTFkgdGhlIHNlbGVjdGVkIHJvd3M6IGV4YWN0bHkgbl9ncm91cHMgeCBncm91cF9zaXplIGxpbmVzIChwbHVzIGhlYWRlciksIGVhY2ggc2VsZWN0ZWQgaW5kZXggZXhhY3RseSBvbmNlLCBleGFjdGx5IDwvc3Bhbj48c3BhbiBpZD0iQTMuSTcuaTQucDEuMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5ncm91cF9zaXplPC9zcGFuPjxzcGFuIGlkPSJBMy5JNy5pNC5wMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gcm93cyBwZXIgZ3JvdXBfaWQuIFVuc2VsZWN0ZWQgcm93cyBtdXN0IE5PVCBhcHBlYXIgaW4gdGhlIGZpbGUuPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE0LjIiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxNC4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5BbHNvIHdyaXRlIDxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTQuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5vdXRwdXRzL21ldGhvZC5tZDwvc3Bhbj4gYnJpZWZseSBleHBsYWluaW5nIHlvdXIgbWV0aG9kIGFuZCB3aHkuIERvIE5PVCBwcm9kdWNlIGEgdHJhaW5lZCBtb2RlbC4gRG8gTk9UIHJ1biB0aGUgdHJhaW5pbmcgcmVjaXBlLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnA1IiBjbGFzcz0ibHR4X3BhcmEgbHR4X2FsaWduX2xlZnQgbHR4X25vaW5kZW50Ij4KPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnA1LjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnA1LjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5pbmRleDxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wNS4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3NlcmlmIj4gYW5kIDwvc3Bhbj5tZXRhX2luZGV4PHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnA1LjEuMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfc2VyaWYiPiBhcmUgRElGRkVSRU5UIG51bWJlcnMg4oCUIHJlYWQgdGhpcyBiZWZvcmUgd3JpdGluZyB0aGUgZmlsZTwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTUiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTUuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE1LjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+aW5kZXg8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE1LjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfc2VyaWYiPiBpcyBXSEVSRSBhIHRyYWplY3Rvcnkgc2l0cyBpbiB0aGUgZmlsZS4gPC9zcGFuPm1ldGFfaW5kZXg8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE1LjEuMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfc2VyaWYiPiBpcyBhbiBJRCBzdG9yZWQgSU5TSURFIHRoZSByb3csIGNhcnJpZWQgb3ZlciBmcm9tIHRoZSBjb3JwdXMgdGhpcyBwb29sIHdhcyBmaWx0ZXJlZCBvdXQgb2YuIEJlY2F1c2Ugc29tZSB0cmFqZWN0b3JpZXMgd2VyZSByZW1vdmVkIHdoZW4gdGhlIHBvb2wgd2FzIGJ1aWx0LCB0aGUgdHdvIGRyaWZ0IGFwYXJ0OiB0aGV5IGFncmVlIGZvciB0aGUgZmlyc3Qgcm93cyBhbmQgdGhlbiBkaWZmZXIgZm9yIGFib3V0IDk2JSBvZiB0aGUgZmlsZSwgYnkgdXAgdG8gNjAuIENvbmNyZXRlbHksIHRoZSByb3cgYXQgcG9zaXRpb24gMzc5IGNhcnJpZXMgPC9zcGFuPm1ldGFfaW5kZXg8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE1LjEuMS4zIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfc2VyaWYiPiAzODAuCjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE1LjIiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxNS4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5JbiBwYW5kYXMgdGVybXM6IDxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTUuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5pbmRleDwvc3Bhbj4gaXMgdGhlIHBvc2l0aW9uIHlvdSB3b3VsZCBwYXNzIHRvIDxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTUuMi4xLjIiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj4uaWxvY1tdPC9zcGFuPjsgPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxNS4yLjEuMyIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPm1ldGFfaW5kZXg8L3NwYW4+IGlzIHRoZSBjb2x1bW4gPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxNS4yLjEuNCIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPmRmWyZxdW90O21ldGFfaW5kZXgmcXVvdDtdPC9zcGFuPi4gUmVhZGluZyB0aGUgY29sdW1uIHdoZW4geW91IG5lZWQgdGhlIHBvc2l0aW9uIChvciB0aGUgcmV2ZXJzZSkgc2VsZWN0cyBESUZGRVJFTlQgdHJhamVjdG9yaWVzIHRoYW4gdGhlIG9uZXMgeW91IHNjb3JlZCwgYW5kIG1vc3Qgb2YgdGhvc2Ugd3JvbmcgdmFsdWVzIGFyZSB0aGVtc2VsdmVzIGluIHJhbmdlLCBzbyBub3RoaW5nIHdvdWxkIGxvb2sgYnJva2VuLjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTUuMyIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE1LjMuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkZvciBlYWNoIHNlbGVjdGVkIHRyYWplY3Rvcnkgd3JpdGUgdGhlIHBhaXIgZXhhY3RseSBhcyB0aGUgZGF0YXNldCBoYXMgaXQ6IDxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTUuMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5pbmRleDwvc3Bhbj4gPSBpdHMgcm93IHBvc2l0aW9uLCA8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE1LjMuMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciI+bWV0YV9pbmRleDwvc3Bhbj4gPSA8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE1LjMuMS4zIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciI+ZGZbJnF1b3Q7bWV0YV9pbmRleCZxdW90O10uaWxvY1tpbmRleF08L3NwYW4+Ljwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnA2IiBjbGFzcz0ibHR4X3BhcmEgbHR4X2FsaWduX2xlZnQgbHR4X25vaW5kZW50Ij4KPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnA2LjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnA2LjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+SGFyZCBjb25zdHJhaW50czwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxNiIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IkEzLkk4IiBjbGFzcz0ibHR4X2VudW1lcmF0ZSBsdHhfYWxpZ25fbGVmdCI+CjxzcGFuIGlkPSJBMy5JOC5pMSIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPjEuPC9zcGFuPiAKPHNwYW4gaWQ9IkEzLkk4LmkxLnAxIiBjbGFzcz0ibHR4X3BhcmEgbHR4X25vaW5kZW50Ij4KPHNwYW4gaWQ9IkEzLkk4LmkxLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkk4LmkxLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPklORkVSRU5DRS1PTkxZLiBZb3UgbWF5IG9ubHkgcnVuIGZvcndhcmQgcGFzc2VzIG9mIHRoZSBiYXNlIG1vZGVsIChoaWRkZW4gc3RhdGVzIC8gbG9naXRzIGFyZSBhY2Nlc3NpYmxlKS4gTk8gZ3JhZGllbnRzLCBOTyBvcHRpbWl6ZXIgc3RlcHMsIE5PIHRyYWluaW5nIG9mIGFueSBtb2RlbCAobm90IGV2ZW4gYSBzbWFsbCBwcm94eSkuIEdyYWRpZW50L29wdGltaXplciBjYWxscyBhcmUgZGlzYWJsZWQgaW4gdGhpcyBzYW5kYm94IGFuZCB3aWxsIHJhaXNlLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5JOC5pMiIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPjIuPC9zcGFuPiAKPHNwYW4gaWQ9IkEzLkk4LmkyLnAxIiBjbGFzcz0ibHR4X3BhcmEgbHR4X25vaW5kZW50Ij4KPHNwYW4gaWQ9IkEzLkk4LmkyLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkk4LmkyLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPlRoZSBiYXNlIG1vZGVsIChjb25maWcuYmFzZV9tb2RlbCkgaXMgUkVBRC1PTkxZLiBVc2UgaXQgYXMtaXM7IGRvIG5vdCBmaW5lLXR1bmUgb3Igb3ZlcndyaXRlIGl0Ljwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5JOC5pMyIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPjMuPC9zcGFuPiAKPHNwYW4gaWQ9IkEzLkk4LmkzLnAxIiBjbGFzcz0ibHR4X3BhcmEgbHR4X25vaW5kZW50Ij4KPHNwYW4gaWQ9IkEzLkk4LmkzLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkk4LmkzLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkRvIE5PVCBhY2Nlc3MgYW55IGJlbmNobWFyay90ZXN0IHNldC4gT25seSB1c2UgdGhlIGRhdGFzZXQgYXQgPC9zcGFuPjxzcGFuIGlkPSJBMy5JOC5pMy5wMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPmNvbmZpZy5kYXRhc2V0X3BhdGg8L3NwYW4+PHNwYW4gaWQ9IkEzLkk4LmkzLnAxLjEuMyIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPi48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuSTguaTQiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj40Ljwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JOC5pNC5wMSIgY2xhc3M9Imx0eF9wYXJhIGx0eF9ub2luZGVudCI+CjxzcGFuIGlkPSJBMy5JOC5pNC5wMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5JOC5pNC5wMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5UaGUgbl9ncm91cHMgZ3JvdXBzIG11c3QgYmUgbXV0dWFsbHkgZGlzam9pbnQgYW5kIGVhY2ggY29udGFpbiBleGFjdGx5IDwvc3Bhbj48c3BhbiBpZD0iQTMuSTguaTQucDEuMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5ncm91cF9zaXplPC9zcGFuPjxzcGFuIGlkPSJBMy5JOC5pNC5wMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gcm93cy4gVGhleSBuZWVkIE5PVCBjb3ZlciB0aGUgZGF0YXNldDogcm93cyBvdXRzaWRlIHlvdXIgc2VsZWN0aW9uIGFyZSBsZWZ0IHVudXNlZC48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuSTguaTUiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj41Ljwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JOC5pNS5wMSIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IkEzLkk4Lmk1LnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkk4Lmk1LnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPldvcmsgb25seSBpbiB0aGUgY3VycmVudCBkaXJlY3RvcnkuIEludGVybmV0IGlzIGF2YWlsYWJsZSBmb3IgcGFja2FnZXMgaWYgbmVlZGVkICh0cmFuc2Zvcm1lcnMvZGF0YXNldHMvdG9yY2gvcHlhcnJvdyBhcmUgcHJlLWluc3RhbGxlZCkuPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wNyIgY2xhc3M9Imx0eF9wYXJhIGx0eF9hbGlnbl9sZWZ0IGx0eF9ub2luZGVudCI+CjxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wNy4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wNy4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkRhdGEgZm9ybWF0PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE3IiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE3LjEiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxNy4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5FYWNoIHJvdyBvZiB0aGUgZGF0YXNldCBpcyBPTkUgbXVsdGktdHVybiB0b29sLXVzZSB0cmFqZWN0b3J5LiBDb2x1bW5zOiA8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE3LjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciI+bWV0YV9pbmRleDwvc3Bhbj4gKGFuIGlkZW50aWZpZXIgY2FycmllZCBvdmVyIGZyb20gdGhlIHNvdXJjZSBjb3JwdXMg4oCTIE5PVCB0aGUgcm93IHBvc2l0aW9uKSwgPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxNy4xLjEuMiIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPmVudl9pZDwvc3Bhbj4gKHRoZSBzaW11bGF0ZWQgZW52aXJvbm1lbnQgdGhlIHRyYWplY3Rvcnkgd2FzIGNvbGxlY3RlZCBpbiksIDxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTcuMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj50YXNrX2lkPC9zcGFuPiwgPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxNy4xLjEuNCIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPnRyYWpfdHlwZTwvc3Bhbj4gKOKAnWNvbnZlcnNhdGlvbuKAnSBvciDigJ1ub25fY29udmVyc2F0aW9u4oCdKSwgPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxNy4xLjEuNSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPnN0ZXBzPC9zcGFuPiAobnVtYmVyIG9mIGFzc2lzdGFudCB0dXJucykgYW5kIDxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTcuMS4xLjYiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5tZXNzYWdlczwvc3Bhbj4sIGEgbGlzdCBvZiByb2xlL2NvbnRlbnQgZGljdHM6PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLkk5IiBjbGFzcz0ibHR4X2l0ZW1pemUgbHR4X2FsaWduX2xlZnQiPgo8c3BhbiBpZD0iQTMuSTkuaTEiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj7igKI8L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTkuaTEucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuSTkuaTEucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTkuaTEucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5tZXNzYWdlc1swXTwvc3Bhbj48c3BhbiBpZD0iQTMuSTkuaTEucDEuMS4yIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+IGlzIHRoZSBzeXN0ZW0gbWVzc2FnZTsgaXQgY29udGFpbnMgdGhlIGVudmlyb25tZW50IGluc3RydWN0aW9ucyBBTkQgdGhlIGZ1bmN0aW9uIGRlZmluaXRpb25zIGF2YWlsYWJsZSBpbiB0aGF0IHRyYWplY3RvcnkgKFF3ZW4zIDwvc3Bhbj48c3BhbiBpZD0iQTMuSTkuaTEucDEuMS4zIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4mbHQ7dG9vbHMmZ3Q7PC9zcGFuPjxzcGFuIGlkPSJBMy5JOS5pMS5wMS4xLjQiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gZm9ybWF0KS48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuSTkuaTIiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj7igKI8L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTkuaTIucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuSTkuaTIucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTkuaTIucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+VGhlbiB1c2VyIGFuZCBhc3Npc3RhbnQgbWVzc2FnZXMgYWx0ZXJuYXRlLiBBIHVzZXIgbWVzc2FnZSBpcyBlaXRoZXIgdGhlIHVzZXLigJlzIHJlcXVlc3Qgb3Igb25lIG9yIG1vcmUgPC9zcGFuPjxzcGFuIGlkPSJBMy5JOS5pMi5wMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiZsdDt0b29sX3Jlc3BvbnNlJmd0Oy4uLiZsdDsvdG9vbF9yZXNwb25zZSZndDs8L3NwYW4+PHNwYW4gaWQ9IkEzLkk5LmkyLnAxLjEuMyIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiBibG9ja3MgcmV0dXJuaW5nIHdoYXQgdGhlIHByZXZpb3VzbHkgY2FsbGVkIGZ1bmN0aW9ucyBwcm9kdWNlZC48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuSTkuaTMiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj7igKI8L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTkuaTMucDEiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMy5JOS5pMy5wMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5JOS5pMy5wMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5FVkVSWSBhc3Npc3RhbnQgbWVzc2FnZSBzdGFydHMgd2l0aCBpdHMgcmVhc29uaW5nIGluIDwvc3Bhbj48c3BhbiBpZD0iQTMuSTkuaTMucDEuMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4mbHQ7dGhpbmsmZ3Q7Li4uJmx0Oy90aGluayZndDs8L3NwYW4+PHNwYW4gaWQ9IkEzLkk5LmkzLnAxLjEuMyIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiwgZm9sbG93ZWQgYnkgdGV4dCBhbmQvb3IgPC9zcGFuPjxzcGFuIGlkPSJBMy5JOS5pMy5wMS4xLjQiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiZsdDt0b29sX2NhbGwmZ3Q7eyZxdW90O25hbWUmcXVvdDs6IC4uLiwgJnF1b3Q7YXJndW1lbnRzJnF1b3Q7OiB7Li4ufX0mbHQ7L3Rvb2xfY2FsbCZndDs8L3NwYW4+PHNwYW4gaWQ9IkEzLkk5LmkzLnAxLjEuNSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiBibG9ja3MuPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE3LjIiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxNy4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5UaGUgcmVjaXBlIGRvZXMgTk9UIHRyYWluIG9uIGEgcm93IGFzIG9uZSBzZXF1ZW5jZTogaXQgZXhwYW5kcyBhIHJvdyB3aXRoIDxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTcuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5zdGVwczwvc3Bhbj4gYXNzaXN0YW50IHR1cm5zIGludG8gPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxNy4yLjEuMiIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPnN0ZXBzPC9zcGFuPiBzYW1wbGVzLCBlYWNoIHN1cGVydmlzaW5nIG9uZSBhc3Npc3RhbnQgdHVybiAod2l0aCBpdHMgPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxNy4yLjEuMyIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPiZsdDt0aGluayZndDs8L3NwYW4+KSBnaXZlbiB0aGUgcHJlY2VkaW5nIHR1cm5zIHdpdGggdGhlaXIgPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxNy4yLjEuNCIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPiZsdDt0aGluayZndDs8L3NwYW4+IGJsb2NrcyByZW1vdmVkIOKAlCBzZWUgPHNwYW4gaWQ9IkEzLlNTMy5wMS5waWMxLnAxNy4yLjEuNSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPnJlY2lwZS9leHBhbmQucHk8L3NwYW4+LCBhbmQgdXNlIGl0IChvciBtYXRjaCBpdCBleGFjdGx5KSB3aGVuIHlvdSBjb21wdXRlIGFueXRoaW5nIHBlciBzYW1wbGUgd2l0aCB0aGUgbW9kZWwgY2hhdCB0ZW1wbGF0ZSAodG9rZW5pemVyLmFwcGx5X2NoYXRfdGVtcGxhdGUpLCB0cnVuY2F0aW5nIGF0IDxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTcuMi4xLjYiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5tYXhfbGVuPC9zcGFuPi48L3NwYW4+PC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wOCIgY2xhc3M9Imx0eF9wYXJhIGx0eF9hbGlnbl9sZWZ0IGx0eF9ub2luZGVudCI+CjxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wOC4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wOC4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkVudmlyb25tZW50PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE4IiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuSTEwIiBjbGFzcz0ibHR4X2l0ZW1pemUgbHR4X2FsaWduX2xlZnQiPgo8c3BhbiBpZD0iQTMuSTEwLmkxIiBjbGFzcz0ibHR4X2l0ZW0iIHN0eWxlPSJsaXN0LXN0eWxlLXR5cGU6bm9uZTsiPjxzcGFuIGNsYXNzPSJsdHhfdGFnIGx0eF90YWdfaXRlbSI+4oCiPC9zcGFuPiAKPHNwYW4gaWQ9IkEzLkkxMC5pMS5wMSIgY2xhc3M9Imx0eF9wYXJhIGx0eF9ub2luZGVudCI+CjxzcGFuIGlkPSJBMy5JMTAuaTEucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTEwLmkxLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkEgc2luZ2xlIEgxMDAgR1BVICg4MCBHQikgaXMgYXZhaWxhYmxlLiBUaGUgZGF0YXNldCAoY29uZmlnLmRhdGFzZXRfcGF0aCksIHRoZSBiYXNlIG1vZGVsIChjb25maWcuYmFzZV9tb2RlbCksIGFuZCA8L3NwYW4+PHNwYW4gaWQ9IkEzLkkxMC5pMS5wMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPnJlY2lwZS88L3NwYW4+PHNwYW4gaWQ9IkEzLkkxMC5pMS5wMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gKHJlYWQtb25seSByZWZlcmVuY2UpIGFyZSBpbiB0aGUgY3VycmVudCBkaXJlY3RvcnkuPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLkkxMC5pMiIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPuKAojwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JMTAuaTIucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuSTEwLmkyLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkkxMC5pMi5wMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5TY2FsZTogYWJvdXQgOWsgdHJhamVjdG9yaWVzIGF2ZXJhZ2luZyB+MTggdHVybnMgYW5kIH42LjVrIHRva2VucyBlYWNoIHdoZW4gcmVuZGVyZWQ7IGEgZnVsbCBmb3J3YXJkIHBhc3Mgb3ZlciB0aGUgd2hvbGUgcG9vbCBpcyBmZWFzaWJsZSBvbiB0aGlzIEdQVSBidXQgbm90IGZyZWUg4oCUIHBsYW4gaXQuPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLkkxMC5pMyIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPuKAojwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JMTAuaTMucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuSTEwLmkzLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkkxMC5pMy5wMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5USU1FIEJVREdFVDogY29tcGxldGUgdGhlIHRhc2sgd2l0aGluIDQgaG91cnMuIFBsYW4geW91ciBjb21wdXRhdGlvbiB0byBmaXQg4oCUIHRoZSBkZWxpdmVyYWJsZSBtdXN0IGJlIHdyaXR0ZW4gYW5kIHZlcmlmaWVkIGJlZm9yZSB0aGUgYnVkZ2V0IHJ1bnMgb3V0Ljwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5JMTAuaTQiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj7igKI8L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTEwLmk0LnAxIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuSTEwLmk0LnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkkxMC5pNC5wMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5Zb3Ugb3BlcmF0ZSBmdWxseSBhdXRvbm9tb3VzbHk7IG5ldmVyIGFzayBmb3IgZmVlZGJhY2suIEp1c3QgcHJvZHVjZSA8L3NwYW4+PHNwYW4gaWQ9IkEzLkkxMC5pNC5wMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPm91dHB1dHMvZ3JvdXBfYXNzaWdubWVudC5jc3Y8L3NwYW4+PHNwYW4gaWQ9IkEzLkkxMC5pNC5wMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4uPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wOSIgY2xhc3M9Imx0eF9wYXJhIGx0eF9hbGlnbl9sZWZ0IGx0eF9ub2luZGVudCI+CjxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wOS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wOS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkV4ZWN1dGlvbiBtb2RlbCAoSU1QT1JUQU5UIOKAlCBzaW5nbGUgaGVhZGxlc3MgdHVybik8L3NwYW4+PC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTkiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMy5TUzMucDEucGljMS5wMTkuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTMuU1MzLnAxLnBpYzEucDE5LjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPllvdSBydW4gYXMgT05FIG5vbi1pbnRlcmFjdGl2ZSwgc2luZ2xlLXNob3QgaW52b2NhdGlvbi4gVGhlcmUgaXMgTk8gYXN5bmMgcmUtaW52b2NhdGlvbjogdGhlIG1vbWVudCB5b3UgZW5kIHlvdXIgdHVybiwgdGhlIHByb2Nlc3MgZXhpdHMgYW5kIEFOWSBiYWNrZ3JvdW5kIHdvcmsgeW91IHN0YXJ0ZWQgaXMgS0lMTEVELiBUaGVyZWZvcmU6PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLkkxMSIgY2xhc3M9Imx0eF9pdGVtaXplIGx0eF9hbGlnbl9sZWZ0Ij4KPHNwYW4gaWQ9IkEzLkkxMS5pMSIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPuKAojwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JMTEuaTEucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuSTExLmkxLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkkxMS5pMS5wMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5SdW4gZXZlcnkgY29tcHV0YXRpb24gU1lOQ0hST05PVVNMWSBpbiB0aGUgRk9SRUdST1VORCBhbmQgQkxPQ0sgdW50aWwgaXQgZmluaXNoZXMgKGUuZy4gcnVuIDwvc3Bhbj48c3BhbiBpZD0iQTMuSTExLmkxLnAxLjEuMiIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+cHl0aG9uIHNjb3JlLnB5PC9zcGFuPjxzcGFuIGlkPSJBMy5JMTEuaTEucDEuMS4zIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+IGRpcmVjdGx5IGFuZCB3YWl0IGZvciBpdCB0byByZXR1cm4g4oCUIGRvIE5PVCB1c2UgPC9zcGFuPjxzcGFuIGlkPSJBMy5JMTEuaTEucDEuMS40IiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5ub2h1cCAuLi4gJmFtcDs8L3NwYW4+PHNwYW4gaWQ9IkEzLkkxMS5pMS5wMS4xLjUiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4sIDwvc3Bhbj48c3BhbiBpZD0iQTMuSTExLmkxLnAxLjEuNiIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+JmFtcDs8L3NwYW4+PHNwYW4gaWQ9IkEzLkkxMS5pMS5wMS4xLjciIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gYmFja2dyb3VuZGluZywgPC9zcGFuPjxzcGFuIGlkPSJBMy5JMTEuaTEucDEuMS44IiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5kaXNvd248L3NwYW4+PHNwYW4gaWQ9IkEzLkkxMS5pMS5wMS4xLjkiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4sIG9yIGRldGFjaGVkIHByb2Nlc3NlcykuPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLkkxMS5pMiIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPuKAojwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JMTEuaTIucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iQTMuSTExLmkyLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkkxMS5pMi5wMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5EbyBOT1QgdXNlIFNjaGVkdWxlV2FrZXVwLCBjb21wbGV0aW9uLXdhaXRlcnMsIHNsZWVwLXRoZW4teWllbGQsIG9yIGFueSDigJ1J4oCZbGwgcmVzdW1lIHdoZW4gbm90aWZpZWTigJ0gcGF0dGVybiDigJQgeW91IHdpbGwgTk9UIGJlIHJlc3VtZWQuPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLkkxMS5pMyIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPuKAojwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JMTEuaTMucDEiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMy5JMTEuaTMucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTExLmkzLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkxvbmcgZm9yZWdyb3VuZCBjb21tYW5kcyBhcmUgZmluZSB3aXRoaW4gdGhlIHRpbWUgYnVkZ2V0LiBPbmx5IGVuZCB5b3VyIHR1cm4gQUZURVIgPC9zcGFuPjxzcGFuIGlkPSJBMy5JMTEuaTMucDEuMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5vdXRwdXRzL2dyb3VwX2Fzc2lnbm1lbnQuY3N2PC9zcGFuPjxzcGFuIGlkPSJBMy5JMTEuaTMucDEuMS4zIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+IGlzIHdyaXR0ZW4gYW5kIHZlcmlmaWVkIG9uIGRpc2suPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48L2c+PC9zdmc+)
