---
identifier: arxiv:2601.18207v1
title: "PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"
authors:
  - James Burgess
  - Jan N. Hansen
  - Duo Peng
  - Yuhui Zhang
  - Alejandro Lozano
  - Min Woo Sun
  - Emma Lundberg
  - Serena Yeung-Levy
published: "2026-01-26T06:46:16+00:00"
url: https://arxiv.org/abs/2601.18207v1
source: arxiv
doi: null
arxiv_id: 2601.18207v1
categories:
  - cs.AI
  - cs.CL
  - cs.IR
  - cs.LG
---

# PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR

James Burgess ^(†)^(†)thanks: Correspondence: jmhb@stanford.edu
Affiliation: Stanford University    Jan N. Hansen Affiliation: Stanford
University    Duo Peng Affiliation: Chan Zuckerberg Biohub Network   
Yuhui Zhang Affiliation: Stanford University    Alejandro Lozano
Affiliation: Stanford University    Min Woo Sun Affiliation: Stanford
University    Emma Lundberg Affiliation: Stanford University
Affiliation: Chan Zuckerberg Biohub Network Affiliation: KTH Royal
Institute of Technology  [Project
Page](https://jmhb0.github.io/PaperSearchQA)
 [Datasets](https://huggingface.co/collections/jmhb/papersearchqa)
 [Code](https://github.com/jmhb0/PaperSearchQA)    Serena Yeung-Levy
Affiliation: Stanford University Affiliation: Chan Zuckerberg Biohub
Network

###### Abstract

Search agents are language models (LMs) that reason and search knowledge
bases (or the web) to answer questions; recent methods supervise only
the final answer accuracy using reinforcement learning with verifiable
rewards (RLVR). Most RLVR search agents tackle general-domain QA, which
limits their relevance to technical AI systems in science, engineering,
and medicine. In this work we propose training agents to search and
reason over scientific papers – this tests technical question-answering,
it is directly relevant to real scientists, and the capabilities will be
crucial to future AI Scientist systems. Concretely, we release a search
corpus of 16 million biomedical paper abstracts and construct a
challenging factoid QA dataset called PaperSearchQA with 60k samples
answerable from the corpus, along with benchmarks. We train search
agents in this environment to outperform non-RL retrieval baselines; we
also perform further quantitative analysis and observe interesting agent
behaviors like planning, reasoning, and self-verification. Our corpus,
datasets, and benchmarks are usable with the popular Search-R1 codebase
for RLVR training and released on [Hugging
Face](https://huggingface.co/collections/jmhb/papersearchqa). Finally,
our data creation methods are scalable and easily extendable to other
scientific domains.

## 1 Introduction

![Refer to caption](2601.18207v1/x1.png)

Figure 1: Search agents interleave reasoning and retrieval for question
answering (QA). We study QA over scientific literature, contributing an
environment for training agents with RL with verifiable rewards (RLVR).
We release a training dataset of factoid QA (yellow boxes), a retrieval
corpus (purple), and benchmarks.

Following the release of Deepseek-R1 [Guo et al. (2025)](#bib.bib18) and
OpenAI’s o1 [Jaech et al. (2024)](#bib.bib19), much large language model
(LLM) research has employed reinforcement learning with verifiable
rewards (RLVR) [Shao et al. (2024)](#bib.bib21); [Lambert et al.
(2024)](#bib.bib20). In RLVR, an LLM is prompted to answer a query, and
a reward is given only if an automatic verifier deems the final output
correct; the corresponding tokens are then used to update the model
([appendix D](#A4 "Appendix D Further explanation of reinforcement learning with verifiable rewards (RLVR) ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR")
has more details). This differs from supervised finetuning (SFT), which
learns directly from labeled text sequences. Early follow-up work
focused on math and code applications [Chen et al. (2025)](#bib.bib22),
followed by tool-use agents [Feng et al. (2025)](#bib.bib23) where the
LLM both calls tools and reasons over their outputs to complete tasks.
Compared to earlier approaches to controlling agents such as prompting,
scaffolding, and supervised finetuning, RLVR is appealing for its
potential to incentivize more general and flexible reasoning and
behavior [Chu et al. (2025)](#bib.bib24); [Guo et al.
(2025)](#bib.bib18).

One major application of tool-use LLMs is knowledge-intensive
question-answering. Here, search agents can reason about the query and
search over knowledge bases (KBs) in an interleaved fashion [Yao et al.
(2023)](#bib.bib25); [Trivedi et al. (2022)](#bib.bib26); [Jin et al.
(2025)](#bib.bib1). RLVR was shown to be effective for training search
agents by Search-R1 [Jin et al. (2025)](#bib.bib1), along with many
concurrent and follow-up papers [Song et al. (2025)](#bib.bib28); [Sun
et al. (2025)](#bib.bib34); [Zheng et al. (2025)](#bib.bib2). However
these works emphasize general-knowledge QA that test simple trivia
[Kwiatkowski et al. (2019)](#bib.bib5); [Yang et al. (2018)](#bib.bib6);
[Joshi et al. (2017)](#bib.bib7); [Ho et al. (2020)](#bib.bib8), and not
technical and knowledge-intensive domains like science, engineering,
law, and medicine. These require more technical knowledge, reasoning
about complex systems, and ability to search technical knowledge bases.

One promising setting for training technical reinforcement learning (RL)
search agents is in scientific AI systems [Lu et al. (2024)](#bib.bib9);
[Gao et al. (2024)](#bib.bib59). Scientific research has a huge volume
of knowledge in databases and literature [Ferguson et al.
(2014)](#bib.bib12); [Delile et al. (2024)](#bib.bib13), and traversing
that knowledge is an essential part of every stage of the research
process [Hope et al. (2023)](#bib.bib51). The interest in AI search has
been established by literature retrieval systems [Lála et al.
(2023)](#bib.bib15); [Asai et al. (2024)](#bib.bib16), and we predict
that future complex agent systems for AI research will include modules
for searching scientific literature and knowledge bases [Lu et al.
(2024)](#bib.bib9). These search modules will require specialist domain
understanding to properly perform query formulation, to reason about
retrieved information, and to evaluate the quality of the retrieved
information.

In this work, we propose training RL search agents to search and reason
over a corpus of research papers to answer scientific questions. We
focus on easily-verified factoid questions, for example What gene is
mutated in childhood retinoblastoma? (Answer RB1); such queries are
amenable to current RLVR training, while also being useful to practicing
scientists [Krithara et al. (2023)](#bib.bib17). Specifically, we first
release a corpus and search index of 16 million abstracts from
biomedical papers in PubMed. Second, we release a dataset of 60k factoid
QAs; the datasets are generated from the Pubmed articles in an LLM
workflow, that underwent rigorous quality assurance by biology experts
for correctness and relevance to a real scientific search application.
The data creation methods are highly scalable, and can be adapted to
other domains like materials science or chemistry. Third, and for
benchmarks, we reserve 5k samples for testing, and we re-distribute the
factoid subset of BioASQ, a small scale but high quality human-created
dataset [Krithara et al. (2023)](#bib.bib17).

We train LLM search agents in our environment, showing that current RL
training techniques [Jin et al. (2025)](#bib.bib1) lead to stronger
performance compared to non-RL baselines. However the overall scores
remain low, which establish our datasets as challenging for training
search systems. We perform quantitative analysis, finding:
general-domain semantic retrievers offer small benefits compared to
syntactic retrievers; LLMs without retrievers have non-negligible
performance; gains to accuracy with model size are likely due to better
parametric knowledge; and paraphrasing in dataset construction adds
dataset difficulty. Additionally, our qualitative results show
interesting behaviors, specifically simple planning about query
rewriting, reasoning about questions before retrieval, and verification
when the model already has an initial answer.

In summary, our contributions are: - A new environment for training
search agents in scientific question answering over papers: specifically
a corpus, training datasets, and benchmarks. - Demonstrating successful
RLVR training of search agents over scientific papers, with quantitative
and qualitative insights.

## 2 Related Work

We review general-domain search agents, followed by systems for
understanding scientific literature.

### 2.1 Search agents

Search-R1 [Jin et al. (2025)](#bib.bib1) and R1-Searcher [Song et al.
(2025)](#bib.bib28) were the first open search agents for question
answering trained using reinforcement learning with final-answer reward.
(Closed systems like OpenAI’s o3 [OpenAI (2025b)](#bib.bib36) and Deep
Research likely explored this earlier [OpenAI (2025a)](#bib.bib37)).
There were many followups exploring, for example, search in web
environments [Zheng et al. (2025)](#bib.bib2); [Li et al.
(2025c)](#bib.bib33), query decomposition [Guan et al.
(2025)](#bib.bib35), and simulating the retrieval environment [Sun et
al. (2025)](#bib.bib34). We contribute to this direction by proposing
new RL training environments; while prior works emphasize general
knowledge QA, we create datasets, evaluations, and a retrieval corpus
for training agents to reason over scientific literature. Earlier,
search agents (and RAG systems) were supervised with supervised
fine-tuning [Schick et al. (2023)](#bib.bib39), few-shot prompting [Yao
et al. (2023)](#bib.bib25); [Trivedi et al. (2022)](#bib.bib26), or
prompt optimization [Opsahl-Ong et al. (2024)](#bib.bib38); these
approaches likely lead to worse generalization [Chu et al.
(2025)](#bib.bib24); [Guo et al. (2025)](#bib.bib18). Concurrent with
recent search agents, many train agents with RL for tool use beyond
search engines [Feng et al. (2025)](#bib.bib23); [Qian et al.
(2025)](#bib.bib3).

### 2.2 Search Agents for Scientific QA

BioASQ [Tsatsaronis et al. (2015)](#bib.bib30); [Krithara et al.
(2023)](#bib.bib17) is an annual challenge run since 2012 for
benchmarking semantic indexing and open-domain question-answering for
scientific literature – its popularity reflects the importance of
literature understanding tasks for practicing scientists. Their task
definitions influence our dataset construction, though a limitation is
that their human-generated data is hard to scale. There are many systems
for open-domain question-answering over literature, include PaperQA
[Lála et al. (2023)](#bib.bib15); [Skarlinski et al. (2024)](#bib.bib14)
and OpenScholar [Asai et al. (2024)](#bib.bib16). They have impressive
capabilities, handling large corpora of full-text articles, however the
agent behavior is controlled by component scaffolding, prompt
engineering, or supervised fine-tuning. Instead, we explore training
agents with RL because it promises stronger generalization in the long
term [Chu et al. (2025)](#bib.bib24); [Jin et al. (2025)](#bib.bib1). To
make progress in this direction, we focus on factoid QA, where answers
are easy to unambiguously verify. Note that current RL-trained search
agents are designed for factoid QA [Jin et al. (2025)](#bib.bib1), while
such questions are useful to applications [Krithara et al.
(2023)](#bib.bib17). This motivates us generating new datasets, since
prior datasets have binary answers [Jin et al. (2019)](#bib.bib40);
[Wadden et al. (2020)](#bib.bib41), have long-form answers with fuzzy
evaluation [Asai et al. (2024)](#bib.bib16); [Lee et al.
(2023)](#bib.bib42), or they have smaller scale [Skarlinski et al.
(2024)](#bib.bib14).

## 3 Methods

In the following sections, we first describe the training data
construction process, then the search corpus and indexing, and finally
the RL training algorithm.

### 3.1 Dataset Construction

![Refer to caption](2601.18207v1/x2.png)

Figure 2: Left: the ten question-answering categories defined with
experts. Right: example question-answer pairs, which are sufficient
supervision for RLVR training methods.

#### Defining Dataset Properties

The first main goal is that question-answer pairs (QA’s) can serve as
training data for methods needing outcome supervision – for example,
reinforcement learning with verifiable rewards (RLVR) [Jin et al.
(2025)](#bib.bib1). Specifically, the answers mut be verifiable – it
should be possible for a reward model to judge whether the prediciton
matches the ground truth answer without any ambiguity. To satisfy
verifiability, we make the following design decisions. QA’s are factoid,
meaning the answer is a single entity; this is similar to the most
popular general-knowledge QA datasets studied by search agents
[Kwiatkowski et al. (2019)](#bib.bib5); [Joshi et al.
(2017)](#bib.bib7). (Alternative and more complex formulations, like
‘list of entities’ have been left to future work). The questions are
unambiguous: written so that only a single entity name (or its synonyms)
are correct. Then, the reward model is simply checking whether the
prediction is equal to the ground truth answer (or its synonyms). We
ensure questions have a low ‘random guessing baseline’, because this can
lead to incorrect reasoning frequently being rewarded, which is noisy
supervision. In particular, we do not allow binary answers (e.g. True or
False), and our quality control process ensures that the question text
rarely gives a small list of options. Another property – implicit in our
construction pipeline – is that questions are single-hop, meaning they
can be answered from a single correctly-retrieved document. Since we
employ outcome-only reward, we do not require annotations for
intermediate reasoning or for retrieved documents.

The second main goal is that QA’s should be relevant to real
applications: they must be questions that real scientists might ask in
their work. To ensure this, our team includes practicing scientists at
all stages – from defining task properties to pipeline construction to
verifying the data. Our construction pipeline also take inspiration from
the BioASQ project [Krithara et al. (2023)](#bib.bib17); [Nentidis et
al. (2023)](#bib.bib29); [Tsatsaronis et al. (2015)](#bib.bib30) – a
challenge for semantic indexing and question-answering (including
factoid-QA) over biomedical articles – that has run since 2015, and
garnered significant attention in bioinformatics and NLP. While a
limitation of BioASQ is that questions are human-created and therefore
difficult to scale, it clearly demonstrates the significant interest in
biomedical question answering over scientific papers; this supports our
claim that PaperSearchQA is interesting to applications.

#### Categories for Question-Answering

To ensure the QA-generation pipeline produces questions that satisfy our
key target properties – unambiguous factoid and relevant to application
– we defined ten target question categories. The categories and examples
are shown in
[Figure 2](#S3.F2 "In 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").

To develop these, first the human experts on our team performed
brainstorming to identify one candidate category set. Next, we sampled
300 questions from the BioASQ database and used LLMs (Claude Opus 4
[Anthropic (2025)](#bib.bib62) and OpenAI o3 [OpenAI
(2025b)](#bib.bib36)) to propose two more candidate category sets. Then,
the human experts synthesized those into a final list, which required
some merging and discarding rare categories. The final category names
with examples were used in the data construction pipeline.

![Refer to caption](2601.18207v1/x3.png)

Figure 3: Data generation pipeline process. Left, generating the
categories from
[Figure 2](#S3.F2 "In 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"):
LLM summarizes categories from human-written questions in BioASQ
[Krithara et al. (2023)](#bib.bib17); humans brainstorm categories in
parallel; humans synthesize both sources into final categories. Right,
QA generation: abstracts from PubMed are sampled and passed to an LLM.
The LLM s prompted with categories (and other guidance) to generate QAs.
A second LLM paraphrases the QAs to limit exact keyword matching.

#### Automatic QA Generation Pipeline

The data generation process
([Figure 3](#S3.F3 "In Categories for Question-Answering ‣ 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"))
uses paper abstracts as a knowledge source, which are then mapped to QAs
using an LLM workflow. The LLM prompts and pipeline architecture were
iteratively designed based on expert review from biomedical scientists.
Specifically we generate 200 questions, the expert provides text
feedback; the human prompt engineer then modifies the workflow topology
or the LLM instructions with metaprompting [Schulhoff et al.
(2024)](#bib.bib32).

First, the paper abstracts are randomly sampled from the corpus
described in
[Section 3.3](#S3.SS3 "3.3 Retrieval Corpus and Index ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR")
– the same corpus that is searched at inference time. The abstract is
passed to an LLM (GPT-4.1 [Achiam et al. (2023)](#bib.bib31)) with a
carefully-designed prompt (see
[Appendix E](#A5 "Appendix E Data construction pipeline ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR")).
This prompt includes the target categories from
[Figure 2](#S3.F2 "In 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
along with guidance ensuring the questions are suitable for open-domain
QA: factoid answers, no acronyms, and no assumed access to the document
(and we add an extra filtering step for phrases like ‘this study’). We
found that generating three questions per abstract led to better dataset
diversity.

Since reward models commonly use exact match comparison of prediction
and target, we generate synonyms of ‘golden answers’, using GPT-4.1
(prompt in
[Appendix E](#A5 "Appendix E Data construction pipeline ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR")).
Next, we notice that questions often use exact keywords and phrasing
found in the abstract, while realistic use-cases would often use
synonyms. We therefore sample 50% of QAs for question rewriting, and use
an LLM prompt to ‘paraphrase’ the question with different terminology
(prompt in
[Appendix E](#A5 "Appendix E Data construction pipeline ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR")).
Finally, dataset is split into train and test randomly.

All LLM calls were made through OpenRouter. The total cost, including
experimentation and final data generation, was estimated at \$600.

#### Dataset Summary

The final PaperSearchQAdataset has 54,907 training samples and 5,000
test samples. For question categories
[Figure 2](#S3.F2 "In 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
the top categories are ‘Experimental & computational methods’ (27%) and
‘Therapeutics, indications & clinical evidence’. Median question word
length is 18 and median answer word length is 2. Each sample is
annotated with the Pubmed ID of the source paper, the category, and
whether the question was paraphrased to avoid easy keyword matching. It
is available on Hugging Face Hub and is released with a CC-BY license.

### 3.2 Evaluation dataset: BioASQ

BioASQ is a popular challenge for biomedical indexing and question
answering, where all samples are human-creating [Krithara et al.
(2023)](#bib.bib17); [Tsatsaronis et al. (2015)](#bib.bib30). Due to
it’s smaller scale, we propose using it for search agent evaluation,
where the search corpus is the same PubMed abstracts from
[Section 3.3](#S3.SS3 "3.3 Retrieval Corpus and Index ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
For convenience, we collect data from all years up to 2025 and
redistribute it on Huggingface Hub. Our only addition is to generate
synonyms for the answer into the ‘golden answer’ list (using the same
LLM call from our own pipeline) which enables exact-match evaluation
metric. It is released under CC-BY-2.5 license. Its ‘factoid’ dataset
has 1,609 samples. BioASQ has question categories other than factoid –
yes/no, list, and summary – which we also release, though we do not use
it in this paper.

### 3.3 Retrieval Corpus and Index

The search corpus is 16 million PubMed abstracts up to 2025, and was
previously distributed by BioASQ [Krithara et al. (2023)](#bib.bib17)¹¹
1 PubMed abstracts originally sourced from [National Library of
Medicine](https://www.nlm.nih.gov/databases/download.html) . We
concatenate the paper title with the abstract text, giving a mean word
length of 245.

We provide BM25 [Robertson and Walker (1994)](#bib.bib63) and e5 [Wang
et al. (2022)](#bib.bib64) search indexes. The corpus and index is small
enough to hold in memory: the corpus is 23GB, the BM25 index is 2.6GB,
and the e5 index is 93GB. At inference time, the e5 retriever index
requires two A100s GPUs (80GB) to avoid memory error at inference.

### 3.4 Training Algorithms

To demonstrate the value of our datasets and retriever, we train search
agents using RLVR.

#### RLVR for Search Agents

We follow Search-R1 [Jin et al. (2025)](#bib.bib1), which uses
reinforcement learning with verifiable rewards (RLVR).

We provide a minimal system prompt
([Appendix F](#A6 "Appendix F System prompt for Search-R1 LLM training ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR")),
which introduces the question-answering task, instructing the model to
leverage reasoning tokens inside \<think\> tokens and to give the final
answer inside \<answer\> tokens. The prompt then describes usage of
search: by wrapping queries in \<query\> tokens. When a query is found,
the system stops generation, extracts the query, and retrieves the top
$`k`$ documents. It then appends the documents to the reasoning trace,
and then continues token generation. Crucially, this system prompt
provides minimal specific guidance about how to perform reasoning and
query rewriting – this allows behaviors to be learned in RL training in
a manner that (hopefully) is more flexible and general [Chu et al.
(2025)](#bib.bib24).

In training, the search agent performs rollouts of token generation and
search. The final answer is extracted and we compute a very simple
reward: 1 if the prediction matches any of the target answers, and 0
otherwise. Reward is applied to all LLM-generated tokens uniformly,
except for the retrieved tokens that are masked out during gradient
computation. More formally (as in Search-R1 [Jin et al.
(2025)](#bib.bib1)) we learn the weights for the policy LLM,
$`\pi_{\theta}`$, conditioned on a retrieval engine $`\mathcal{R}`$
using a QA dataset, $`\mathcal{D}`$:

|     |                                        |                                                                                                                     |     |
| --- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | --- | ------------------------------------------------- | --- |
|     | $`\displaystyle\max_{\pi_{\theta}}\ `$ | $`\displaystyle\mathbb{E}_{x\sim\mathcal{D},y\sim\pi_{\theta}(\cdot\mid x;\mathcal{R})}\left[r_{\phi}(x,y)\right]`$ |     |
|     |                                        | $`\displaystyle-\beta\mathbb{D}_{\text{KL}}\left[\pi_{\theta}(y\mid x;\mathcal{R})\,                                |     | \,\pi\_{\text{ref}}(y\mid x;\mathcal{R})\right]`$ |     |

In the first term, the LLM generates tokens, $`y`$ from the question
$`x`$, conditioned on a retriever: $`y\sim\pi_{\theta}(\cdot\|R)`$. The
reward model, $`r_{\phi}(x,y)`$, extracts the answer from the sequence
and compares against ground truth. In the second term, the policy LLM,
$`\pi_{\theta}`$, is discouraged from diverging too far from a reference
LLM $`\pi_{\theta}`$, which is the LLM’s initial state. We use Group
Relative Policy Optimization (GRPO) to optimize the LLM based on the
samples; further details in
[Appendix I](#A9 "Appendix I Training RLVR details ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").

## 4 Results

To demonstrate the utility of our dataset, corpus, and benchmarks, we
train the LLM with reinforcement learning with verifiable rewards
(RLVR). Our experiments show that RLVR training improves performance on
scientific paper question-answering evaluations. We also provide further
quantitative and qualitative analysis.

### 4.1 Experiment details

#### Baseline methods

We build our dataset to facilitate training with RLVR, which supervises
only the final answer and thus, promises stronger generalization
compared to methods with heavy scaffolding or with reasoning SFT [Guo et
al. (2025)](#bib.bib18); [Chu et al. (2025)](#bib.bib24). To validate
this strategy, we compare RLVR training to baseline LLM training
approaches that impose few assumptions: direct LLM inference,
chain-of-thought prompting [Wei et al. (2022)](#bib.bib46); [Kojima et
al. (2022)](#bib.bib47), retrieval augmented generation (RAG) [Lewis et
al. (2020)](#bib.bib48), Search-o1 [Li et al. (2025b)](#bib.bib27), and
PaperQA2 with the same retriever as other methods [Skarlinski et al.
(2024)](#bib.bib14). For a fair comparison, we apply the same base LLM
that was used in agent training.

#### Search-R1 RLVR Training

We follow the Search-R1 training setup [Jin et al. (2025)](#bib.bib1) as
described in
[section 3.4](#S3.SS4 "3.4 Training Algorithms ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
and experiment with two retrievers: bm25 and e5. we use eight a100s
(80gb) for training, using grpo for 150 steps (runtime: ca. 30 hrs). we
have batch size 512 and minibatch size 256 for two gradient updates per
batch (full configuration is in the code). the training framework is
_verl_ [Sheng et al. (2024)](#bib.bib45). the base llms are qwen2.5 3b
and 7b, and we experiment with both _base_ and _instruct_ [Team
(2024)](#bib.bib43); [Yang et al. (2024)](#bib.bib44).

#### Evaluation

We evaluate with the test set of PaperSearchQA, and the BioASQ-factoid
benchmark [Krithara et al. (2023)](#bib.bib17) version that we release
([Section 3.2](#S3.SS2 "3.2 Evaluation dataset: BioASQ ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR")).
The evaluation metric is the same as the RL training reward term: the
prediction must exactly match one of the ground-truth answers, which are
all synonyms (for example target answer ‘APOC3’ has synonyms
‘apolipoprotein C-III’, ‘apoC-III’, ‘apoCIII’, ‘apolipoprotein C-III’,
‘apolipoprotein C3’, among others). The matching function includes
‘normalization’: conversion to lower case, stripping leading and
trailing whitespace, removing articles like ‘a’ and ‘the’.

|                     |               |        |
| ------------------- | ------------- | ------ |
|                     | PaperSearchQA | BioASQ |
| Qwen2.5-3b-Instruct |               |        |
| Direct              | 16.7          | 15.8   |
| CoT                 | 20.3          | 16.5   |
| RAG                 | 32.0          | 30.0   |
| Search-o1           | 30.8          | 29.4   |
| PaperQA2            | 32.4          | 33.1   |
| SearchR1            | 41.6          | 35.5   |
| Qwen2.5-7b-Instruct |               |        |
| Direct              | 27.5          | 24.9   |
| CoT                 | 29.7          | 23.4   |
| RAG                 | 36.5          | 29.7   |
| Search-o1           | 36.5          | 31.5   |
| PaperQA2            | 37.1          | 32.8   |
| SearchR1            | 51.0          | 44.8   |

Table 1: Main results of baselines vs Search-R1 training [Jin et al.
(2025)](#bib.bib1) that uses RLVR. The metric is accuracy, where
‘correct’ is exact match of prediction to target (or a synonym for the
target). PaperSearchQA is the test set of our dataset, while BioASQ is a
human-created evaluation. The RAG and Search-R1 systems used BM25
retrieval, and we compare to e5 retriever in the text.

![Refer to caption](2601.18207v1/x4.png)

Figure 4: Three interesting behaviours that we observe in search agent
traces. We bold some words for emphasis. Since traces are long, we
abbreviate them, as indicated by ‘\[…\]’. These are discussed further in
[Section 4.3](#S4.SS3 "4.3 Qualitative Results ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").

### 4.2 Quantitative Results

The main results are in
[Table 1](#S4.T1 "In Evaluation ‣ 4.1 Experiment details ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
showing accuracy on the target benchmarks, and where the base model was
_Qwen-instruct_ (both, 3B and 7B variants). Training with RLVR
(specifically using Search-R1 [Jin et al. (2025)](#bib.bib1)) clearly
leads to the strongest results. For the 3B LLMs, RL improves over RAG by
9.6 and 5.5 points for PaperSearchQA and BioASQ respectively. For 7B
models, the difference is 14.5 and 9.3. RAG outperforms the
retrieval-free methods by 17 points on average. Chain-of-thought
prompting outperforms direct inference by only 1.2 points on average.

[Table 2](#A7.T2 "In Appendix G Results: per-category performance ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR")
shows per-category results for all models. The easiest overall
categories are ‘Biomarkers & diagnostics’ and ‘Protein function &
signalling’, while ‘Genetic mutations’ is the most challenging.

We perform further quantitative analysis and share these additional
findings:

Semantic retrieval gives little benefit over syntactic retrieval For
both RAG and RL training, we experimented with the BM25 syntactic
retriever, and the e5 semantic retriever. While the semantic retriever
should help search where exact keywords differ, the performance benefit
was minor – within 2 points in all experiments. One possibility is that,
even when paraphrasing questions, it must include certain technical
keywords, which makes retrieval easier. Another possibility is that the
e5 retriever under-performs for scientific domains (which involve highly
technical terminology), thus removing the benefit of semantic retrieval.

LLMs encode scientific knowledge The retrieval-free baseline scores
(from
[Table 1](#S4.T1 "In Evaluation ‣ 4.1 Experiment details ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"))
are reasonably high, and scale with model size. For example on
PaperSearchQA they score 20.3 and 29.7 for 3B and 7B models. This is
probably explained by the fact that PubMed abstracts are easy to
download, and so they likely appear in pretraining mixtures. Despite
this data (probably) being seen by the model, memorization is far from
perfect, so retrieval remains necessary.

Superior performance with model size is likely due to knowledge Averaged
across benchmarks, Search-R1 outperforms CoT by 20.2 points for the 3B
model and 21.4 for the 7B model. This suggests that the performance gain
is due to improved parametric knowledge, and not due to superior
capabilities in query formulation or comprehension.

Paraphrasing in data construction is beneficial In dataset construction,
we observed that LLM-generated questions would often mirror keywords or
phrasing from the source document in
[Section 3.1](#S3.SS1 "3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
and so we added a paraphrasing step to 50% of the QAs, allowing to
compare non-paraphrases and paraphrased QAs. For SearchR1 trained on
PaperSearchQA, non-paraphrased questions scored 57.2 while paraphrased
questions scored 44.9, highlighting the importance of paraphrasing for
sustaining question difficulty.

Training dynamics are similar to general-domain QA training environments
The Search-R1 study [Jin et al. (2025)](#bib.bib1) observed certain
dynamics that we also observe. Specifically, we observed small
performance difference between base and instruct models, albeit the base
model required more training time to converge. We also found that
training with GRPO was unstable, and reward would collapse to zero for
some training runs – the base (non-instruct) models were generally more
stable.

### 4.3 Qualitative Results

To better understand the system performance, we manually reviewed the
reasoning traces for models at multiple stages in training. We highlight
three prevalent patterns in
[Figure 4](#S4.F4 "In Evaluation ‣ 4.1 Experiment details ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
The format of the traces includes reasoning inside ‘\<think\>’ tokens
and the final answer is in ‘\<answer\>’ tokens. To perform retrieval,
the LLM outputs text in ‘\<search\>‘ tags; the retrieved documents are
dumped into the trace inside ‘\<information\>‘.

Behavior 1 – explicit planning and keyword extraction. We find this
pattern to be very common in later training. The model follows a clear
and simple strategy common in RAG with rewriting: extracting the
keywords for search and then combining them into a search query. After
performing search, the LLM summarizes the final conclusion.

Behavior 2 – reasoning before search. Here, the LLM reasons about the
question using only its parametric knowledge before performing any
search. In the example problem, it observes that disease symptoms vary
based on stage, and suggests symptoms from its own parametric knowledge.
The trace acknowledges that it does not have the answer, and performs
search. After viewing the retrieved information, the presence of earlier
reasoning tokens may impact the final answer.

Behavior 3 – verification of in-parameter knowledge. The LLMs have
sufficient knowledge to answer between 15% and 30% of questions
([Table 1](#S4.T1 "In Evaluation ‣ 4.1 Experiment details ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR")),
so how does the agent behave when it already knows the answer? We find
that it performs search anyway, but in the reasoning trace it will state
its initial answer, and explicitly declare that it is doing further
verification. Verification is generally good, since the LLM can gather
more evidence for a reliable answer. More sophisticated systems however
should only search when not confident in its initial answer.

Agent behavior becomes less varied with more training With more
training, behavior 1 becomes much more common. We suspect this is due to
lack of training data diversity – PaperSearchQAonly includes factoid-QA,
and so this learned strategy is effective for most samples. Future
systems trained on more QA types and elicit more varied behavior.

Very little reasoning after viewing documents After adding retrieved
documents, the LLM tends to answer immediately, without explicit
reasoning about document contents. This could be explained by
comprehension being simpler with factoid-QA; it is also possible that RL
training led to better comprehension due to parameter weight updates.

## 5 Discussion

We show that search agents can be trained using RL to perform
question-answering by reasoning and gathering knowledge from scientific
papers, a crucial intellectual part of science [Tsatsaronis et al.
(2015)](#bib.bib30); [Hope et al. (2023)](#bib.bib51). Search agents –
and more generally RL-trained tool-use agents – are rapidly advancing in
general-domain AI. Our aim in designing the training datasets,
benchmarks, and corpus was to ensure compatibility with these methods.
We hope that advances to general-domain agents – both in open research
and in private labs – will translate to stronger capabilities in
scientific literature understanding by leveraging our artifacts and
others from the AI for science community.

While our datasets represent progress for scientific search agents, the
scope is limited to only single-hop factoid-QA and simple retrieval over
a database of abstracts – there is huge potential for further work.
Interesting directions include factoid-QA designed to be multihop [Kim
et al. (2025)](#bib.bib57), answers with list-of-entities, and questions
requiring extended answers or summaries [Krithara et al.
(2023)](#bib.bib17); [Asai et al. (2024)](#bib.bib16); these can require
more complex agent planning behavior and fuzzy reward models. Even more
ambitiously, future work could aim to resolve questions with conflicting
evidence, like in critical literature review, [Lieberum et al.
(2025)](#bib.bib53); [Polzak et al. (2025)](#bib.bib52); [Clark et al.
(2025)](#bib.bib54). Moreover, future datasets should consider that
recent results in RLVR for (non-tool-use) LLMs are leveraging
LLM-as-a-judge for reward modeling [Su et al. (2025)](#bib.bib55);
[Gunjal et al. (2025)](#bib.bib56). Meanwhile, other tool-use and search
agent works consider text and images, which is relevant to scientific
papers as well [Wu et al. (2025)](#bib.bib4); [Wang et al.
(2025)](#bib.bib58).

Other research directions are more specific to literature understanding
applications. Agents could be equipped with tools and metadata that
would be used by real scientists in their work, for example citation
traversal and source reliability metrics. For example, one could
implement a scoring on to what extent the conclusions extracted from a
scientific article are supported by the figure images / data presented
in the article – an assessment that is typically made by scientists when
they deeply review literature. This could aid in valuing contradicting
or diverging scientific results for a reply. Such metrics could be
provided in the output, which could contain multiple answers with
scores.

On a final note, our data generation pipeline is quite general – it
could be adapted to generate QA datasets in other domains like
chemistry, materials science, and computer science.

## 6 Conclusion

AI holds great potential to transform science. One exciting cluster of
methods are LLM agents or multi-agent systems – sometimes called AI
Scientists [Gao et al. (2024)](#bib.bib59); [Lu et al.
(2024)](#bib.bib9); [Gottweis et al. (2025)](#bib.bib60); [Huang et al.
(2025)](#bib.bib11); [Hope et al. (2023)](#bib.bib51). This research
program anticipates agents becoming more and more autonomous – first by
performing well-defined tasks like data analysis and experimental
execution (e.g., [Huang et al. (2025)](#bib.bib11)) – and later
performing more open-ended tasks [Hughes et al. (2024)](#bib.bib61) like
planning new experiments and even forming new hypotheses. But scientific
fields are deeply knowledge-intensive: scientific discovery requires
recalling, retrieving, and evaluating arcane information in the massive
corpus of human knowledge. We therefore claim that future AI Scientist
systems will require the capability of knowledge intensive search.
Literature understanding is therefore fundamental to AI systems in
science, and we believe that RL training of search agents – like in this
paper – is an essential approach.

## 7 Limitations

First, the data generation pipeline is automatic and uses LLMs, which
could lead to factually incorrect QAs. One source of risk is LLM
hallucination, though the risk is small since each prompt has a smaller
context, and we use strong LLMs (GPT-4.1).

Another risk from our data generation pipeline is that it is challenging
to infer a ‘general QA’ from a single specific abstract. For example, an
abstract might claim “mutation in gene X correlates with disease Y”, and
our pipeline might derive the question “what gene mutation is correlated
with disease A?”. But since we only have one abstract in context, we
cannot be sure that ‘gene mutation X’ is the only answer – some other
abstract might report that ‘gene mutation Y’ also correlates with the
disease. In designing our data generation pipeline, expert review found
such cases to be rare, and so we did not design complex mitigations. (It
is possible that some such questions were avoided due to the parametric
knowledge in the LLM generating the questions – GPT-4.1 – which is a
more capable model than the smaller models used in these experiments).
Future work that follow our data generation methodology could apply
mitigations if needed. For example, if human review finds the issue
prevalent for certain question categories, then that category could be
excluded. Or, a workflow could be designed to retrieve all relevant
papers to check for conflicts (which would be allowed a large retrieval
budget).

In terms of scope, this is a first study in using RLVR to train search
agents, so we restricted it to factoid QA. While this is a similar
restriction to other early search agent papers, it represents only one
of the possible question types important for real applications – we
discuss future directions in
[Section 5](#S5 "5 Discussion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
Likewise, our dataset covers scientific papers in biology & medicine,
but not other domains commonly studied in AI for science like chemistry,
materials science, computer science. However most AI for science papers
have a similar limitation because significant domain expertise is
required, making highly general studies challenging [Mirza et al.
(2025)](#bib.bib49); [Burgess et al. (2025)](#bib.bib10); [Tang et al.
(2025)](#bib.bib50).

Still on scope, another limitation is that we tackle text-only problems,
however it is scientific reasoning obviously goes beyond text, for
example to consider images [Yue et al. (2024)](#bib.bib65); [Burgess et
al. (2025)](#bib.bib10) and more general data types [Huang et al.
(2025)](#bib.bib11). Future work could use data sources like BIOMEDICA
[Lozano et al. (2025)](#bib.bib66) that include paper figures for PubMed
open-access articles.

This synthetic data generation procedure requires access to research
articles, which are often protected by copyright; the field should
consider approaches similar to [Schuhmann et al. (2025)](#bib.bib67) to
overcome this. Future systems could also better leverage
science-specific retrieval systems [Li et al. (2025a)](#bib.bib68);
[Asai et al. (2024)](#bib.bib16).

While the study provides resources towards building useful search agents
for scientific practitioners, the derived agent system is a research
prototype and is not suitable for real-world use. Apart from having a
too-restricted scope, it has not undergone thorough evaluation needed
for real-world deployment.

## References

- Achiam et al. (2023) J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I.
  Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S.
  Anadkat, et al. Gpt-4 technical report. arXiv preprint
  arXiv:2303.08774. Cited by:
  [§3.1](#S3.SS1.SSS0.Px3.p2.1 "Automatic QA Generation Pipeline ‣ 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Anthropic (2025) Anthropic System card: claude opus 4 & claude sonnet 4. Technical report Anthropic. Note: PDF, May 2025, “System card
  introduces Claude Opus 4 and Claude Sonnet 4” External Links:
  [Link](https://www-cdn.anthropic.com/4263b940cabb546aa0e3283f35b686f4f3b2ff47.pdf)
  Cited by:
  [§3.1](#S3.SS1.SSS0.Px2.p2.1 "Categories for Question-Answering ‣ 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Asai et al. (2024) A. Asai, J. He, R. Shao, W. Shi, A. Singh, J. C.
  Chang, K. Lo, L. Soldaini, S. Feldman, M. D’arcy, et al. Openscholar:
  synthesizing scientific literature with retrieval-augmented lms. arXiv
  preprint arXiv:2411.14199. Cited by:
  [§1](#S1.p3.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§2.2](#S2.SS2.p1.1 "2.2 Search Agents for Scientific QA ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§5](#S5.p2.1 "5 Discussion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§7](#S7.p5.1 "7 Limitations ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Burgess et al. (2025) J. Burgess, J. J. Nirschl, L. Bravo-Sánchez, A.
  Lozano, S. R. Gupte, J. G. Galaz-Montoya, Y. Zhang, Y. Su, D.
  Bhowmik, Z. Coman, et al. Microvqa: a multimodal reasoning benchmark
  for microscopy-based scientific research. In Proceedings of the
  Computer Vision and Pattern Recognition Conference, pp. 19552–19564.
  Cited by:
  [§7](#S7.p3.1 "7 Limitations ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§7](#S7.p4.1 "7 Limitations ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Chen et al. (2025) Q. Chen, L. Qin, J. Liu, D. Peng, J. Guan, P.
  Wang, M. Hu, Y. Zhou, T. Gao, and W. Che Towards reasoning era: a
  survey of long chain-of-thought for reasoning large language models.
  arXiv preprint arXiv:2503.09567. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Chu et al. (2025) T. Chu, Y. Zhai, J. Yang, S. Tong, S. Xie, D.
  Schuurmans, Q. V. Le, S. Levine, and Y. Ma Sft memorizes, rl
  generalizes: a comparative study of foundation model post-training.
  arXiv preprint arXiv:2501.17161. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§2.1](#S2.SS1.p1.1 "2.1 Search agents ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§2.2](#S2.SS2.p1.1 "2.2 Search Agents for Scientific QA ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§3.4](#S3.SS4.SSS0.Px1.p2.1 "RLVR for Search Agents ‣ 3.4 Training Algorithms ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Baseline methods ‣ 4.1 Experiment details ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Clark et al. (2025) J. Clark, B. Barton, L. Albarqouni, O.
  Byambasuren, T. Jowsey, J. Keogh, T. Liang, C. Moro, H. O’Neill,
  and M. Jones Generative artificial intelligence use in evidence
  synthesis: a systematic review. Research Synthesis Methods, pp. 1–19.
  Cited by:
  [§5](#S5.p2.1 "5 Discussion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Delile et al. (2024) J. Delile, S. Mukherjee, A. Van Pamel, and L.
  Zhukov Graph-based retriever captures the long tail of biomedical
  knowledge. arXiv preprint arXiv:2402.12352. Cited by:
  [§1](#S1.p3.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Feng et al. (2025) J. Feng, S. Huang, X. Qu, G. Zhang, Y. Qin, B.
  Zhong, C. Jiang, J. Chi, and W. Zhong Retool: reinforcement learning
  for strategic tool use in llms. arXiv preprint arXiv:2504.11536. Cited
  by:
  [§1](#S1.p1.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§2.1](#S2.SS1.p1.1 "2.1 Search agents ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Ferguson et al. (2014) A. R. Ferguson, J. L. Nielson, M. H.
  Cragin, A. E. Bandrowski, and M. E. Martone Big data from small data:
  data-sharing in the’long tail’of neuroscience. Nature neuroscience 17
  (11), pp. 1442–1447. Cited by:
  [§1](#S1.p3.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Gao et al. (2024) S. Gao, A. Fang, Y. Huang, V. Giunchiglia, A.
  Noori, J. R. Schwarz, Y. Ektefaie, J. Kondic, and M. Zitnik Empowering
  biomedical discovery with ai agents. Cell 187 (22), pp. 6125–6151.
  Cited by:
  [§1](#S1.p3.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§6](#S6.p1.1 "6 Conclusion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Gottweis et al. (2025) J. Gottweis, W. Weng, A. Daryin, T. Tu, A.
  Palepu, P. Sirkovic, A. Myaskovsky, F. Weissenberger, K. Rong, R.
  Tanno, et al. Towards an ai co-scientist. arXiv preprint
  arXiv:2502.18864. Cited by:
  [§6](#S6.p1.1 "6 Conclusion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Guan et al. (2025) X. Guan, J. Zeng, F. Meng, C. Xin, Y. Lu, H.
  Lin, X. Han, L. Sun, and J. Zhou DeepRAG: thinking to retrieve step by
  step for large language models. arXiv preprint arXiv:2502.01142. Cited
  by:
  [§2.1](#S2.SS1.p1.1 "2.1 Search agents ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Gunjal et al. (2025) A. Gunjal, A. Wang, E. Lau, V. Nath, B. Liu,
  and S. Hendryx Rubrics as rewards: reinforcement learning beyond
  verifiable domains. arXiv preprint arXiv:2507.17746. Cited by:
  [§5](#S5.p2.1 "5 Discussion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Guo et al. (2025) D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R.
  Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. Deepseek-r1: incentivizing
  reasoning capability in llms via reinforcement learning. arXiv
  preprint arXiv:2501.12948. Cited by: [Appendix
  I](#A9.p1.1 "Appendix I Training RLVR details ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§1](#S1.p1.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§2.1](#S2.SS1.p1.1 "2.1 Search agents ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Baseline methods ‣ 4.1 Experiment details ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Ho et al. (2020) X. Ho, A. D. Nguyen, S. Sugawara, and A. Aizawa
  Constructing a multi-hop qa dataset for comprehensive evaluation of
  reasoning steps. arXiv preprint arXiv:2011.01060. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Hope et al. (2023) T. Hope, D. Downey, D. S. Weld, O. Etzioni, and E.
  Horvitz A computational inflection for scientific discovery.
  Communications of the ACM 66 (8), pp. 62–73. Cited by:
  [§1](#S1.p3.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§5](#S5.p1.1 "5 Discussion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§6](#S6.p1.1 "6 Conclusion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Huang et al. (2025) K. Huang, S. Zhang, H. Wang, Y. Qu, Y. Lu, Y.
  Roohani, R. Li, L. Qiu, J. Zhang, Y. Di, et al. Biomni: a
  general-purpose biomedical ai agent. bioRxiv, pp. 2025–05. Cited by:
  [§6](#S6.p1.1 "6 Conclusion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§7](#S7.p4.1 "7 Limitations ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Hughes et al. (2024) E. Hughes, M. Dennis, J. Parker-Holder, F.
  Behbahani, A. Mavalankar, Y. Shi, T. Schaul, and T. Rocktaschel
  Open-endedness is essential for artificial superhuman intelligence.
  arXiv preprint arXiv:2406.04268. Cited by:
  [§6](#S6.p1.1 "6 Conclusion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Jaech et al. (2024) A. Jaech, A. Kalai, A. Lerer, A. Richardson, A.
  El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney, et al.
  Openai o1 system card. arXiv preprint arXiv:2412.16720. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Jin et al. (2025) B. Jin, H. Zeng, Z. Yue, J. Yoon, S. Arik, D.
  Wang, H. Zamani, and J. Han Search-r1: training llms to reason and
  leverage search engines with reinforcement learning. arXiv preprint
  arXiv:2503.09516. Cited by: [Table
  2](#A7.T2 "In Appendix G Results: per-category performance ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§1](#S1.p2.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§1](#S1.p5.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§2.1](#S2.SS1.p1.1 "2.1 Search agents ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§2.2](#S2.SS2.p1.1 "2.2 Search Agents for Scientific QA ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§3.1](#S3.SS1.SSS0.Px1.p1.1 "Defining Dataset Properties ‣ 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§3.4](#S3.SS4.SSS0.Px1.p1.1 "RLVR for Search Agents ‣ 3.4 Training Algorithms ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§3.4](#S3.SS4.SSS0.Px1.p3.1 "RLVR for Search Agents ‣ 3.4 Training Algorithms ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Search-R1 RLVR Training ‣ 4.1 Experiment details ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§4.2](#S4.SS2.p1.1 "4.2 Quantitative Results ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§4.2](#S4.SS2.p8.1 "4.2 Quantitative Results ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [Table
  1](#S4.T1 "In Evaluation ‣ 4.1 Experiment details ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Jin et al. (2019) Q. Jin, B. Dhingra, Z. Liu, W. W. Cohen, and X. Lu
  Pubmedqa: a dataset for biomedical research question answering. arXiv
  preprint arXiv:1909.06146. Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Search Agents for Scientific QA ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Joshi et al. (2017) M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer
  Triviaqa: a large scale distantly supervised challenge dataset for
  reading comprehension. arXiv preprint arXiv:1705.03551. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§3.1](#S3.SS1.SSS0.Px1.p1.1 "Defining Dataset Properties ‣ 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Kim et al. (2025) Y. Kim, Y. Abdulle, and H. Wu BioHopR: a benchmark
  for multi-hop, multi-answer reasoning in biomedical domain. arXiv
  preprint arXiv:2505.22240. Cited by:
  [§5](#S5.p2.1 "5 Discussion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Kojima et al. (2022) T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y.
  Iwasawa Large language models are zero-shot reasoners. Advances in
  neural information processing systems 35, pp. 22199–22213. Cited by:
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Baseline methods ‣ 4.1 Experiment details ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Krithara et al. (2023) A. Krithara, A. Nentidis, K. Bougiatiotis,
  and G. Paliouras BioASQ-qa: a manually curated corpus for biomedical
  question answering. Scientific Data 10 (1), pp. 170. Cited by:
  [Appendix
  A](#A1.SS0.SSS0.Px3.p1.1 "Licenses ‣ Appendix A Dataset and Code Availability ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§1](#S1.p4.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§2.2](#S2.SS2.p1.1 "2.2 Search Agents for Scientific QA ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [Figure
  3](#S3.F3 "In Categories for Question-Answering ‣ 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§3.1](#S3.SS1.SSS0.Px1.p2.1 "Defining Dataset Properties ‣ 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§3.2](#S3.SS2.p1.1 "3.2 Evaluation dataset: BioASQ ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§3.3](#S3.SS3.p1.1 "3.3 Retrieval Corpus and Index ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§4.1](#S4.SS1.SSS0.Px3.p1.1 "Evaluation ‣ 4.1 Experiment details ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§5](#S5.p2.1 "5 Discussion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Kwiatkowski et al. (2019) T. Kwiatkowski, J. Palomaki, O. Redfield, M.
  Collins, A. Parikh, C. Alberti, D. Epstein, I. Polosukhin, J.
  Devlin, K. Lee, et al. Natural questions: a benchmark for question
  answering research. Transactions of the Association for Computational
  Linguistics 7, pp. 453–466. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§3.1](#S3.SS1.SSS0.Px1.p1.1 "Defining Dataset Properties ‣ 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Lála et al. (2023) J. Lála, O. O’Donoghue, A. Shtedritski, S.
  Cox, S. G. Rodriques, and A. D. White Paperqa: retrieval-augmented
  generative agent for scientific research. arXiv preprint
  arXiv:2312.07559. Cited by:
  [§1](#S1.p3.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§2.2](#S2.SS2.p1.1 "2.2 Search Agents for Scientific QA ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Lambert et al. (2024) N. Lambert, J. Morrison, V. Pyatkin, S.
  Huang, H. Ivison, F. Brahman, L. J. V. Miranda, A. Liu, N. Dziri, S.
  Lyu, et al. T$`\backslash`$" ulu 3: pushing frontiers in open language
  model post-training. arXiv preprint arXiv:2411.15124. Cited by:
  [Appendix
  D](#A4.p1.1 "Appendix D Further explanation of reinforcement learning with verifiable rewards (RLVR) ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§1](#S1.p1.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Lee et al. (2023) Y. Lee, K. Lee, S. Park, D. Hwang, J. Kim, H. Lee,
  and M. Lee Qasa: advanced question answering on scientific articles.
  In International Conference on Machine Learning, pp. 19036–19052.
  Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Search Agents for Scientific QA ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Lewis et al. (2020) P. Lewis, E. Perez, A. Piktus, F. Petroni, V.
  Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, et
  al. Retrieval-augmented generation for knowledge-intensive nlp tasks.
  Advances in neural information processing systems 33, pp. 9459–9474.
  Cited by:
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Baseline methods ‣ 4.1 Experiment details ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Li et al. (2025a) L. Li, X. Zhou, and Z. Liu R2MED: a benchmark for
  reasoning-driven medical retrieval. arXiv preprint arXiv:2505.14558.
  Cited by:
  [§7](#S7.p5.1 "7 Limitations ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Li et al. (2025b) X. Li, G. Dong, J. Jin, Y. Zhang, Y. Zhou, Y.
  Zhu, P. Zhang, and Z. Dou Search-o1: agentic search-enhanced large
  reasoning models. arXiv preprint arXiv:2501.05366. Cited by:
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Baseline methods ‣ 4.1 Experiment details ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Li et al. (2025c) X. Li, J. Jin, G. Dong, H. Qian, Y. Zhu, Y. Wu, J.
  Wen, and Z. Dou Webthinker: empowering large reasoning models with
  deep research capability. arXiv preprint arXiv:2504.21776. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 Search agents ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Lieberum et al. (2025) J. Lieberum, M. Toews, M. Metzendorf, F.
  Heilmeyer, W. Siemens, C. Haverkamp, D. Böhringer, J. J. Meerpohl,
  and A. Eisele-Metzger Large language models for conducting systematic
  reviews: on the rise, but not yet ready for use—a scoping review.
  Journal of Clinical Epidemiology 181, pp. 111746. Cited by:
  [§5](#S5.p2.1 "5 Discussion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Lozano et al. (2025) A. Lozano, M. W. Sun, J. Burgess, L. Chen, J. J.
  Nirschl, J. Gu, I. Lopez, J. Aklilu, A. Rau, A. W. Katzer, et al.
  Biomedica: an open biomedical image-caption archive, dataset, and
  vision-language models derived from scientific literature. In
  Proceedings of the Computer Vision and Pattern Recognition Conference,
  pp. 19724–19735. Cited by:
  [§7](#S7.p4.1 "7 Limitations ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Lu et al. (2024) C. Lu, C. Lu, R. T. Lange, J. Foerster, J. Clune,
  and D. Ha The ai scientist: towards fully automated open-ended
  scientific discovery. arXiv preprint arXiv:2408.06292. Cited by:
  [§1](#S1.p3.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§6](#S6.p1.1 "6 Conclusion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Mirza et al. (2025) A. Mirza, N. Alampara, S. Kunchapu, M.
  Ríos-García, B. Emoekabu, A. Krishnan, T. Gupta, M.
  Schilling-Wilhelmi, M. Okereke, A. Aneesh, et al. A framework for
  evaluating the chemical knowledge and reasoning abilities of large
  language models against the expertise of chemists. Nature Chemistry,
  pp. 1–8. Cited by:
  [§7](#S7.p3.1 "7 Limitations ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Nentidis et al. (2023) A. Nentidis, G. Katsimpras, A. Krithara, S.
  Lima López, E. Farré-Maduell, L. Gasco, M. Krallinger, and G.
  Paliouras Overview of bioasq 2023: the eleventh bioasq challenge on
  large-scale biomedical semantic indexing and question answering. In
  International Conference of the Cross-Language Evaluation Forum for
  European Languages, pp. 227–250. Cited by:
  [§3.1](#S3.SS1.SSS0.Px1.p2.1 "Defining Dataset Properties ‣ 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- OpenAI (2025a) OpenAI Introducing deep research. Note:
  [https://openai.com/index/introducing-deep-research/](https://openai.com/index/introducing-deep-research/)Accessed
  2025-07-27 Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 Search agents ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- OpenAI (2025b) OpenAI Introducing openai o3 and o4‑mini. Note:
  [https://openai.com/index/introducing-o3-and-o4-mini/](https://openai.com/index/introducing-o3-and-o4-mini/)Accessed 2025-07-27
  Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 Search agents ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§3.1](#S3.SS1.SSS0.Px2.p2.1 "Categories for Question-Answering ‣ 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Opsahl-Ong et al. (2024) K. Opsahl-Ong, M. J. Ryan, J. Purtell, D.
  Broman, C. Potts, M. Zaharia, and O. Khattab Optimizing instructions
  and demonstrations for multi-stage language model programs. arXiv
  preprint arXiv:2406.11695. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 Search agents ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Polzak et al. (2025) C. Polzak, A. Lozano, M. W. Sun, J. Burgess, Y.
  Zhang, K. Wu, and S. Yeung-Levy Can large language models match the
  conclusions of systematic reviews?. arXiv preprint arXiv:2505.22787.
  Cited by:
  [§5](#S5.p2.1 "5 Discussion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Qian et al. (2025) C. Qian, E. C. Acikgoz, Q. He, H. Wang, X. Chen, D.
  Hakkani-Tür, G. Tur, and H. Ji Toolrl: reward is all tool learning
  needs. arXiv preprint arXiv:2504.13958. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 Search agents ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Robertson and Walker (1994) S. E. Robertson and S. Walker Some simple
  effective approximations to the 2-poisson model for probabilistic
  weighted retrieval. In SIGIR’94: Proceedings of the Seventeenth Annual
  International ACM-SIGIR Conference on Research and Development in
  Information Retrieval, organised by Dublin City University,
  pp. 232–241. Cited by:
  [§3.3](#S3.SS3.p2.1 "3.3 Retrieval Corpus and Index ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Schick et al. (2023) T. Schick, J. Dwivedi-Yu, R. Dessì, R.
  Raileanu, M. Lomeli, E. Hambro, L. Zettlemoyer, N. Cancedda, and T.
  Scialom Toolformer: language models can teach themselves to use tools.
  Advances in Neural Information Processing Systems 36, pp. 68539–68551.
  Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 Search agents ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Schuhmann et al. (2025) C. Schuhmann, G. Rabby, A. Prabhu, T.
  Ahmed, A. Hochlehnert, H. Nguyen, N. Akinci, L. Schmidt, R.
  Kaczmarczyk, S. Auer, et al. Project alexandria: towards freeing
  scientific knowledge from copyright burdens via llms. arXiv preprint
  arXiv:2502.19413. Cited by:
  [§7](#S7.p5.1 "7 Limitations ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Schulhoff et al. (2024) S. Schulhoff, M. Ilie, N. Balepur, K.
  Kahadze, A. Liu, C. Si, Y. Li, A. Gupta, H. Han, S. Schulhoff, et al.
  The prompt report: a systematic survey of prompt engineering
  techniques. arXiv preprint arXiv:2406.06608. Cited by:
  [§3.1](#S3.SS1.SSS0.Px3.p1.1 "Automatic QA Generation Pipeline ‣ 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Shao et al. (2024) Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H.
  Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: pushing the limits
  of mathematical reasoning in open language models. arXiv preprint
  arXiv:2402.03300. Cited by: [Appendix
  I](#A9.p1.1 "Appendix I Training RLVR details ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§1](#S1.p1.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Sheng et al. (2024) G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R.
  Zhang, Y. Peng, H. Lin, and C. Wu HybridFlow: a flexible and efficient
  rlhf framework. arXiv preprint arXiv: 2409.19256. Cited by:
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Search-R1 RLVR Training ‣ 4.1 Experiment details ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Skarlinski et al. (2024) M. D. Skarlinski, S. Cox, J. M.
  Laurent, J. D. Braza, M. Hinks, M. J. Hammerling, M. Ponnapati, S. G.
  Rodriques, and A. D. White Language agents achieve superhuman
  synthesis of scientific knowledge. arXiv preprint arXiv:2409.13740.
  Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Search Agents for Scientific QA ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Baseline methods ‣ 4.1 Experiment details ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Song et al. (2025) H. Song, J. Jiang, Y. Min, J. Chen, Z. Chen, W. X.
  Zhao, L. Fang, and J. Wen R1-searcher: incentivizing the search
  capability in llms via reinforcement learning. arXiv preprint
  arXiv:2503.05592. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§2.1](#S2.SS1.p1.1 "2.1 Search agents ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Su et al. (2025) Y. Su, D. Yu, L. Song, J. Li, H. Mi, Z. Tu, M. Zhang,
  and D. Yu Crossing the reward bridge: expanding rl with verifiable
  rewards across diverse domains. arXiv preprint arXiv:2503.23829. Cited
  by:
  [§5](#S5.p2.1 "5 Discussion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Sun et al. (2025) H. Sun, Z. Qiao, J. Guo, X. Fan, Y. Hou, Y.
  Jiang, P. Xie, Y. Zhang, F. Huang, and J. Zhou Zerosearch: incentivize
  the search capability of llms without searching. arXiv preprint
  arXiv:2505.04588. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§2.1](#S2.SS1.p1.1 "2.1 Search agents ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Tang et al. (2025) Y. Tang, W. Xu, J. Cao, W. Gao, S. Farrell, B.
  Erichson, M. W. Mahoney, A. Nonaka, and Z. Yao Matterchat: a
  multi-modal llm for material science. arXiv preprint arXiv:2502.13107.
  Cited by:
  [§7](#S7.p3.1 "7 Limitations ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Team (2024) Q. Team Qwen2.5: a party of foundation models. External
  Links: [Link](https://qwenlm.github.io/blog/qwen2.5/) Cited by:
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Search-R1 RLVR Training ‣ 4.1 Experiment details ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Trivedi et al. (2022) H. Trivedi, N. Balasubramanian, T. Khot, and A.
  Sabharwal Interleaving retrieval with chain-of-thought reasoning for
  knowledge-intensive multi-step questions. arXiv preprint
  arXiv:2212.10509. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§2.1](#S2.SS1.p1.1 "2.1 Search agents ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Tsatsaronis et al. (2015) G. Tsatsaronis, G. Balikas, P.
  Malakasiotis, I. Partalas, M. Zschunke, M. R. Alvers, D.
  Weissenborn, A. Krithara, S. Petridis, D. Polychronopoulos, et al. An
  overview of the bioasq large-scale biomedical semantic indexing and
  question answering competition. BMC bioinformatics 16 (1), pp. 138.
  Cited by: [Appendix
  A](#A1.SS0.SSS0.Px3.p1.1 "Licenses ‣ Appendix A Dataset and Code Availability ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§2.2](#S2.SS2.p1.1 "2.2 Search Agents for Scientific QA ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§3.1](#S3.SS1.SSS0.Px1.p2.1 "Defining Dataset Properties ‣ 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§3.2](#S3.SS2.p1.1 "3.2 Evaluation dataset: BioASQ ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§5](#S5.p1.1 "5 Discussion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Wadden et al. (2020) D. Wadden, S. Lin, K. Lo, L. L. Wang, M. van
  Zuylen, A. Cohan, and H. Hajishirzi Fact or fiction: verifying
  scientific claims. arXiv preprint arXiv:2004.14974. Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Search Agents for Scientific QA ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Wang et al. (2022) L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D.
  Jiang, R. Majumder, and F. Wei Text embeddings by weakly-supervised
  contrastive pre-training. arXiv preprint arXiv:2212.03533. Cited by:
  [§3.3](#S3.SS3.p2.1 "3.3 Retrieval Corpus and Index ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Wang et al. (2025) Q. Wang, R. Ding, Y. Zeng, Z. Chen, L. Chen, S.
  Wang, P. Xie, F. Huang, and F. Zhao VRAG-rl: empower
  vision-perception-based rag for visually rich information
  understanding via iterative reasoning with reinforcement learning.
  arXiv preprint arXiv:2505.22019. Cited by:
  [§5](#S5.p2.1 "5 Discussion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Wei et al. (2022) J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E.
  Chi, Q. V. Le, D. Zhou, et al. Chain-of-thought prompting elicits
  reasoning in large language models. Advances in neural information
  processing systems 35, pp. 24824–24837. Cited by:
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Baseline methods ‣ 4.1 Experiment details ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Wu et al. (2025) J. Wu, Z. Deng, W. Li, Y. Liu, B. You, B. Li, Z. Ma,
  and Z. Liu MMSearch-r1: incentivizing lmms to search. arXiv preprint
  arXiv:2506.20670. Cited by:
  [§5](#S5.p2.1 "5 Discussion ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Yang et al. (2024) A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C.
  Zhou, C. Li, C. Li, D. Liu, F. Huang, G. Dong, H. Wei, H. Lin, J.
  Tang, J. Wang, J. Yang, J. Tu, J. Zhang, J. Ma, J. Xu, J. Zhou, J.
  Bai, J. He, J. Lin, K. Dang, K. Lu, K. Chen, K. Yang, M. Li, M.
  Xue, N. Ni, P. Zhang, P. Wang, R. Peng, R. Men, R. Gao, R. Lin, S.
  Wang, S. Bai, S. Tan, T. Zhu, T. Li, T. Liu, W. Ge, X. Deng, X.
  Zhou, X. Ren, X. Zhang, X. Wei, X. Ren, Y. Fan, Y. Yao, Y. Zhang, Y.
  Wan, Y. Chu, Y. Liu, Z. Cui, Z. Zhang, and Z. Fan Qwen2 technical
  report. arXiv preprint arXiv:2407.10671. Cited by:
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Search-R1 RLVR Training ‣ 4.1 Experiment details ‣ 4 Results ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Yang et al. (2018) Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. W.
  Cohen, R. Salakhutdinov, and C. D. Manning HotpotQA: a dataset for
  diverse, explainable multi-hop question answering. arXiv preprint
  arXiv:1809.09600. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Yao et al. (2023) S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K.
  Narasimhan, and Y. Cao React: synergizing reasoning and acting in
  language models. In International Conference on Learning
  Representations (ICLR), Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§2.1](#S2.SS1.p1.1 "2.1 Search agents ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Yue et al. (2024) X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G.
  Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, et al. Mmmu: a massive
  multi-discipline multimodal understanding and reasoning benchmark for
  expert agi. In Proceedings of the IEEE/CVF Conference on Computer
  Vision and Pattern Recognition, pp. 9556–9567. Cited by:
  [§7](#S7.p4.1 "7 Limitations ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").
- Zheng et al. (2025) Y. Zheng, D. Fu, X. Hu, X. Cai, L. Ye, P. Lu,
  and P. Liu Deepresearcher: scaling deep research via reinforcement
  learning in real-world environments. arXiv preprint arXiv:2504.03160.
  Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
  [§2.1](#S2.SS1.p1.1 "2.1 Search agents ‣ 2 Related Work ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").

## Acknowledgments

We gratefully acknowledge NVIDIA’s Academic Grant Program for providing
cloud GPU resources used in this research.

## Appendix A Dataset and Code Availability

#### Accessing data

We release all artifacts on the Huggingface Hub at
[https://huggingface.co/collections/jmhb/papersearchqa](https://huggingface.co/collections/jmhb/papersearchqa).

#### Accessing Code

The code at
[https://github.com/jmhb0/PaperSearchQA](https://github.com/jmhb0/PaperSearchQA)
is for

#### Licenses

Our dataset, PaperSearchQA, is released under a fully open license
CC-BY-4.0, permitting redistribution, remixing, and commercial use. The
data is derived from PubMed abstracts that are available for bulk
download under NLM’s Terms and Conditions²² 2
[https://www.nlm.nih.gov/databases/download.html](https://www.nlm.nih.gov/databases/download.html).
The search corpus and the BioASQ evaluation set are sourced from the
BioASQ project [Krithara et al. (2023)](#bib.bib17); [Tsatsaronis et al.
(2015)](#bib.bib30), and inherit their CC-BY-2.5 license.

## Appendix B Ethical considerations

This paper advances systems that answer scientific questions from
literature, but this presents some risks:

- •
  Agents may retrieve and amplify outdated, retracted, or flawed studies
  without quality assessment mechanisms.
- •
  Papers retrieved by the agent may have some selection bias that is
  poorly understood, thus impacting papers seen by scientists.
- •
  Hallucinations in LLM outputs and incorrect QA responses may harm
  scientific practice.
- •
  Our dataset was generated in an automated pipeline, which may have
  introduced errors.

Future deployments should consider uncertainty quantification, and
source quality indicators. More broadly, the scientific community must
develop its own standards for the appropriate use of LLM tools that
consider these risks.

## Appendix C Statement on use of LLMs

LLMs were used at many points in the project. Other than what is
discussed in the main paper, we had these use cases:

- •
  In project conception: brainstorming ideas; giving feedback and
  criticism on project plans; searching related work; summarizing and
  answering questions about specific related work.
- •
  In project execution: LLMs for code generation in the Cursor IDE.
- •
  Paper writing: rephrasing individual sentences.

## Appendix D Further explanation of reinforcement learning with verifiable rewards (RLVR)

RLVR [Lambert et al. (2024)](#bib.bib20) is a post-training procedure in
which a language model is optimized only from whether its _final_ output
can be automatically verified as correct. At a high level, the model
proposes a solution to a task, a separate verifier evaluates that
solution, and the model is updated to make successful solutions more
likely in the future.

#### Single-turn RLVR.

Much of the earliest RLVR work uses a single-turn setting, where the
model answers in one shot without explicit tool calls or multiple
interaction steps. Given a query $`x`$, the model samples a final answer
$`y\sim\pi_{\theta}(\cdot\mid x)`$, such as a free-form solution to a
math problem or a code snippet. A verifier $`V`$ then returns a
(typically scalar) reward

|     |     |     |
| --- | --- | --- |
|     |

       ``` math
       r=V(x,y),
       ```        |     |

for example by exact-match against a reference answer, a numerical
tolerance check, or running unit tests on the generated code. In many
RLVR setups, $`r`$ is binary ($`r\in\{0,1\}`$) to indicate pass/fail,
but the formulation also allows graded or shaped rewards (e.g., partial
credit or the proportion of tests passed).

In this setting, the RLVR objective is

|     |     |     |
| --- | --- | --- |
|     |

````math
J(\theta)=\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_{\theta}(\cdot\mid x)}[\,r\,],
``` |  |

which says: sample questions $`x`$ from a data distribution
$`\mathcal{D}`$, sample answers $`y`$ from the model, and maximize the
expected reward returned by the verifier. This captures the basic
“generate–verify–reinforce” loop used in early RLVR for math and code.

#### Multi-step RLVR with trajectories.

For agents that call tools or take multiple reasoning steps, it is
helpful to view RLVR in a more general trajectory form. Given a query
$`x`$, the model interacts with its environment to produce a trajectory

|     |                                               |     |
|-----|-----------------------------------------------|-----|
|     |
       ``` math
       \tau=(o_{0},a_{0},o_{1},a_{1},\dots,o_{T},y),
       ```                                            |     |

where $`o_{t}`$ are observations (e.g., tool outputs or intermediate
text), $`a_{t}`$ are actions (e.g., tool calls or tokens), and $`y`$ is
the final answer returned to the user. The single-turn setting above is
a special case where there are no intermediate observations or actions
and $`\tau`$ consists only of the generated answer $`y`$.

A verifier $`V`$ now maps $`(x,\tau)`$ or $`(x,y)`$ to a scalar reward

|     |              |     |
|-----|--------------|-----|
|     |
       ``` math
       r=V(x,\tau).
       ```           |     |

The verifier can use only the final answer (e.g., exact match or unit
tests) or the whole interaction (e.g., whether a sequence of tool calls
satisfies some constraints). Let $`\pi_{\theta}(\tau\mid x)`$ denote the
model’s policy over trajectories; RLVR then maximizes

|  |  |  |
|----|----|----|
|  |
``` math
J(\theta)=\mathbb{E}_{x\sim\mathcal{D},\,\tau\sim\pi_{\theta}(\cdot\mid x)}[\,r\,].
``` |  |

When $`r`$ is binary, this reduces to maximizing the probability that
the verifier accepts the trajectory, but the same objective accommodates
more general reward shapes.

In practice, $`J(\theta)`$ is maximized using policy-gradient methods.
In our experiments we use Group Relative Policy Optimization (GRPO; see
[appendix I](#A9 "Appendix I Training RLVR details ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR")),
a variant that uses group-normalized advantages, clipping, and a KL
penalty to a reference policy. For intuition, one can view these methods
as refinements of the basic REINFORCE estimator

|  |  |  |
|----|----|----|
|  |
``` math
\nabla_{\theta}J(\theta)\approx\mathbb{E}\Big[(r-b)\sum_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}(a_{t}\mid h_{t})\Big],
``` |  |

where $`b`$ is a baseline that reduces variance. High-reward
trajectories increase the log-probabilities of their actions, while
low-reward trajectories decrease them.

#### Relation to SFT and RLHF.

RLVR differs from supervised finetuning (SFT) and RLHF in two key ways.
First, RLVR uses only verifiable success or failure of the *final*
output as a learning signal; there are no human-written labels on
intermediate steps and no preference scores over partial generations.
Second, credit assignment is purely outcome-based: all intermediate
reasoning, tool calls, and textual tokens are reinforced or discouraged
according to the reward returned by the verifier. This makes RLVR
particularly natural for tasks where correctness can be automatically
judged but good intermediate supervision is expensive or unavailable.

## Appendix E Data construction pipeline

We show the prompts here. The prompts are long, so for more readablity,
refer to the code at `data_gen/generate_questions_from_abstracts.py`

Here is the main data generation prompt mapping an abstract to QAs.

[⬇](data:text/plain;base64,QkFDS0dST1VORApZb3UgYXJlIGEgZG9tYWluLWV4cGVydCBiaW9tZWRpY2FsIE5MUCBhc3Npc3RhbnQuCllvdSBhcmUgaGVscGluZyBtZSB0byBjcmVhdGUgYW4gb3Blbi1kb21haW4gUUEgZGF0YXNldC4KVGhlIGRvd25zdHJlYW0gdGFzayB3aWxsIHJlYWQgYSBxdWVyeSBhbmQgcmVxdWlyZSBhbiBhZ2VudCB0byBzZWFyY2ggb3ZlciBQdWJtZWQgYWJzdHJhY3RzCgotLS0tLS0tLQpZT1VSIFRBU0sKSSB3aWxsIHByb3ZpZGUgeW91IHdpdGggdGl0bGUgYW5kIGFic3RyYWN0IG9mIGEgUHVibWVkIGFydGljbGUuCllvdXIgdGFzayBpcyB0byBjcmVhdGUgMyBuZXcgcXVlc3Rpb24tYW5zd2VyIHBhaXJzLgoKLS0tLS0tLS0KVFlQRVMgT0YgUVVFU1RJT05TClRoZSBxdWVzdGlvbnMgc2hvdWxkIGJlICdmYWN0b2lkIGJhc2VkJy4KVGhlIGFuc3dlciBzaG91bGQgYmUgYSBzaW1wbGUgZW50aXR5LgpJdCBzaG91bGQgbm90IGJlIGFtYmlndW91cy4KRG9uJ3QgYmUgcHJldGVudGlvdXMuCgotLS0tLS0tLQpJTVBPUlRBTlQgTk9URVMKVGhlIHF1ZXN0aW9uLWFuc3dlciBwYWlyIHdpbGwgYmUgdXNlZCB0byBldmFsdWF0aW9uIHF1ZXN0aW9uLWFuc3dlcmluZyBzeXN0ZW1zIHdpdGggcmV0cmlldmFsLiBUaHMgbWVhbnMgdGhlIHRhcmdldCBzeXN0ZW0gZG9lcyBub3Qga25vdyB3aGljaCBwYXBlciB0aGUgcXVlc3Rpb24gd2FzIHNvdXJjZWQgZnJvbS4gU28gYW4gaW5hcHByb3ByaWF0ZSBxdWVzdGlvbiB3b3VsZCBiZSAiV2hhdCB0ZWNobm9sb2d5IGlzIHVzZWQgaW4gdGhpcyBzdHVkeSB0byAuLi4iLiBvciAid2hhdCB0eXBlIG9mIHRyZWF0bWVudCBpcyBhc3Nlc3NlZCBpbiB0aGlzIHN0dWR5PyIgKHdoZXJlIHRoZSBzdHVkeSBuYW1lIGlzIG5vdCBzcGVjaWZpZmllZCkuCklmIHRoZSBxdWVzdGlvbiBjb250YWlucyBhY3JvbnltcyB0aGF0IGFyZSBub3Qgd2VsbCBrbm93biwgdGhlbiBleHBsYWluIHRoZSBhY3JvbnltLgoKLS0tLS0tLS0KRVhBTVBMRSBDQVRFR09SSUVTCkJlbG93IGFyZSBzYW1wbGUgY2F0ZWdvcmllcyB3aXRoIHNhbXBsZSBxdWVzdGlvbnMuCgpDYXRlZ29yeTogMSAtIEdlbmV0aWMgaW5oZXJpdGFuY2UgJiBkaXNlYXNlLWxpbmtlZCBtdXRhdGlvbnMKcXVlc3Rpb246IFdoYXQgZ2VuZSBpcyBtdXRhdGVkIGluIFNpY2tsZSBDZWxsIEFuZW1pYT8KYW5zd2VyOiBIQkIKcXVlc3Rpb246IFdoaWNoIHVsdHJhY29uc2VydmVkIGVsZW1lbnQgaXMgYXNzb2NpYXRlZCB3aXRoIEVtYnJ5b25pYyBTdGVtIENlbGxzIChFU0MpIHNlbGYtcmVuZXdhbD8KYW5zd2VyOiBULVVDc3RlbTEKcXVlc3Rpb246IElzIEh1bnRpbmd0b24ncyBkaXNlYXNlIGNhdXNlZCBieSBhIGRvbWluYXRlIG9yIHJlY2Vzc2l2ZSBnZW5lPwphbnN3ZXI6IGRvbWluYW50CgpDYXRlZ29yeTogMiAtIFRoZXJhcGV1dGljcywgaW5kaWNhdGlvbnMgJiBjbGluaWNhbCBldmlkZW5jZQpxdWVzdGlvbjogV2hhdCBpcyB0aGUgbW9zdCBlZmZlY3RpdmUgZHJ1ZyBmb3Igb3hhbGlwbGF0aW4taW5kdWNlZCBuZXVyb3BhdGh5PwphbnN3ZXI6IER1bG94ZXRpbmUKcXVlc3Rpb246IFdoaWNoIGNhbmNlciBpcyB0aGUgQkNHIHZhY2NpbmUgdXNlZCBmb3I/CmFuc3dlcjogTm9uLW11c2NsZSBJbnZhc2l2ZSBCbGFkZGVyIENhbmNlcgpxdWVzdGlvbjogSG93IG1hbnkgaW5qZWN0aW9ucyBvZiBDTFMtVEEgZGlkIHRoZSBwYXRpZW50cyBwYXJ0aWNpcGF0aW5nIGluIHRoZSBQRUFDSFRSRUUgdHJpYWwgcmVjZWl2ZT8KYW5zd2VyOiB0d28KCkNhdGVnb3J5OiAzIC0gUHJvdGVpbiBmdW5jdGlvbiwgbG9jYWxpemF0aW9uICYgc2lnbmFsbGluZy9lbnp5bWF0aWMgaW50ZXJhY3Rpb25zCnF1ZXN0aW9uOiBXaGljaCBoaXN0b25lIG1hcmsgZGlzdGluZ3Vpc2hlcyBhY3RpdmUgZnJvbSBpbmFjdGl2ZSBlbmhhbmNlcnM/CmFuc3dlcjogSDNLMjdhYwpxdWVzdGlvbjogV2hpY2ggY29tcG9uZW50IG9mIHRoZSBJbmZsdWVuemEgQSBWaXJ1cyBhZmZlY3RzIG1STkEgdHJhbnNjcmlwdGlvbiB0ZXJtaW5hdGlvbj8KYW5zd2VyOiBOUzEKcXVlc3Rpb246IFdoaWNoIGlzIHRoZSBtYWluIGNhbGNpdW0gYmluZGluZyBwcm90ZWluIG9mIHRoZSBzYXJjb3BsYXNtaWMgcmV0aWN1bHVtPwphbnN3ZXI6IENhbHNlcXVlc3RyaW4KCkNhdGVnb3J5OiA0IC0gRXhwZXJpbWVudGFsICYgY29tcHV0YXRpb25hbCBtZXRob2RzLCByZXNvdXJjZXMgJiBhY3JvbnltcwpxdWVzdGlvbjogV2hpY2ggYWxnb3JpdGhtIGhhcyBiZWVuIHByb3Bvc2VkIGZvciBlZmZpY2llbnQgc3RvcmFnZSBvZiBXR1MgdmFyaWFudCBjYWxscz8KYW5zd2VyOiBTZXFBcnJheQpxdWVzdGlvbjogV2hhdCBpcyBhbiBhY2NlcHRhYmxlIHNlcXVlbmNlIGNvdmVyYWdlKGRlcHRoKSByZXF1aXJlZCBmb3IgaHVtYW4gd2hvbGUtZXhvbWUgc2VxdWVuY2luZz8KYW5zd2VyOiAzMHgtNjB4CgpDYXRlZ29yeTogNSAtIERpc2Vhc2UgY2F1c2F0aW9uICYgcGF0aG9nZW5zCnF1ZXN0aW9uOiBXaGljaCBpcyB0aGUgbW9zdCBjb21tb24gZGlzZWFzZSBhdHRyaWJ1dGVkIHRvIG1hbGZ1bmN0aW9uIG9yIGFic2VuY2Ugb2YgcHJpbWFyeSBjaWxpYT8KYW5zd2VyOiBbJ1BvbHljeXN0aWMga2lkbmV5IGRpc2Vhc2UnLCAnUEtEJ10KcXVlc3Rpb246IFdoYXQgb3JnYW5pc20gY2F1c2VzIHNjYXJsZXQgZmV2ZXIgYWxzbyBrbm93biBhcyBzY2FybGV0aW5hPwphbnN3ZXI6IFsnR3JvdXAgQSBTdHJlcHRvY29jY3VzJywgJ1N0cmVwdG9jb2NjdXMgcHlvZ2VuZXMnXQpxdWVzdGlvbjogVGhlIHBhdGhvZ2VuIEZ1c2FyaXVtIGdyYW1pbmVhcnVtIGFmZmVjdHMgd2hhdCB0eXBlIG9mIHBsYW50IHNwZWNpZXM/CmFuc3dlcjogY2VyZWFsIGNyb3BzCgpDYXRlZ29yeTogNiAtIEJpb21hcmtlcnMgJiBkaWFnbm9zdGljIHRlc3RzCnF1ZXN0aW9uOiBTYWxpdmFyeSBDb3J0aXNvbCBpcyBhIGJpb21hcmtlciBmb3Igd2hhdCBkaXNlYXNlL3N5bmRyb21lL2NvbmRpdGlvbj8KYW5zd2VyOiBzdHJlc3MKcXVlc3Rpb246IFdoYXQgaXMgdGhlIGdvbGQgc3RhbmRhcmQgZm9yIGEgZGlhZ25vc2lzIG9mIG5hcmNvbGVwc3k/CmFuc3dlcjogWydTbGVlcCBzdHVkeScsICdvdmVybmlnaHQgcG9seXNvbW5vZ3JhcGh5J10KCkNhdGVnb3J5OiA3IC0gQmlvaW5mb3JtYXRpY3MgZGF0YWJhc2VzICYgY3VyYXRlZCByZXNvdXJjZXMKcXVlc3Rpb246IFdoaWNoIFIvYmlvY29uZHVjdG9yIHBhY2thZ2UgaGFzIGJlZW4gZGV2ZWxvcGVkIHRvIGFpZCBpbiBlcGlnZW5vbWljIGFuYWx5c2lzPwphbnN3ZXI6IERlZXBCbHVlUgpxdWVzdGlvbjogV2hpY2ggZGF0YWJhc2UgYXNzb2NpYXRlcyBodW1hbiBub25jb2RpbmcgU05QcyB3aXRoIHRoZWlyIHRocmVlLWRpbWVuc2lvbmFsIGludGVyYWN0aW5nIGdlbmVzPwphbnN3ZXI6IDNEU05QCnF1ZXN0aW9uOiBXaGF0IGlzIHRoZSBSRVNJRCBkYXRhYmFzZT8KcXVlc3Rpb246IFdoaWNoIGlzIHRoZSBsaXRlcmF0dXJlLWJhc2VkIGRhdGFiYXNlIG9mIHBoZW5vdHlwZXM/CmFuc3dlcjogUGhlbmVCYW5rCgpDYXRlZ29yeTogOCAtIENsaW5pY2FsIGdyYWRpbmcgJiBkaWFnbm9zdGljIHNjYWxlcyAvIGNsYXNzaWZpY2F0aW9uIHN5c3RlbXMKcXVlc3Rpb246IFdoYXQgY2FuIGJlIHByZWRpY3RlZCB3aXRoIHRoZSBXZWxscyBjcml0ZXJpYT8KYW5zd2VyOiBwdWxtb25hcnkgZW1ib2xpc20KcXVlc3Rpb246IFN5bXB0b21zIG9mIHdoaWNoIGRpc29yZGVyIGFyZSBldmFsdWF0ZWQgd2l0aCB0aGUgRGF2aWRzb24gVHJhdW1hIFNjYWxlPwphbnN3ZXI6IFsncG9zdC10cmF1bWF0aWMgc3RyZXNzIGRpc29yZGVyJywgJ1BUU0QnXQpxdWVzdGlvbjogV2hpY2ggdmFsdWUgb2YgbnVjaGFsIHRyYW5zbHVjZW5jeSB0aGlja25lc3MgaXMgc2V0IGFzIHRoZSB0aHJlc2hvbGQgZm9yIGhpZ2gtcmlzayBmb3IgRG93biBTeW5kcm9tZT8KYW5zd2VyOiAzbW0KCkNhdGVnb3J5OiA5IC0gQW5hdG9taWNhbCAvIGNlbGx1bGFyIHN0cnVjdHVyZXMgJiBsb2NhbGlzYXRpb24KcXVlc3Rpb246IFdoZXJlIGlzIGNvcnRpY29zdGVyb25lIHN5bnRoZXNpemVkPwphbnN3ZXI6IEFkcmVuYWwgZ2xhbmRzCnF1ZXN0aW9uOiBXaGljaCBpcyB0aGUgY2hyb21vc29tZSBhcmVhIHRoYXQgdGhlIGh1bWFuIGdlbmUgY29kaW5nIGZvciB0aGUgZG9wYW1pbmUgdHJhbnNwb3J0ZXIgKERBVDEpIGlzIGxvY2F0ZWQgdG8/CmFuc3dlcjogNXAxNS4zCnF1ZXN0aW9uOiBXaGVyZSBpcyB0aGUgcmVzcGlyYXNvbWUgbG9jYXRlZD8KYW5zd2VyOiBpbm5lciBtaXRvY2hvbmRyaWFsIG1lbWJyYW5lCgpDYXRlZ29yeTogMTAgLSBQc3ljaG9sb2d5IGFuZCBiZWhhdmlvcmFsIGhlYWx0aApRdWVzdGlvbjogV2hpY2ggcHN5Y2hvbW90b3IgZG9tYWluIHNob3dlZCBhIHNpZ25pZmljYW50IGRpZmZlcmVuY2UgYmV0d2VlbiBpbnN0aXR1dGlvbmFsaXplZCBhbmQgbm9uLWluc3RpdHV0aW9uYWxpemVkIHNoZWx0ZXJlZCBjaGlsZHJlbiBhbmQgYWRvbGVzY2VudHM/CkFuc3dlcjogQm9keSBhd2FyZW5lc3MKUXVlc3Rpb246IFdoYXQgZXRoaWNhbCBwcmluY2lwbGUganVzdGlmaWVzIGFjdGlvbnMgdGhhdCBoYXZlIGJvdGggZ29vZCBhbmQgaGFybWZ1bCBlZmZlY3RzLCBhcyBsb25nIGFzIHRoZSBoYXJtIGlzIG5vdCBpbnRlbmRlZCBidXQgb25seSBmb3Jlc2Vlbj8KQW5zd2VyOiBSdWxlIG9mIERvdWJsZSBFZmZlY3QKUXVlc3Rpb25zOiBXaGF0IHBzeWNob2xvZ2ljYWwgcHJvY2VzcyBkdXJpbmcgYW4gaW5jdWJhdGlvbiBwZXJpb2QgaXMgYXNzb2NpYXRlZCB3aXRoIGVuaGFuY2VkIGNyZWF0aXZlIHByb2JsZW0gc29sdmluZz8KQW5zd2VyOiBNaW5kLXdhbmRlcmluZwoKLS0tLS0tLS0KCk9VVFBVVCBGT1JNQVQKQSBzaW5nbGUgUUEgaGFzIHRhZ3MgYDxxdWVzdGlvbj4uLi48L3F1ZXN0aW9uPmAsIGFuc3dlciBpbnNpZGUgYDxhbnN3ZXI+Li4uPC9hbnN3ZXI+YC4KSWYgdGhlIFFBIGNvcnJlc3BvbmRzIHRvIG9uZSBvZiB0aGUgYWJvdmUgY2F0ZWdvcmllcyBwdXQgaXRzIG51bWJlciBpbiA8Y2F0X251bT4uLi48L2NhdF9udW0+IGFuZCBjYXRlZ29yeSBkZXNjcmlwdGlvbiBpbiA8Y2F0Pi4uLjwvY2F0Pi4KRWFjaCBRQSBzaG91bGQgZXhpc3QgaW4gaXRzIG93biB0YWcgPHFhPi4uLjwvcWE+CgpUaGVyZWZvcmUgdGhlIGZpcnN0IDIgcXVlc3Rpb25zIHdvdWxkIGJlOgo8cWFzPgogICA8cWE+IDxxdWVzdGlvbj4gLi4uIDwvcXVlc3Rpb24+CiAgICAgIDxhbnN3ZXI+IC4uLiA8L2Fuc3dlcj4KICAgICAgPGNhdF9udW0+IC4uLiA8L2NhdF9udW0+CiAgICAgIDxjYXQ+IC4uLiA8L2NhdD4KICAgPC9xYT4KICAgPHFhPgogICAgICAgLi4uLi4KICAgPC9xYT4KICAgLi4uCjwvcWFzPgoKLS0tLS0tLS0KVElUTEUgQU5EIEFCU1RSQUNUCnt0aXRsZV9hYnN0cmFjdH0KIiIi)

BACKGROUND

You are a domain-expert biomedical NLP assistant.

You are helping me to create an open-domain QA dataset.

The downstream task will read a query and require an agent to search
over Pubmed abstracts

--------

YOUR TASK

I will provide you with title and abstract of a Pubmed article.

Your task is to create 3 new question-answer pairs.

--------

TYPES OF QUESTIONS

The questions should be ’factoid based’.

The answer should be a simple entity.

It should not be ambiguous.

Don’t be pretentious.

--------

IMPORTANT NOTES

The question-answer pair will be used to evaluation question-answering
systems with retrieval. Ths means the target system does not know which
paper the question was sourced from. So an inappropriate question would
be "What technology is used in this study to ...". or "what type of
treatment is assessed in this study?" (where the study name is not
specifified).

If the question contains acronyms that are not well known, then explain
the acronym.

--------

EXAMPLE CATEGORIES

Below are sample categories with sample questions.

Category: 1 - Genetic inheritance & disease-linked mutations

question: What gene is mutated in Sickle Cell Anemia?

answer: HBB

question: Which ultraconserved element is associated with Embryonic Stem
Cells (ESC) self-renewal?

answer: T-UCstem1

question: Is Huntington’s disease caused by a dominate or recessive
gene?

answer: dominant

Category: 2 - Therapeutics, indications & clinical evidence

question: What is the most effective drug for oxaliplatin-induced
neuropathy?

answer: Duloxetine

question: Which cancer is the BCG vaccine used for?

answer: Non-muscle Invasive Bladder Cancer

question: How many injections of CLS-TA did the patients participating
in the PEACHTREE trial receive?

answer: two

Category: 3 - Protein function, localization & signalling/enzymatic
interactions

question: Which histone mark distinguishes active from inactive
enhancers?

answer: H3K27ac

question: Which component of the Influenza A Virus affects mRNA
transcription termination?

answer: NS1

question: Which is the main calcium binding protein of the sarcoplasmic
reticulum?

answer: Calsequestrin

Category: 4 - Experimental & computational methods, resources & acronyms

question: Which algorithm has been proposed for efficient storage of WGS
variant calls?

answer: SeqArray

question: What is an acceptable sequence coverage(depth) required for
human whole-exome sequencing?

answer: 30x-60x

Category: 5 - Disease causation & pathogens

question: Which is the most common disease attributed to malfunction or
absence of primary cilia?

answer: \[’Polycystic kidney disease’, ’PKD’\]

question: What organism causes scarlet fever also known as scarletina?

answer: \[’Group A Streptococcus’, ’Streptococcus pyogenes’\]

question: The pathogen Fusarium graminearum affects what type of plant
species?

answer: cereal crops

Category: 6 - Biomarkers & diagnostic tests

question: Salivary Cortisol is a biomarker for what
disease/syndrome/condition?

answer: stress

question: What is the gold standard for a diagnosis of narcolepsy?

answer: \[’Sleep study’, ’overnight polysomnography’\]

Category: 7 - Bioinformatics databases & curated resources

question: Which R/bioconductor package has been developed to aid in
epigenomic analysis?

answer: DeepBlueR

question: Which database associates human noncoding SNPs with their
three-dimensional interacting genes?

answer: 3DSNP

question: What is the RESID database?

question: Which is the literature-based database of phenotypes?

answer: PheneBank

Category: 8 - Clinical grading & diagnostic scales / classification
systems

question: What can be predicted with the Wells criteria?

answer: pulmonary embolism

question: Symptoms of which disorder are evaluated with the Davidson
Trauma Scale?

answer: \[’post-traumatic stress disorder’, ’PTSD’\]

question: Which value of nuchal translucency thickness is set as the
threshold for high-risk for Down Syndrome?

answer: 3mm

Category: 9 - Anatomical / cellular structures & localisation

question: Where is corticosterone synthesized?

answer: Adrenal glands

question: Which is the chromosome area that the human gene coding for
the dopamine transporter (DAT1) is located to?

answer: 5p15.3

question: Where is the respirasome located?

answer: inner mitochondrial membrane

Category: 10 - Psychology and behavioral health

Question: Which psychomotor domain showed a significant difference
between institutionalized and non-institutionalized sheltered children
and adolescents?

Answer: Body awareness

Question: What ethical principle justifies actions that have both good
and harmful effects, as long as the harm is not intended but only
foreseen?

Answer: Rule of Double Effect

Questions: What psychological process during an incubation period is
associated with enhanced creative problem solving?

Answer: Mind-wandering

--------

OUTPUT FORMAT

A single QA has tags ‘\<question\>...\</question\>‘, answer inside
‘\<answer\>...\</answer\>‘.

If the QA corresponds to one of the above categories put its number in
\<cat_num\>...\</cat_num\> and category description in
\<cat\>...\</cat\>.

Each QA should exist in its own tag \<qa\>...\</qa\>

Therefore the first 2 questions would be:

\<qas\>

\<qa\> \<question\> ... \</question\>

\<answer\> ... \</answer\>

\<cat_num\> ... \</cat_num\>

\<cat\> ... \</cat\>

\</qa\>

\<qa\>

.....

\</qa\>

...

\</qas\>

--------

TITLE AND ABSTRACT

{title_abstract}

"""

And here is the prompt for generating ‘golden answers’ or synonyms to
the ground truth answer.

[⬇](data:text/plain;base64,WW91IGFyZSBnaXZlbiBhIHF1ZXN0aW9uIHRoYXQgd2FzIHdyaXR0ZW4gdXNpbmcgYSBwYXJ0aWN1bGFyIGRvY3VtZW50IGFzIGl0cyBtYWluIHNvdXJjZS4gWW91ciB0YXNrIGlzIHRvIHJld3JpdGUgdGhlIHF1ZXN0aW9uIHNvIHRoYXQgaXQgcmV0YWlucyB0aGUgb3JpZ2luYWwgbWVhbmluZyBhbmQgd291bGQgcmVzdWx0IGluIHRoZSBzYW1lIGNvcnJlY3QgYW5zd2VyLCBidXQgdXNlcyBkaWZmZXJlbnQgd29yZGluZyBhbmQgcGhyYXNpbmcuIEltcG9ydGFudCBjb25zdHJhaW50czoKRG8gbm90IGJyb2FkZW4gb3IgbmFycm93IHRoZSBzY29wZSBvZiB0aGUgcXVlc3Rpb24uCkRvIG5vdCBpbnRyb2R1Y2UgYW1iaWd1aXR5IG9yIGFsdGVyIGNsaW5pY2FsL3RlY2huaWNhbCBjb250ZXh0LgpNYWtlIHN1cmUgdGhlIGNvcnJlY3QgYW5zd2VyIHJlbWFpbnMgZXhhY3RseSB0aGUgc2FtZS4KWW91ciBnb2FsIGlzIHRvIGNoYW5nZSB0aGUgc3VyZmFjZSB3b3JkaW5nIHNvIHRoYXQgc2ltcGxlIGJhZy1vZi13b3JkcyBzZWFyY2ggKGxpa2UgQk0yNSkgbWF5IG5vdCBlYXNpbHkgbWF0Y2ggdGhlIG9yaWdpbmFsIGRvY3VtZW50LCB3aGlsZSBhbiBleHBlcnQgaHVtYW4gb3Igc3Ryb25nIGxhbmd1YWdlIG1vZGVsIGNvdWxkIHN0aWxsIGFuc3dlciBjb3JyZWN0bHkuCkF2b2lkIGNvcHlpbmcgYW55IHNpZ25pZmljYW50IHBocmFzZSAodGhyZWUgb3IgbW9yZSB3b3JkcyBpbiBzZXF1ZW5jZSkgZnJvbSB0aGUgb3JpZ2luYWwgcXVlc3Rpb24uCgpFeGFtcGxlOgotIE9yaWdpbmFsOiBXaGF0IGNvbmdlbml0YWwgYWJub3JtYWxpdHkgY2FuIGNhdXNlIHVuaWxhdGVyYWwgaHlkcm9jZXBoYWx1cyBpbiB0aGUgcGVyaW5hdGFsIHBlcmlvZD8KLSBFZGl0ZWQ6IFdoaWNoIGJpcnRoIGRlZmVjdCBwcmVzZW50IGR1cmluZyB0aGUgcGVyaW5hdGFsIHN0YWdlIG1heSByZXN1bHQgaW4gaHlkcm9jZXBoYWx1cyBhZmZlY3Rpbmcgb25seSBvbmUgc2lkZSBvZiB0aGUgYnJhaW4/CgpPdXRwdXQgc2hvdWxkIGJlIGluIHRhZ3MgbGlrZSA8cXVlc3Rpb24+IC4uLiA8L3F1ZXN0aW9uPgoKUXVlc3Rpb246IHtxdWVzdGlvbn0KQW5zd2VyOiB7YW5zd2VyfQ==)

You are given a question that was written using a particular document as
its main source. Your task is to rewrite the question so that it retains
the original meaning and would result in the same correct answer, but
uses different wording and phrasing. Important constraints:

Do not broaden or narrow the scope of the question.

Do not introduce ambiguity or alter clinical/technical context.

Make sure the correct answer remains exactly the same.

Your goal is to change the surface wording so that simple bag-of-words
search (like BM25) may not easily match the original document, while an
expert human or strong language model could still answer correctly.

Avoid copying any significant phrase (three or more words in sequence)
from the original question.

Example:

- Original: What congenital abnormality can cause unilateral
hydrocephalus in the perinatal period?

- Edited: Which birth defect present during the perinatal stage may
result in hydrocephalus affecting only one side of the brain?

Output should be in tags like \<question\> ... \</question\>

Question: {question}

Answer: {answer}

## Appendix F System prompt for Search-R1 LLM training

The LLM system prompt provides basic guidance about what tools are
available, as well as guidance about putting the final answer in tags.

[⬇](data:text/plain;base64,QW5zd2VyIHRoZSBnaXZlbiBxdWVzdGlvbi4gWW91IG11c3QgY29uZHVjdCByZWFzb25pbmcgaW5zaWRlIDx0aGluaz4gYW5kIDwvdGhpbms+IGZpcnN0IGV2ZXJ5IHRpbWUgeW91IGdldCBuZXcgaW5mb3JtYXRpb24uIEFmdGVyIHJlYXNvbmluZywgaWYgeW91IGZpbmQgeW91IGxhY2sgc29tZSBrbm93bGVkZ2UsIHlvdSBjYW4gY2FsbCBhIHNlYXJjaCBlbmdpbmUgYnkgPHNlYXJjaD4gcXVlcnkgPC9zZWFyY2g+IGFuZCBpdCB3aWxsIHJldHVybiB0aGUgdG9wIHNlYXJjaGVkIHJlc3VsdHMgYmV0d2VlbiA8aW5mb3JtYXRpb24+IGFuZCA8L2luZm9ybWF0aW9uPi4gWW91IGNhbiBzZWFyY2ggYXMgbWFueSB0aW1lcyBhcyB5b3VyIHdhbnQuIElmIHlvdSBmaW5kIG5vIGZ1cnRoZXIgZXh0ZXJuYWwga25vd2xlZGdlIG5lZWRlZCwgeW91IGNhbiBkaXJlY3RseSBwcm92aWRlIHRoZSBhbnN3ZXIgaW5zaWRlIDxhbnN3ZXI+IGFuZCA8L2Fuc3dlcj4sIHdpdGhvdXQgZGV0YWlsZWQgaWxsdXN0cmF0aW9ucy4gRm9yIGV4YW1wbGUsIDxhbnN3ZXI+IEJlaWppbmcgPC9hbnN3ZXI+LiBRdWVzdGlvbjoge3F1ZXN0aW9ufVxu)

Answer the given question. You must conduct reasoning inside \<think\>
and \</think\> first every time you get new information. After
reasoning, if you find you lack some knowledge, you can call a search
engine by \<search\> query \</search\> and it will return the top
searched results between \<information\> and \</information\>. You can
search as many times as your want. If you find no further external
knowledge needed, you can directly provide the answer inside \<answer\>
and \</answer\>, without detailed illustrations. For example, \<answer\>
Beijing \</answer\>. Question: {question}\\n

For baseline experiments we apply the same formatting instruction.

## Appendix G Results: per-category performance

Since PaperQA2 has per-category labels
([fig. 2](#S3.F2 "In 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR")),
we report the main results split by these category values. The main
results are in
[table 2](#A7.T2 "In Appendix G Results: per-category performance ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").

|  | Data portion | 3b models |  |  |  |  | 7b models |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|
|  |  | Direct | CoT | RAG | PaperQA2 | Search-R1 | Direct | CoT | RAG | PaperQA2 | Search-R1 |
| Genetic mutations | 3.6 | 12 | 17 | 40 | 20 | 27 | 18 | 18 | 45 | 19 | 26 |
| Therapeutics & clinical evidence | 17 | 17 | 23 | 31 | 28 | 38 | 27 | 32 | 37 | 32 | 46 |
| Protein function & signalling | 12.36 | 15 | 20 | 39 | 32 | 44 | 28 | 29 | 46 | 37 | 53 |
| Methods & resources | 26.36 | 14 | 16 | 26 | 25 | 35 | 26 | 25 | 30 | 27 | 37 |
| Disease causation & pathogens | 12.96 | 24 | 27 | 38 | 33 | 39 | 34 | 38 | 43 | 36 | 52 |
| Biomarkers & diagnostics | 10.38 | 20 | 19 | 26 | 34 | 46 | 29 | 32 | 30 | 40 | 56 |
| Bioinformatics databases | 0.16 | 13 | 25 | 13 | 100 | 100 | 13 | 13 | 13 | 100 | 100 |
| Clinical scales & classifications | 2.82 | 16 | 16 | 26 | 25 | 34 | 23 | 26 | 28 | 34 | 50 |
| Anatomy & cellular localisation | 8.74 | 13 | 22 | 39 | 24 | 32 | 27 | 30 | 42 | 28 | 37 |
| Psychology & behavioural health | 3.4 | 16 | 19 | 28 | 26 | 33 | 27 | 31 | 31 | 30 | 39 |

Table 2: Main results of baselines vs Search-R1 training [Jin et al.
(2025)](#bib.bib1) that uses RLVR. Unlike the table in the main results,
we show the per-category scores, where the categories are defined in
[fig. 2](#S3.F2 "In 3.1 Dataset Construction ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR").

## Appendix H Results: a note on PaperQA baseline

For PaperQA baselines we used the official codebase
([https://github.com/Future-House/paper-qa](https://github.com/Future-House/paper-qa))
and then for fair comparison with other methods, we matched the model
and system components – the result is in the our released code, in the
baselines/ folder.

For retrieval backend: we replaced PaperQA’s retrieval system with
SearchR1’s retrieval servers, using the same BM25 and E5 dense retrieval
on the PubMed corpus that was used in all paper experiments. For LLM
Integration: switched from proprietary APIs to local Qwen 2.5 models
(3B/7B variants) served via vLLM, matching the exact models used in the
SearchR1 training experiments. For answer format compatibility: we
appended an instruction to the end of the PaperQA text prompt
instructing the system to put the final single-entity answer into
\<answer\> blocks, consistent with all the other content. We created a
standalone evaluation module, qa_em.py, which matched the evaluation
logic used in SearchR1 codebase (which is inside the verl/ folder.

## Appendix I Training RLVR details

This section is single-column due to the large equation below.
Continuing the description of the RL training algorithm from
[section 3.4](#S3.SS4 "3.4 Training Algorithms ‣ 3 Methods ‣ PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR"),
we leverage Group Relative Policy Optimization (GRPO) [Shao et al.
(2024)](#bib.bib21); [Guo et al. (2025)](#bib.bib18). At each iteration,
we have the current policy $`\pi_{\theta}`$, which we now temporarily
call the ‘old policy’ $`\pi_{old}`$. For each question, $`x`$, GRPO
computes multiple rollouts $`\{y_{1},y_{2},\dots,y_{G}\}`$ using
$`\pi_{old}`$, and we can now consider some averaging of rewards ina
group. The policy model is then optimized by maximizing:

|  |  |  |  |  |
|----|----|----|----|----|
|  | $`\displaystyle\mathcal{J}_{GRPO}(\theta)=\,`$ | $`\displaystyle\mathbb{E}_{x\sim\mathcal{D},\{y_{i}\}_{i=1}^{G}\sim\pi_{\text{old}}(\cdot|x;\mathcal{R})}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{\sum_{t=1}^{|y_{i}|}I(y_{i,t})}\sum_{t=1:I(y_{i,t})=1}^{|y_{i}|}\min\Bigg(\frac{\pi_{\theta}(y_{i,t}|x,y_{i,<t};\mathcal{R})}{\pi_{\text{old}}(y_{i,t}|x,y_{i,<t};\mathcal{R})}\hat{A}_{i,t},`$ |  |  |
|  |  | $`\displaystyle\hskip 120.0pt\text{clip}\Bigg(\frac{\pi_{\theta}(y_{i,t}|x,y_{i,<t};\mathcal{R})}{\pi_{\text{old}}(y_{i,t}|x,y_{i,<t};\mathcal{R})},1-\epsilon,1+\epsilon\Bigg)\hat{A}_{i,t}\Bigg)-\beta\mathbb{D}_{KL}\left[\pi_{\theta}||\pi_{\text{ref}}\right]\Bigg],`$ |  | (1) |

Here, $`\mathcal{R}`$ is the retriever (as before), $`\epsilon`$
controls clipping range, and $`\beta`$ controls KL penalty. We compute
‘advantages’ (rather than raw reward), $`\hat{A}_{i,t}`$ by normalizing
rewards within each group of $`G`$ responses by using group mean as
baseline and group standard deviation for scaling.

The full training scripts with all hyperparameters are available in the
released code.
````
