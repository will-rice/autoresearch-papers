---
identifier: arxiv:2607.13960
title: "GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"
authors:
  - GigaWorld Team
  - Angen Ye
  - Angyuan Ma
  - Boyuan Wang
  - Chaojun Ni
  - Fangzheng Ye
  - Guan Huang
  - Guo Li
  - Guosheng Zhao
  - Haodong Yan
  - Hengtao Li
  - Jiwen Lu
  - Kai Wang
  - Mingming Yu
  - Qitang Hu
  - Qiuping Deng
  - Songling Liu
  - Xiaoyu Tian
  - Xiaofeng Wang
  - Xinyu Zhou
  - Xiuwei Xu
  - Xinze Chen
  - Yang Wang
  - Yejun Zeng
  - Yifan Chang
  - Yun Ye
  - Zhenyu Wu
  - Zhanqian Wu
  - Zheng Zhu
published: "2026-07-15T00:00:00+00:00"
url: https://huggingface.co/papers/2607.13960
source: huggingface
doi: null
arxiv_id: "2607.13960"
categories: []
---

# GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch

GigaAI  
Tsinghua University  
Project Page:
[https://open-gigaai.github.io/giga-world-policy/](https://open-gigaai.github.io/giga-world-policy)  
Alphabetical Order: Angen Ye    Angyuan Ma    Boyuan Wang    Chaojun Ni
   Fangzheng Ye    Guan Huang    Guo Li    Guosheng Zhao     
Haodong Yan    Hengtao Li    Jiwen Lu    Kai Wang    Mingming Yu   
Qitang Hu    Qiuping Deng    Songling Liu    Xiaoyu Tian    Xiaofeng
Wang     
Xinyu Zhou    Xiuwei Xu    Xinze Chen    Yang Wang    Yejun Zeng   
Yifan Chang    Yun Ye    Zhenyu Wu    Zhanqian Wu    Zheng Zhu

###### Abstract

World Action Models (WAMs) improve robot policy learning by jointly
modeling actions and future visual observations, using future scene
evolution as dense supervision for physically grounded action
generation. However, a common design in existing WAMs is to explicitly
generate future videos at inference time, incurring substantial
computational overhead and hindering real-time closed-loop deployment.
GigaWorld-Policy addresses this issue with an action-centered
formulation, where future visual dynamics are used during training while
action-only decoding is used at inference time. Building upon this
framework, we present GigaWorld-Policy-0.5, an enhanced action-centered
WAM designed for more efficient robot control. During pretraining,
GigaWorld-Policy-0.5 adopts a mixed Action-Conditioned World Modeling
(AC-WM) and WAM training strategy. This strengthens the coupling between
visual dynamics and robot actions and improves the transferability of
action representations for downstream policy learning. For efficient
inference, GigaWorld-Policy-0.5 introduces a Mixture-of-Transformers
architecture that separates visual dynamics modeling and action
generation into specialized experts, reducing active computation during
action-only inference and achieving 85 ms inference latency on a local
RTX 4090 setup. In addition, we employ an agent-based AutoResearch
pipeline to systematically search training configurations, enabling more
efficient identification of optimal experimental setups while reducing
the time and manual intervention required for hyperparameter tuning.
Experiments and ablations show that GigaWorld-Policy-0.5 preserves the
training benefits of future visual dynamics while improving inference
efficiency for robot control.

\abscontent

## 1 Introduction

Figure 1: Comparison of GigaWorld-Policy-0.5 with baselines on inference
frequency and success rate across hardware platforms in real-world
settings.

World models have recently made significant progress in learning
predictive representations of the physical world from large-scale visual
data \[[41](#bib.bib2), [2](#bib.bib5), [1](#bib.bib6), [46](#bib.bib3),
[62](#bib.bib4), [39](#bib.bib1), [20](#bib.bib7), [24](#bib.bib62),
[68](#bib.bib63), [47](#bib.bib64)\]. By modeling how scenes evolve over
time, these models capture rich spatiotemporal priors about object
motion, interaction dynamics, and long-horizon state transitions. Such
predictive capability is naturally attractive for robotic control, where
a policy must not only recognize the current observation, but also
anticipate how its actions may change the environment.

Motivated by this progress, recent Vision-Language-Action (VLA) models
have started to leverage world models to improve policy learning
\[[14](#bib.bib8), [37](#bib.bib12), [36](#bib.bib13), [34](#bib.bib9),
[27](#bib.bib10), [8](#bib.bib11)\]. Compared with prior VLA methods
\[[6](#bib.bib15), [16](#bib.bib16), [15](#bib.bib17), [37](#bib.bib12),
[36](#bib.bib13), [30](#bib.bib14), [5](#bib.bib18), [58](#bib.bib19),
[50](#bib.bib23), [55](#bib.bib65)\] that primarily rely on sparse
action supervision from robot demonstrations, world models provide
complementary predictive information about how the scene may evolve.
Incorporating such predictive information into VLA policies provides
richer temporal context for action prediction and helps ground policy
learning in future scene evolution. This line of work suggests that
predictive world knowledge is an important ingredient for building more
capable robot policies.

Beyond incorporating world model predictions into VLA policies, another
line of work explores a more unified formulation through World Action
Models (WAMs), which jointly model robot actions and future observations
\[[4](#bib.bib20), [40](#bib.bib21), [22](#bib.bib22), [53](#bib.bib66),
[28](#bib.bib67)\]. Instead of treating future prediction as an external
input or auxiliary signal to a policy model, WAMs integrate action
generation and future-dynamics modeling within the same framework. This
formulation allows action representations to be learned together with
their visual consequences: the model predicts not only what action
should be executed, but also how the scene is expected to evolve under
that action. As a result, future visual dynamics can serve as dense
supervision for learning physically grounded actions.

While this unified formulation enables WAMs to acquire certain zero-shot
generalization capabilities by coupling action generation with future
scene evolution \[[56](#bib.bib25)\], it also introduces a practical
deployment challenge. In many existing WAM approaches, the same joint
modeling process used for training is also invoked during inference,
requiring explicit future-video generation, iterative denoising, or
predictive rollout at deployment time \[[33](#bib.bib24),
[56](#bib.bib25)\]. Since video tokens are substantially more expensive
than action tokens, this design introduces significant computational
overhead and limits real-time closed-loop control. Moreover, errors in
imagined future frames may accumulate over long horizons and weaken
downstream action generation. Therefore, a key challenge is how to
preserve the generalization benefit of action-dynamics coupling while
avoiding costly future rollout during inference.

GigaWorld-Policy \[[54](#bib.bib27)\] addresses this challenge with an
efficient action-centered World Action Model, following the broader
direction of decoupling future prediction between training and inference
\[[57](#bib.bib26), [54](#bib.bib27)\]. It leverages future visual
dynamics to provide dense supervision during training, while allowing
the policy to skip explicit future-video generation and directly decode
actions at inference time. Through an action-centered causal token
structure, action prediction is separated from future-video prediction,
retaining the benefit of WAM training while substantially reducing
inference latency for closed-loop deployment. As a result,
GigaWorld-Policy demonstrates that WAMs can achieve low-latency action
generation comparable to efficient VLA policies while retaining the
stronger supervision and physical grounding provided by future visual
dynamics.

Building upon GigaWorld-Policy, we present GigaWorld-Policy-0.5.
GigaWorld-Policy-0.5 inherits the core formulation of GigaWorld-Policy,
where future visual dynamics are used to provide dense supervision
during training, while inference can be performed through action-only
decoding without explicit future-video generation. It also preserves the
action-centered causal token structure: action tokens are predicted from
the current observation, robot state, and language instruction, while
future visual tokens are predicted conditioned on the action sequence.
This masking strategy prevents future visual information from leaking
into action prediction, and allows future-video prediction to remain
optional during deployment, thereby retaining the low-latency control
advantage of GigaWorld-Policy.

Compared with GigaWorld-Policy, GigaWorld-Policy-0.5 introduces several
updates to the model architecture and training pipeline. First, it
adopts a Mixture-of-Transformers (MoT) architecture that separates
visual dynamics modeling and action generation into specialized experts
while preserving the action-centered WAM design. Although MoT increases
the total parameter count, its expert specialization enables an
action-only inference pathway that avoids unnecessary visual-dynamics
computation, achieving an inference latency of approximately 85 ms on an
NVIDIA RTX 4090 and thereby improving deployment efficiency. Second,
during pretraining, we adopt a mixed training strategy that combines
Action-Conditioned World Modeling (AC-WM) with World Action Model
training. This encourages the model to better capture the relationship
between visual dynamics and robot actions, learning both general scene
evolution and action-induced future changes, which in turn strengthens
its action modeling capability. Finally, we employ an agent-based
AutoResearch \[[18](#bib.bib61)\] pipeline to make the experimental
search process more systematic and efficient. The pipeline automatically
launches pilot runs, monitors validation metrics, compares candidate
configurations, and promotes promising settings to longer training.
Through this iterative process, AutoResearch helps derive a reliable
training recipe for GigaWorld-Policy-0.5 with reduced manual
intervention.

## 2 Related Work

### 2.1 World Models as Data Engines for Robotics

Recent advances in world models \[[21](#bib.bib42), [31](#bib.bib43),
[60](#bib.bib45), [63](#bib.bib57), [65](#bib.bib58), [45](#bib.bib59)\]
have improved robotic video generation and prediction
\[[11](#bib.bib46), [25](#bib.bib39), [30](#bib.bib14),
[42](#bib.bib47), [64](#bib.bib54)\] and have increasingly served as
scalable data engines or learned simulators for robot learning
\[[25](#bib.bib39), [3](#bib.bib40), [39](#bib.bib1), [17](#bib.bib48),
[44](#bib.bib68)\]. The central goal is to learn a generative or
predictive model that captures the temporal evolution of embodied
environments, enabling the synthesis or imagination of diverse visual
experiences from limited real-world or simulated trajectories.

At the level of action-conditioned future prediction, Pandora
\[[51](#bib.bib37)\] demonstrates that free-form text actions can be
used to control video world models, enabling interactive prediction
beyond passive video generation. FreeAction \[[19](#bib.bib38)\] further
considers continuous robot actions and introduces action-scaled
classifier-free guidance to control the intensity of generated motions.
Building on such controllability, subsequent efforts have explored
scalable generation and transfer of embodied visual data. RoboTransfer
\[[25](#bib.bib39)\] and Cosmos-Transfer1 \[[3](#bib.bib40)\] focus on
transferring visual experiences across domains, leveraging 3D or
multimodal spatial conditions to improve geometric consistency and
Sim2Real data generation. GigaWorld-0 \[[39](#bib.bib1)\] provides a
high-fidelity video-and-3D data engine that synthesizes temporally
coherent embodied trajectories with fine-grained control over
appearance, scene layouts, viewpoints, and action semantics.
Qwen-RobotWorld \[[59](#bib.bib41)\] further scales action-conditioned
embodied video generation across multiple robotics domains, including
manipulation, navigation, and driving. Going beyond video generation
alone, Aether \[[67](#bib.bib60)\] unifies geometry-aware world modeling
by jointly optimizing 4D dynamic reconstruction, action-conditioned
video prediction, and goal-conditioned visual planning.

However, most existing efforts use world models primarily as external
data engines or simulators \[[43](#bib.bib44), [61](#bib.bib56)\], while
largely overlooking how their predictive priors can be embedded into
deployable robot policies. This motivates recent work on world models
for robot control, where future prediction and action generation are
jointly considered.

### 2.2 World Models for Robot Control

World models have long been studied as predictive representations for
decision making, and recent progress in large-scale video generation has
renewed their relevance to robotic control. Instead of mapping
observations directly to actions, world-model-based policies exploit
predictions of future scene evolution to provide temporal structure,
dynamics priors, or intermediate plans. Early video-based policy
methods, such as UniPi \[[12](#bib.bib28)\], formulate control as future
video generation followed by action recovery, while generative robot
models such as GR-2 \[[48](#bib.bib29)\] further combine large-scale
video pretraining with action prediction for generalizable manipulation.

World Action Models \[[4](#bib.bib20), [22](#bib.bib22),
[33](#bib.bib24), [54](#bib.bib27)\], grounded in the video generation
paradigm, aim to model robot actions and future visual dynamics within a
unified framework. By coupling action prediction with future visual
modeling, WAMs provide dense temporal supervision and learned predictive
priors beyond sparse demonstration actions. VideoVLA
\[[33](#bib.bib24)\] adapts pretrained video generation models into
robotic manipulators, jointly predicting future visual outcomes and
action sequences. Motus \[[4](#bib.bib20)\] introduces a unified latent
action world model with a Mixture-of-Transformer architecture and
UniDiffuser-style scheduling, enabling multiple modes including world
modeling, inverse dynamics, policy learning, video generation, and joint
video-action prediction. UWM \[[66](#bib.bib30)\] similarly couples
video and action diffusion within a unified framework, showing that
heterogeneous video and robot data can support scalable policy learning.
Building on this direction, MotuBrain \[[40](#bib.bib21)\] extends Motus
with multiview modeling, cross-embodiment action representations, and
deployment-oriented optimization for real-time control.

Another branch investigates how much future prediction must be
explicitly realized during policy inference. Mimic-video
\[[32](#bib.bib31)\] uses an Internet-scale video backbone to produce
latent video plans, which are decoded into actions through a
flow-matching inverse-dynamics model. LingBot-VA \[[22](#bib.bib22)\]
adopts autoregressive video-action world modeling with closed-loop
rolling prediction to mitigate error accumulation during long-horizon
execution. S-VAM \[[52](#bib.bib32)\] distills multi-step video
foresight into efficient geometric and semantic representations, DiT4DiT
\[[29](#bib.bib33)\] uses intermediate denoising features rather than
decoded frames for action prediction, and LaWAM \[[9](#bib.bib34)\]
exposes future dynamics through compact latent visual subgoals instead
of pixel-level video.

Complementary to these approaches, GigaWorld-Policy \[[54](#bib.bib27)\]
and Fast-WAM \[[57](#bib.bib26)\] explore more efficient ways to
incorporate predictive world knowledge into action policies. Both
methods use video-based predictive supervision during training while
avoiding the need to explicitly generate future videos at test time.
Together, these works indicate that the benefits of world modeling need
not rely on computationally expensive pixel-level future prediction
during policy execution.

Overall, prior work has explored a spectrum of designs that reduce the
reliance on explicit pixel-level future rollout during policy inference.
However, achieving the low latency required for closed-loop control on
edge devices remains challenging. This motivates GigaWorld-Policy-0.5,
which retains an action-centered world action modeling formulation while
improving inference efficiency through a lightweight action-only
execution pathway.

## 3 Method

![Refer to caption](2607.13960v3/framework.png)

Figure 2: Overview of GigaWorld-Policy-0.5, an MoT-based action-centered
World Action Model. The model consists of a visual expert and an action
expert: the visual expert specializes in processing video tokens, while
the action expert focuses on action-token modeling. The two experts are
connected through multi-modal self-attention, which follows the same
causal masking strategy as GigaWorld-Policy \[[54](#bib.bib27)\] to
preserve action-centered dependency modeling.

### 3.1 Overview of GigaWorld-Policy-0.5

The overall framework of GigaWorld-Policy-0.5 is illustrated in Fig.
[2](#S3.F2 "Figure 2 ‣ 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
It follows the formulation of the action-centered world action model
introduced in GigaWorld-Policy \[[54](#bib.bib27)\]. The policy
prediction problem under this action-centered setting can be formulated
as follows. Given the current multi-view observation
$`\{o_{t}^{\mathrm{left}},o_{t}^{\mathrm{front}},o_{t}^{\mathrm{right}}\}`$,
proprioceptive state $`s_{t}`$, and language instruction $`l`$, the
model jointly predicts an action chunk $`\hat{\mathbf{a}}_{t:t+p-1}`$ of
length $`p`$ and future visual observations
$`\hat{\mathbf{o}}_{t+\Delta:t+K\Delta}^{\mathrm{comp}}`$:

|     |     |     |     |
| --- | --- | --- | --- |
|     |

````math
\left(\hat{\mathbf{a}}_{t:t+p-1},\hat{\mathbf{o}}_{t+\Delta:t+K\Delta}^{\mathrm{comp}}\right)\sim g_{\theta}\left(\cdot\mid o_{t}^{\mathrm{comp}},s_{t},l\right),
``` |  | (1) |

where

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\hat{\mathbf{a}}_{t:t+p-1}=\left(\hat{a}_{t},\hat{a}_{t+1},\ldots,\hat{a}_{t+p-1}\right),\quad\hat{\mathbf{o}}_{t+\Delta:t+K\Delta}^{\mathrm{comp}}=\left(\hat{o}_{t+\Delta}^{\mathrm{comp}},\hat{o}_{t+2\Delta}^{\mathrm{comp}},\ldots,\hat{o}_{t+K\Delta}^{\mathrm{comp}}\right).
``` |  | (2) |

Here, $`\Delta`$ denotes the temporal stride for future visual
prediction, and $`K=\lfloor p/\Delta\rfloor`$ denotes the number of
predicted future observations within the action horizon.

The composite observation $`o^{\mathrm{comp}}`$ is constructed by
concatenating the left, front, and right camera views:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
o^{\mathrm{comp}}=\mathrm{Compose}\left(o^{\mathrm{left}},o^{\mathrm{front}},o^{\mathrm{right}}\right).
``` |  | (3) |

Notably, GigaWorld-Policy-0.5 adopts a similar multi-view composition
strategy as Motus \[[4](#bib.bib20)\], where the front view is placed on
the top and the left and right views are concatenated in the bottom row
to form a unified image input for visual encoding.

To better transfer world-model priors into policy learning,
GigaWorld-Policy-0.5 adopts the same causal masking strategy as
GigaWorld-Policy \[[54](#bib.bib27)\]. Specifically, future visual
tokens are allowed to attend to action tokens, while action tokens are
prevented from attending to future visual tokens. This design has two
benefits. First, during training, the causal attention pattern
implicitly conditions future visual prediction on the action tokens,
encouraging the model to learn action representations that are
consistent with the world-model prior and the expected scene evolution.
Second, during inference, future visual prediction is optional; when
low-latency control is desired, the model can skip the large number of
future visual tokens and directly decode action tokens.

GigaWorld-Policy-0.5 is optimized with the flow matching framework
\[[13](#bib.bib35), [23](#bib.bib36)\]. For action tokens and future
visual tokens, we sample modality-specific flow timesteps with different
flow-shift factors. Given
$`r^{\mathrm{a}},r^{\mathrm{v}}\sim\mathcal{U}(0,1)`$ and flow-shift
factors $`\gamma_{\mathrm{a}}`$ and $`\gamma_{\mathrm{v}}`$, the shifted
timesteps are computed as

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\tau^{\mathrm{a}}=\frac{\gamma_{\mathrm{a}}r^{\mathrm{a}}}{1+(\gamma_{\mathrm{a}}-1)r^{\mathrm{a}}},\quad\tau^{\mathrm{v}}=\frac{\gamma_{\mathrm{v}}r^{\mathrm{v}}}{1+(\gamma_{\mathrm{v}}-1)r^{\mathrm{v}}}.
``` |  | (4) |

We then form a joint timestep vector for the action and future visual
modalities:

|     |                                         |     |     |
|-----|-----------------------------------------|-----|-----|
|     |
       ``` math
       \tau=\begin{bmatrix}\tau^{\mathrm{a}}\\
       \tau^{\mathrm{v}}\end{bmatrix}.
       ```                                      |     | (5) |

Let $`x_{1}^{\mathrm{a}}`$ and $`x_{1}^{\mathrm{v}}`$ denote the clean
action tokens and future visual tokens, respectively, and let
$`x_{0}^{\mathrm{a}}`$ and $`x_{0}^{\mathrm{v}}`$ denote their
corresponding Gaussian noise. The noisy joint token is constructed by
linearly interpolating between noise and data for each modality:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
x_{\tau}=\tau\begin{bmatrix}x_{1}^{\mathrm{a}}\\
x_{1}^{\mathrm{v}}\end{bmatrix}+(1-\tau)\begin{bmatrix}x_{0}^{\mathrm{a}}\\
x_{0}^{\mathrm{v}}\end{bmatrix}.
``` |  | (6) |

The corresponding ground-truth velocity is given by

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\nu_{\tau}=\frac{dx_{\tau}}{d\tau}=\begin{bmatrix}x_{1}^{\mathrm{a}}-x_{0}^{\mathrm{a}}\\
x_{1}^{\mathrm{v}}-x_{0}^{\mathrm{v}}\end{bmatrix}.
``` |  | (7) |

Finally, GigaWorld-Policy-0.5 is optimized by regressing the
model-predicted velocity field to the ground-truth flow velocity:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}=\mathbb{E}_{x_{1},x_{0},\tau}\left[\left\|g_{\theta}\left(x_{\tau},\tau\mid o_{t}^{\mathrm{comp}},s_{t},l\right)-\nu_{\tau}\right\|_{2}^{2}\right].
``` |  | (8) |

### 3.2 Model Architecture

Fig.
[2](#S3.F2 "Figure 2 ‣ 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch")
illustrates the MoT-based architecture of GigaWorld-Policy-0.5. The
model takes visual observations, proprioceptive states, action chunks,
and language instructions as inputs. For visual inputs, we use the
visual VAE from Wan \[[41](#bib.bib2)\] to encode the composite
observations $`o^{\mathrm{comp}}`$ into visual latent tokens. For
non-visual inputs, robot states and action chunks are mapped to the
visual hidden dimension via multi-layer perceptrons (MLPs), producing
state tokens and action tokens. The language instruction is encoded by
umT5 \[[10](#bib.bib49)\], whose text embeddings are used as
conditioning signals for instruction-following control.

Given these tokens, GigaWorld-Policy-0.5 follows the action-centered
world action framework but replaces the fully shared Transformer
backbone with an MoT structure. The MoT architecture separates the two
central modeling components of action-centered WAMs, namely visual
dynamics modeling and action generation, into a visual expert and an
action expert. The visual expert processes current and future visual
tokens and models scene evolution in the latent visual space, while the
action expert focuses on action-token denoising and action-chunk
prediction. Each expert is equipped with its own cross-attention and
feed-forward network (FFN) modules, enabling modality-specific language
conditioning and nonlinear transformation. These two experts are
connected through multi-modal self-attention, enabling information
exchange across visual, state, and action tokens while preserving a
unified token sequence for joint world action modeling.

In particular, the multi-modal self-attention module follows
GigaWorld-Policy \[[54](#bib.bib27)\] by adopting an action-centered
causal mask to regulate cross-modal information flow. Specifically,
action tokens are allowed to attend to the current visual tokens, state
tokens, and language conditioning, but are prevented from attending to
future visual tokens. Future visual tokens, on the other hand, can
attend to the current context and action tokens, enabling
action-conditioned future visual prediction. This causal attention mask
prevents information leakage from future observations into action
prediction, while allowing future visual dynamics to provide dense
training-time supervision. At inference time, future visual tokens can
be omitted, and the model directly decodes action tokens for low-latency
control.

For parameter initialization, GigaWorld-Policy-0.5 initializes the
visual expert with pretrained visual weights from GigaWorld-1
\[[38](#bib.bib55)\], a large-scale pretrained world model. This
provides the visual expert with a world-model prior for visual dynamics
modeling, allowing the model to inherit general knowledge about scene
evolution before being adapted to robot-specific observations and
interactions. For the newly introduced action expert, we initialize its
parameters from the corresponding visual-expert weights. For parameters
whose dimensions are not aligned, we initialize the action-expert
parameters by taking the leading $`n`$ dimensions of the corresponding
visual-expert weights, where $`n`$ denotes the target dimensionality of
the action-expert parameter.

### 3.3 Training Pipeline

The training pipeline of GigaWorld-Policy-0.5 consists of two stages:
robot-data pretraining and target-robot post-training. Before these two
stages, the visual expert is initialized from GigaWorld-1
\[[38](#bib.bib55)\], which is pretrained on over ten thousand hours of
video data. This initialization provides GigaWorld-Policy-0.5 with a
general world-model prior for visual dynamics modeling.

In the pretraining stage, we train GigaWorld-Policy-0.5 on 2K hours of
filtered open-source robot data \[[7](#bib.bib50), [26](#bib.bib51),
[49](#bib.bib52), [35](#bib.bib53)\] and internally collected real-robot
data. This stage adapts the pretrained world-model prior to
robot-centric scenarios, including robot embodiments, manipulation
workspaces, camera viewpoints, and interaction patterns. The model
learns to predict future visual evolution under robot-relevant
observations and actions, improving its alignment with robotic control
settings before target-domain specialization. Notably, to better model
the relationship between actions and visual dynamics, we also
incorporate action-conditioned world-model training during this stage,
where future visual evolution is predicted under robot-relevant
observations and actions. This encourages the model to learn how robot
actions affect scene changes before target-domain specialization.

In the post-training stage, GigaWorld-Policy-0.5 is trained on target
real-robot trajectories that contain aligned observations, language
instructions, robot states, and action sequences. The model jointly
optimizes action prediction and future visual dynamics modeling under
the action-centered causal structure. The action objective teaches the
model to generate instruction-conditioned action chunks for closed-loop
control, while the future-visual objective provides auxiliary
supervision that encourages action representations to remain consistent
with plausible scene evolution. During deployment, we use the
action-only inference path: future visual tokens are not generated, and
the model directly decodes actions conditioned on the current
observation, state, and instruction.

![Refer to caption](2607.13960v3/demo-canjv.png)

Figure 3: Real-world demonstration of GigaWorld-Policy-0.5 on the
Tableware Arrangement task.

### 3.4 Inference Acceleration

To support real-world deployment on edge computing platforms, we
optimize the inference stack through KV caching, graph compilation, and
a lightweight C++ runtime. During autoregressive action generation, the
visual and language context remains unchanged across decoding steps; we
therefore cache the key-value states of all attention layers after
encoding the input context and reuse them for subsequent action-token
predictions, avoiding redundant attention computation. We further apply
torch.compile to optimize the inference graph, which reduces
Python-level dispatch overhead and enables backend-level optimizations
such as operator fusion and more efficient memory scheduling. Finally,
we deploy the optimized policy in a C++ runtime that integrates image
preprocessing, tensor construction, model execution, KV-cache
management, and action post-processing into a unified native pipeline.
This design removes the Python dependency, reduces runtime overhead, and
simplifies integration with robot-control middleware, enabling
low-latency closed-loop policy execution on edge devices.

![Refer to caption](2607.13960v3/demo-heatfood.png)

Figure 4: Real-world demonstration of GigaWorld-Policy-0.5 on the Food
Heating task.

## 4 Experiments

Evaluation Metrics. We primarily report the Success Rate (SR) to
evaluate task completion performance. For real-world experiments, we
evaluate gripper-based manipulation under two settings: text following
and long-horizon task execution. For text following, we use a graded SR
to better capture partial task completion: each trial is scored by four
equally weighted stages, with 0.25 assigned to each stage, including
reaching the target object, grasping it, moving to the target location,
and successfully placing it. For long-horizon tasks, SR follows the same
binary definition as in simulation, where SR = 1 only if the full task
sequence is completed and SR = 0 otherwise.

Baselines. We compare GigaWorld-Policy-0.5 with several representative
baselines spanning two major paradigms. For VLM-based VLA methods, we
include $`\pi_{0.5}`$ \[[16](#bib.bib16)\], which uses a large
vision-language backbone to align visual observations with language
instructions and decode low-level control actions through an action
head. For WAM-based methods, we include Motus \[[4](#bib.bib20)\],
FastWAM \[[57](#bib.bib26)\], and GigaWorld-Policy \[[54](#bib.bib27)\],
which incorporate visual dynamics or world-modeling objectives to
support policy learning.

Implementation Details. We use the pretrained GigaWorld-1
\[[38](#bib.bib55)\] to initialize the visual expert, enabling the model
to inherit strong visual generative representations. The action expert
is then initialized from the visual expert by copying the corresponding
weights; for layers with mismatched dimensions, we retain the compatible
submatrices and truncate the remaining entries to match the
action-expert configuration. In our implementation, the visual expert
uses a hidden dimension of 3072 and an FFN dimension of 14336, while the
action expert is made more lightweight with a hidden dimension of 1024
and an FFN dimension of 4096. For the action chunk length and
future-observation stride, we follow the settings adopted in
GigaWorld-Policy.

|  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
| Task Name | Text Instruction | $`\pi_{0.5}`$ \[[16](#bib.bib16)\] | Motus \[[4](#bib.bib20)\] | FastWAM \[[57](#bib.bib26)\] | GigaWorld-Policy \[[54](#bib.bib27)\] | GigaWorld-Policy-0.5 |
| Pick the Fruit | Pick the banana and place it into the basket. | 0.88 | 0.93 | 0.83 | 0.90 | 0.95 |
|  | Pick the apple and place it into the basket. | 0.73 | 0.78 | 0.75 | 0.73 | 0.80 |
|  | Pick the lemon and place it into the basket. | 0.68 | 0.75 | 0.73 | 0.78 | 0.83 |
|  | Pick the grape and place it into the basket. | 0.85 | 0.83 | 0.88 | 0.88 | 0.93 |
|  | Pick the avocado and place it into the basket. | 0.68 | 0.70 | 0.75 | 0.73 | 0.78 |
|  | Pick the strawberry and place it into the basket. | 0.78 | 0.80 | 0.73 | 0.78 | 0.85 |
| Average | – | 0.76 | 0.80 | 0.78 | 0.80 | 0.85 |

Table 1: Evaluation of text-following ability on the fruit-picking task.

### 4.1 Real-World Results

We conduct gripper-based manipulation experiments on an AgileX PiPER
6-DoF robotic arm from two complementary perspectives. Text following
evaluates language grounding by executing multiple instructions under
the same scene configuration. Each trial is scored using four equally
weighted stages (0.25 point each): reaching the target object, grasping
the target object, moving to the target placement location, and
successfully placing the object. The final score is averaged over 10
real-world trials. Long-horizon tasks evaluate end-to-end sequential
manipulation, where each task is executed 10 times on the real robot and
only the overall task success rate is reported.

Tab.
[1](#S4.T1 "Table 1 ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch")
and
[2](#S4.T2 "Table 2 ‣ 4.1 Real-World Results ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch")
evaluate text-following ability under variations in object semantics and
target locations. On the fruit-picking task, GigaWorld-Policy-0.5
achieves an average success rate of 0.85, outperforming $`\pi_{0.5}`$
\[[16](#bib.bib16)\], Motus \[[4](#bib.bib20)\], FastWAM
\[[57](#bib.bib26)\], and GigaWorld-Policy \[[54](#bib.bib27)\] by 0.09,
0.05, 0.07, and 0.05, respectively. It consistently obtains the highest
success rate across all six fruit categories, with particularly clear
gains on lemon and avocado, suggesting that the model can reliably
ground fine-grained language descriptions to visually distinct objects.
On the more compositional object-placement task, GigaWorld-Policy-0.5
achieves an average success rate of 0.89, outperforming the strongest
baseline, Motus, by 0.06. This task requires jointly identifying the
referenced object and following the specified destination relation
(e.g., placing an object *on* a plate or *into* a basket). Our method
attains the best result on every instruction and shows notable
improvements for bowl-to-basket and fork-to-basket instructions,
indicating stronger grounding of both object identities and spatial goal
specifications. Overall, the results demonstrate that
GigaWorld-Policy-0.5 follows diverse natural-language instructions more
reliably than prior methods, particularly when task execution requires
precise object selection and compositional object-destination reasoning.

|  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
| Task Name | Text Instruction | $`\pi_{0.5}`$ \[[16](#bib.bib16)\] | Motus \[[4](#bib.bib20)\] | FastWAM \[[57](#bib.bib26)\] | GigaWorld-Policy \[[54](#bib.bib27)\] | GigaWorld-Policy-0.5 |
| Place Objects | Pick up the bowl and place it on the plate. | 0.87 | 0.88 | 0.73 | 0.80 | 0.93 |
|  | Pick up the fork and place it on the plate. | 0.78 | 0.83 | 0.85 | 0.80 | 0.88 |
|  | Pick up the spoon and place it on the plate. | 0.73 | 0.83 | 0.80 | 0.75 | 0.85 |
|  | Pick up the bowl and place it into the basket. | 0.75 | 0.83 | 0.80 | 0.88 | 0.95 |
|  | Pick up the fork and place it into the basket. | 0.65 | 0.75 | 0.68 | 0.78 | 0.85 |
|  | Pick up the spoon and place it into the basket. | 0.80 | 0.85 | 0.75 | 0.83 | 0.88 |
| Average | – | 0.76 | 0.83 | 0.77 | 0.81 | 0.89 |

Table 2: Evaluation of text-following ability on the object-placement
task.

Table
[3](#S4.T3 "Table 3 ‣ 4.1 Real-World Results ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch")
reports end-to-end success rates on three long-horizon manipulation
tasks, each of which requires the policy to execute multiple temporally
dependent steps. GigaWorld-Policy-0.5 achieves the best performance on
every task, attaining an average success rate of 0.80. Compared with the
strongest baseline, it yields an absolute improvement of 0.20 and a
relative improvement of 33%. This substantial gain highlights the
effectiveness of our method in long-horizon task execution, where
successful completion depends on maintaining coherent action sequences
and accurately handling intermediate state transitions. The results
further suggest that the predictive representations acquired through
world modeling provide useful temporal and interaction-aware information
for long-horizon robotic manipulation.

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Task Name | $`\pi_{0.5}`$ \[[16](#bib.bib16)\] | Motus \[[4](#bib.bib20)\] | FastWAM \[[57](#bib.bib26)\] | GigaWorld-Policy \[[54](#bib.bib27)\] | GigaWorld-Policy-0.5 |
| Food Heating | 0.50 | 0.60 | 0.50 | 0.60 | 0.80 |
| Tableware Arrangement | 0.70 | 0.60 | 0.50 | 0.60 | 0.80 |
| Average | 0.60 | 0.60 | 0.50 | 0.60 | 0.80 |

Table 3: Evaluation of end-to-end success rates on long-horizon tasks.

### 4.2 Ablation Studies

We conduct two groups of ablation studies to analyze
GigaWorld-Policy-0.5. The first group focuses on standard component
ablations, where we isolate the contribution of key design choices in
the model and training pipeline. The second group uses the AutoResearch
\[[18](#bib.bib61)\] pipeline to study the effect of hyperparameter
choices, such as learning rate and warmup strategy, under a unified
evaluation protocol.

#### Effect of the mixed AC-WM and WAM pretraining.

We evaluate whether incorporating Action-Conditioned World-Model (AC-WM)
training during pretraining improves downstream policy learning. This
ablation compares two pretraining settings: one using standard WAM
pretraining only, and the other mixing AC-WM and WAM pretraining, where
future visual observations are predicted conditioned on robot actions.
After pretraining, both models are post-trained using the same

policy training recipe and evaluated on the pick the fruit task.

Fig.
[5](#S4.F5 "Figure 5 ‣ Effect of the mixed AC-WM and WAM pretraining. ‣ 4.2 Ablation Studies ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch")
shows that the model pretrained with mixed AC-WM and WAM outperforms the
baseline throughout post-training: it converges faster, attains higher
success rates, and ultimately reaches a success rate of 0.85. Notably,
strong performance emerges at substantially earlier training steps,
indicating that explicitly modeling how robot actions drive visual state
transitions yields more transferable action representations. As a
result, downstream policy learning becomes more sample-efficient,
requiring fewer post-training updates to achieve competitive closed-loop
control performance.

![\[Uncaptioned image\]](2607.13960v3/figures/ablation_acwm.png)

Figure 5: Success rates at different training steps in the AC-WM
ablation study.

#### Effect of MoT architecture.

We evaluate the effect of the MoT architecture on inference efficiency.
The MoT architecture processes action generation and video dynamics
modeling with separate experts. In our design, we reduce the dimensional
configuration of the action expert relative to the video expert, making
the action pathway more lightweight than the video modeling pathway. As
a result, although GigaWorld-Policy-0.5 introduces additional parameters
due to the expert-based architecture, the computation required for
action-only inference is reduced. In addition, this expert-separated
design makes it easier to initialize the video expert from pretrained
video generation models, which provides stronger visual dynamics
representations and further improves downstream policy performance.

As shown in Tab.
[4](#S4.T4 "Table 4 ‣ Effect of MoT architecture. ‣ 4.2 Ablation Studies ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
this architectural design leads to clear inference efficiency gains.
With KV cache and torch compilation enabled, GigaWorld-Policy-0.5
achieves an inference latency of 189 ms on an A100 GPU. Compared with
FastWAM under the same torch-compiled setting, GigaWorld-Policy-0.5
reduces latency from 229 ms to 189 ms, corresponding to a 17.5% speedup.
It also outperforms the VLA-based $`\pi_{0.5}`$ \[[16](#bib.bib16)\] in
inference efficiency, reducing latency from 225 ms to 189 ms. Beyond the
A100 setting, GigaWorld-Policy-0.5 further achieves 110 ms latency on a
local RTX 4090 edge-side setup, matching the inference speed of
$`\pi_{0.5}`$. With a C++ deployment, latency can be further reduced to
85 ms, 23% faster than $`\pi_{0.5}`$ and 53% faster than FastWAM
\[[57](#bib.bib26)\], demonstrating its potential for efficient
real-world deployment.

|  |  |  |  |
|----|----|----|----|
| Method | Latency on A100 (ms) $`\downarrow`$ | Latency on RTX 4090 (ms) $`\downarrow`$ | SR on Real-Robot $`\uparrow`$ |
| $`\pi_{0.5}`$ \[[16](#bib.bib16)\] | 225 | 110 | 0.76 |
| Motus \[[4](#bib.bib20)\] | 3231 | \- | 0.80 |
| FastWAM \[[57](#bib.bib26)\] | 229 | 182 | 0.78 |
| GigaWorld-Policy \[[54](#bib.bib27)\] | 360 | 293 | 0.80 |
| GigaWorld-Policy-0.5 | 189 | 110 | 0.85 |
| w/ C++ deployment | 140 | 85 | 0.85 |

Table 4: Comparison of inference efficiency and real-robot performance.

![Refer to caption](2607.13960v3/figures/autoresearch.png)

Figure 6: AutoResearch hyperparameter search and training progression on
the pick the fruit task. Left: 1K-step pilot sweep over learning rate
and batch-size configurations, where green bars indicate retained
candidates and red bars indicate discarded configurations. Right:
extended training progression under the selected configuration, showing
that the model achieves the best validation action MSE at 30K steps.

#### AutoResearch-driven hyperparameter study.

We use AutoResearch \[[18](#bib.bib61)\] to systematically explore the
training hyperparameters of GigaWorld-Policy-0.5, including the learning
rate and warmup schedule. AutoResearch automates the full optimization
loop, including candidate configuration generation, pilot training,
validation metric collection, candidate selection, and extended
training. This procedure is particularly useful for
GigaWorld-Policy-0.5, as the model jointly optimizes action prediction
and future visual dynamics modeling, and different hyperparameter
choices may affect these two objectives in different ways.

We conduct the study on the pick the fruit task using approximately 3.9
hours of robot demonstration data. The dataset contains 930 episodes, of
which 300 episodes are used for training and 30 episodes are held out
for validation. All candidate runs use the same model initialization,
data split, optimizer, and evaluation protocol. To improve the
efficiency of hyperparameter exploration, AutoResearch first uses short
1K-step pilot runs to rapidly iterate over different candidate
configurations. After identifying promising hyperparameters,
AutoResearch performs longer training runs under the selected
configuration for final model selection.

|  |  |  |  |
|----|----|----|----|
| Learning Rate | Train Action Loss $`\downarrow`$ | Train Visual Loss $`\downarrow`$ | Eval Action MSE $`\downarrow`$ |
| $`3\times 10^{-5}`$ | 0.257300 | 0.172330 | 0.416832 |
| $`4.316\times 10^{-5}`$ | 0.256593 | 0.172893 | 0.449387 |
| $`6\times 10^{-5}`$ | 0.252476 | 0.173318 | 0.409764 |
| $`8\times 10^{-5}`$ | 0.261832 | 0.175986 | 0.461381 |

Table 5: AutoResearch learning-rate sweep. Each candidate is trained for
1K steps under the same setting.

As shown on the left side of Fig.
[6](#S4.F6 "Figure 6 ‣ Effect of MoT architecture. ‣ 4.2 Ablation Studies ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
AutoResearch first sweeps over candidate learning rates using these
1K-step pilot runs. The detailed learning-rate ablation results are
reported in Tab.
[5](#S4.T5 "Table 5 ‣ AutoResearch-driven hyperparameter study. ‣ 4.2 Ablation Studies ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"):
among the tested candidates, $`6\times 10^{-5}`$ achieves the lowest
training action loss of 0.252476 and the best evaluation action MSE of
0.409764, while $`3\times 10^{-5}`$ obtains the lowest training visual
loss of 0.172330. Since action prediction quality is more directly
related to downstream policy execution, AutoResearch selects
$`6\times 10^{-5}`$ as the learning rate for the next-stage search.

AutoResearch then fixes this learning rate and conducts an additional
batch-size sweep. The batch-size search shows that alternative batch
sizes do not outperform the original setting, so AutoResearch retains
the original batch size of 16. Based on these results, AutoResearch
selects $`6\times 10^{-5}`$ and a batch size of 16 as the final
hyperparameter configuration. After fixing these hyperparameters,
AutoResearch further scales up the training by extending the number of
training steps. As shown on the right side of Fig.
[6](#S4.F6 "Figure 6 ‣ Effect of MoT architecture. ‣ 4.2 Ablation Studies ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
the model reaches its best validation performance at 30K training steps.
We therefore use the 30K-step checkpoint for subsequent real-robot
evaluation.

## 5 Conclusion

In this technical report, we introduced GigaWorld-Policy-0.5, an
enhanced action-centered World Action Model built upon GigaWorld-Policy.
GigaWorld-Policy-0.5 preserves the key principle of using future visual
dynamics as training-time supervision while enabling action-only
decoding during inference. On top of this formulation, the model adopts
a Mixture-of-Transformers architecture to separate visual dynamics
modeling from action generation, reducing active computation for
low-latency deployment. We also introduce a mixed pretraining strategy
that combines action-conditioned world modeling with WAM training,
encouraging the model to better capture action-induced visual changes
and improving action modeling. Finally, an agent-based AutoResearch
pipeline is used to make hyperparameter search more systematic and
efficient. Overall, GigaWorld-Policy-0.5 demonstrates a practical path
toward faster and more deployable World Action Models for robot control.
With C++ deployment on a local RTX 4090, it further achieves 85 ms
inference latency, improving the efficiency-performance trade-off of
action-centered WAMs by retaining dense supervision from future visual
dynamics while reducing the cost of action generation at inference time.

## References

- \[1\] N. Agarwal, A. Ali, J. Allen, M. Antolini, A. Aubame, A.
  Azzolini, J. Bai, M. Bala, Y. Balaji, J. Bapst, et al. (2026) Cosmos
  3: omnimodal world models for physical ai. arXiv preprint
  arXiv:2606.02800. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[2\] N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P.
  Chattopadhyay, Y. Chen, Y. Cui, Y. Ding, et al. (2025) Cosmos world
  foundation model platform for physical ai. arXiv preprint
  arXiv:2501.03575. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[3\] H. A. Alhaija, J. Alvarez, M. Bala, T. Cai, T. Cao, L. Cha, J.
  Chen, M. Chen, F. Ferroni, S. Fidler, et al. (2025) Cosmos-transfer1:
  conditional world generation with adaptive multimodal control. arXiv
  preprint arXiv:2503.14492. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§2.1](#S2.SS1.p2.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[4\] H. Bi, H. Tan, S. Xie, Z. Wang, S. Huang, H. Liu, R. Zhao, Y.
  Feng, C. Xiang, Y. Rong, et al. (2026) Motus: a unified latent action
  world model. In Proceedings of the IEEE/CVF Conference on Computer
  Vision and Pattern Recognition, pp. 35101–35113. Cited by:
  [§1](#S1.p3.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§2.2](#S2.SS2.p2.1 "2.2 World Models for Robot Control ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§3.1](#S3.SS1.p3.1 "3.1 Overview of GigaWorld-Policy-0.5 ‣ 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§4.1](#S4.SS1.p2.1 "4.1 Real-World Results ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Table
  1](#S4.T1.5.1.1.4 "In 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Table
  2](#S4.T2.5.1.1.4 "In 4.1 Real-World Results ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Table
  3](#S4.T3.5.1.1.3 "In 4.1 Real-World Results ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Table
  4](#S4.T4.5.3.1 "In Effect of MoT architecture. ‣ 4.2 Ablation Studies ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§4](#S4.p2.1 "4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[5\] J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L.
  Fan, Y. Fang, D. Fox, F. Hu, S. Huang, et al. (2025) Gr00t n1: an open
  foundation model for generalist humanoid robots. arXiv preprint
  arXiv:2503.14734. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[6\] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N.
  Fusai, L. Groom, K. Hausman, B. Ichter, et al. (2024) $`\pi_{0}`$: A
  vision-language-action flow model for general robot control. arXiv
  preprint arXiv:2410.24164. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[7\] Q. Bu, J. Cai, L. Chen, X. Cui, Y. Ding, S. Feng, S. Gao, X.
  He, X. Hu, X. Huang, et al. (2025) Agibot world colosseo: a
  large-scale manipulation platform for scalable and intelligent
  embodied systems. arXiv preprint arXiv:2503.06669. Cited by:
  [§3.3](#S3.SS3.p2.1 "3.3 Training Pipeline ‣ 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[8\] J. Cen, C. Yu, H. Yuan, Y. Jiang, S. Huang, J. Guo, X. Li, Y.
  Song, H. Luo, F. Wang, et al. (2025) Worldvla: towards autoregressive
  action world model. arXiv preprint arXiv:2506.21539. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[9\] J. Chen, K. Wang, K. Chen, S. Chen, F. Gao, W. Tang, Z. Li, W.
  Liu, Z. Yao, B. Li, et al. (2026) LaWAM: latent world action models
  for efficient dynamics-aware robot policies. arXiv preprint
  arXiv:2606.15768. Cited by:
  [§2.2](#S2.SS2.p3.1 "2.2 World Models for Robot Control ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[10\] H. W. Chung, X. Garcia, A. Roberts, Y. Tay, O. Firat, S.
  Narang, and N. Constant (2023) UniMax: fairer and more effective
  language sampling for large-scale multilingual pretraining. In The
  Eleventh International Conference on Learning Representations, Cited
  by:
  [§3.2](#S3.SS2.p1.1 "3.2 Model Architecture ‣ 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[11\] Z. Dong, X. Wang, Z. Zhu, Y. Wang, Y. Wang, Y. Zhou, B.
  Wang, C. Ni, R. Ouyang, W. Qin, et al. (2025) Emma: generalizing
  real-world robot manipulation via generative visual transfer. arXiv
  preprint arXiv:2509.22407. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[12\] Y. Du, S. Yang, B. Dai, H. Dai, O. Nachum, J. Tenenbaum, D.
  Schuurmans, and P. Abbeel (2023) Learning universal policies via
  text-guided video generation. Advances in neural information
  processing systems 36, pp. 9156–9172. Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 World Models for Robot Control ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[13\] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Müller, H.
  Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel, et al. (2024) Scaling
  rectified flow transformers for high-resolution image synthesis. In
  Forty-first international conference on machine learning, Cited by:
  [§3.1](#S3.SS1.p5.1 "3.1 Overview of GigaWorld-Policy-0.5 ‣ 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[14\] P. Intelligence, B. Ai, A. Amin, R. Aniceto, A. Balakrishna, G.
  Balke, K. Black, G. Bokinsky, S. Cao, T. Charbonnier, et al. (2026)
  $`\pi_{0.7}`$: A steerable generalist robotic foundation model with
  emergent capabilities. arXiv preprint arXiv:2604.15483. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[15\] P. Intelligence, A. Amin, R. Aniceto, A. Balakrishna, K.
  Black, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, J. DiCarlo,
  et al. (2025) $`\pi_{0.6}`$: A vla that learns from experience. arXiv
  preprint arXiv:2511.14759. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[16\] P. Intelligence, K. Black, N. Brown, J. Darpinian, K.
  Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et
  al. (2025) $`\pi_{0.5}`$: A vision-language-action model with
  open-world generalization. arXiv preprint arXiv:2504.16054. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§4.1](#S4.SS1.p2.1 "4.1 Real-World Results ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§4.2](#S4.SS2.SSS0.Px2.p2.1 "Effect of MoT architecture. ‣ 4.2 Ablation Studies ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Table
  1](#S4.T1.5.1.1.3 "In 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Table
  2](#S4.T2.5.1.1.3 "In 4.1 Real-World Results ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Table
  3](#S4.T3.5.1.1.2 "In 4.1 Real-World Results ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Table
  4](#S4.T4.5.2.1 "In Effect of MoT architecture. ‣ 4.2 Ablation Studies ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§4](#S4.p2.1 "4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[17\] Z. Jiang, S. Zhou, Y. Jiang, Z. Huang, M. Wei, Y. Chen, T.
  Zhou, Z. Guo, H. Lin, Q. Zhang, et al. (2026) Wovr: world models as
  reliable simulators for post-training vla policies with rl. arXiv
  preprint arXiv:2602.13977. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[18\] A. Karpathy (2026) Autoresearch. External Links:
  [Link](https://github.com/karpathy/autoresearch) Cited by:
  [§1](#S1.p7.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§4.2](#S4.SS2.SSS0.Px3.p1.1 "AutoResearch-driven hyperparameter study. ‣ 4.2 Ablation Studies ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§4.2](#S4.SS2.p1.1 "4.2 Ablation Studies ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[19\] S. Kim, S. Lee, and M. Cho (2025) FreeAction: training-free
  techniques for enhanced fidelity of trajectory-to-video generation.
  arXiv preprint arXiv:2509.24241. Cited by:
  [§2.1](#S2.SS1.p2.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[20\] Y. LeCun (2022) A path towards autonomous machine intelligence
  version 0.9. 2, 2022-06-27. Open Review. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[21\] H. Li, I. Zhang, R. Ouyang, X. Wang, Z. Zhu, Z. Yang, Z.
  Zhang, B. Wang, C. Ni, W. Qin, et al. (2025) Mimicdreamer: aligning
  human and robot demonstrations for scalable vla training. arXiv
  preprint arXiv:2509.22199. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[22\] L. Li, Q. Zhang, Y. Luo, S. Yang, R. Wang, F. Han, M. Yu, Z.
  Gao, N. Xue, X. Zhu, et al. (2026) Causal world modeling for robot
  control. arXiv preprint arXiv:2601.21998. Cited by:
  [§1](#S1.p3.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§2.2](#S2.SS2.p2.1 "2.2 World Models for Robot Control ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§2.2](#S2.SS2.p3.1 "2.2 World Models for Robot Control ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[23\] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le (2022)
  Flow matching for generative modeling. arXiv preprint
  arXiv:2210.02747. Cited by:
  [§3.1](#S3.SS1.p5.1 "3.1 Overview of GigaWorld-Policy-0.5 ‣ 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[24\] F. Liu, S. Zhang, X. Wang, Y. Wei, H. Qiu, Y. Zhao, Y.
  Zhang, Q. Ye, and F. Wan (2025) Timestep embedding tells: it’s time to
  cache for video diffusion model. In Proceedings of the Computer Vision
  and Pattern Recognition Conference, pp. 7353–7363. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[25\] L. Liu, X. Wang, G. Zhao, K. Li, W. Qin, J. Zhu, J. Qiu, G.
  Huang, and Z. Su (2026) RoboTransfer: controllable geometry-consistent
  video diffusion for manipulation policy transfer. In Proceedings of
  the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
  pp. 1410–1420. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§2.1](#S2.SS1.p2.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[26\] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su,
  and J. Zhu (2025) Rdt-1b: a diffusion foundation model for bimanual
  manipulation. In International Conference on Learning Representations,
  Vol. 2025, pp. 29982–30009. Cited by:
  [§3.3](#S3.SS3.p2.1 "3.3 Training Pipeline ‣ 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[27\] H. Luo, W. Zhang, Y. Feng, S. Zheng, H. Xu, C. Xu, Z. Xi, Y.
  Fu, and Z. Lu (2026) Being-h0. 7: a latent world-action model from
  egocentric videos. arXiv preprint arXiv:2605.00078. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[28\] J. Lv, H. Li, J. Li, F. Kong, Y. Wang, P. Yi, Y. Nie, X.
  Wang, Z. Zhu, C. Ni, et al. (2026) ViVa: a video-generative value
  model for robot reinforcement learning. arXiv preprint
  arXiv:2604.08168. Cited by:
  [§1](#S1.p3.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[29\] T. Ma, J. Zheng, Z. Wang, C. Jiang, A. Cui, J. Liang, and S.
  Yang (2026) Dit4dit: jointly modeling video dynamics and actions for
  generalizable robot control. arXiv preprint arXiv:2603.10448. Cited
  by:
  [§2.2](#S2.SS2.p3.1 "2.2 World Models for Robot Control ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[30\] C. Ni, C. Chen, X. Wang, Z. Zhu, W. Zheng, B. Wang, T. Chen, G.
  Zhao, H. Li, Z. Dong, et al. (2026) Swiftvla: unlocking spatiotemporal
  dynamics for lightweight vla models at minimal overhead. In
  Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
  Recognition, pp. 13474–13485. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§2.1](#S2.SS1.p1.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[31\] C. Ni, G. Zhao, X. Wang, Z. Zhu, W. Qin, X. Chen, G. Jia, G.
  Huang, and W. Mei (2025) Recondreamer-rl: enhancing reinforcement
  learning via diffusion-based scene reconstruction. arXiv preprint
  arXiv:2508.08170. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[32\] J. Pai, L. Achenbach, V. Montesinos, B. Forrai, O. Mees, and E.
  Nava (2025) Mimic-video: video-action models for generalizable robot
  control beyond vlas. arXiv preprint arXiv:2512.15692. Cited by:
  [§2.2](#S2.SS2.p3.1 "2.2 World Models for Robot Control ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[33\] Y. Shen, F. Wei, Z. Du, Y. Liang, Y. Lu, J. Yang, N. Zheng,
  and B. Guo (2026) Videovla: video generators can be generalizable
  robot manipulators. Advances in neural information processing systems
  38, pp. 95597–95621. Cited by:
  [§1](#S1.p4.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§2.2](#S2.SS2.p2.1 "2.2 World Models for Robot Control ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[34\] Y. Su, S. Chen, H. Shi, M. Liu, Z. Zhang, N. Huang, W.
  Zhong, Z. Zhu, Y. Liu, and X. Liu (2026) World guidance: world
  modeling in condition space for action generation. arXiv preprint
  arXiv:2602.22010. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[35\] H. Tan, Y. Feng, X. Mao, S. Huang, G. Liu, Z. Hao, H. Su,
  and J. Zhu (2025) Anypos: automated task-agnostic actions for bimanual
  manipulation. arXiv preprint arXiv:2507.12768. Cited by:
  [§3.3](#S3.SS3.p2.1 "3.3 Training Pipeline ‣ 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[36\] G. Team, B. Wang, B. Li, C. Ni, G. Huang, G. Zhao, H. Li, J.
  Li, J. Lv, J. Liu, et al. (2026) Gigabrain-0.5 m\*: a vla that learns
  from world model-based reinforcement learning. arXiv preprint
  arXiv:2602.12099. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[37\] G. Team, A. Ye, B. Wang, C. Ni, G. Huang, G. Zhao, H. Li, J.
  Li, J. Zhu, L. Feng, et al. (2025) Gigabrain-0: a world model-powered
  vision-language-action model. arXiv preprint arXiv:2510.19430. Cited
  by:
  [§1](#S1.p2.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[38\] G. Team, A. Ma, B. Wang, B. Li, C. Ni, G. Li, G. Huang, G.
  Zhao, H. Li, H. Li, et al. (2026) GigaWorld-1: a roadmap to build
  world models for robot policy evaluation. arXiv preprint
  arXiv:2607.02642. Cited by:
  [§3.2](#S3.SS2.p4.1 "3.2 Model Architecture ‣ 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§3.3](#S3.SS3.p1.1 "3.3 Training Pipeline ‣ 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§4](#S4.p3.1 "4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[39\] G. Team, A. Ye, B. Wang, C. Ni, G. Huang, G. Zhao, H. Li, J.
  Zhu, K. Li, M. Xu, et al. (2025) Gigaworld-0: world models as data
  engine to empower embodied ai. arXiv preprint arXiv:2511.19861. Cited
  by:
  [§1](#S1.p1.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§2.1](#S2.SS1.p1.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§2.1](#S2.SS1.p2.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[40\] M. Team, C. Xiang, F. Bao, H. Liu, H. Tan, H. Bi, J. Li, J.
  Liu, J. Pang, K. Jing, et al. (2026) Motubrain: an advanced world
  action model for robot control. arXiv preprint arXiv:2604.27792. Cited
  by:
  [§1](#S1.p3.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§2.2](#S2.SS2.p2.1 "2.2 World Models for Robot Control ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[41\] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C. Xie, D. Chen, F.
  Yu, H. Zhao, J. Yang, et al. (2025) Wan: open and advanced large-scale
  video generative models. arXiv preprint arXiv:2503.20314. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§3.2](#S3.SS2.p1.1 "3.2 Model Architecture ‣ 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[42\] B. Wang, X. Meng, X. Wang, Z. Zhu, A. Ye, Y. Wang, Z. Yang, C.
  Ni, G. Huang, and X. Wang (2025) Embodiedreamer: advancing
  real2sim2real transfer for policy training via embodied world
  modeling. arXiv preprint arXiv:2507.05198. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[43\] B. Wang, R. Ouyang, X. Wang, Z. Zhu, G. Zhao, C. Ni, X.
  Zhang, G. Huang, Y. Ren, L. Liu, et al. (2025) Humandreamer-x:
  photorealistic single-image human avatars reconstruction via gaussian
  restoration. arXiv preprint arXiv:2504.03536. Cited by:
  [§2.1](#S2.SS1.p3.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[44\] B. Wang, X. Wang, Y. Li, Z. Zhu, Y. Chang, A. Ye, G. Zhao, C.
  Ni, G. Huang, Y. Ren, et al. (2026) ReconPhys: reconstruct appearance
  and physical attributes from single video. arXiv preprint
  arXiv:2604.07882. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[45\] X. Wang, K. Zhao, F. Liu, J. Wang, G. Zhao, X. Bao, Z. Zhu,
  and Y. Zhang (2026) EgoVid-5m: a large-scale video-action dataset for
  egocentric videos generation. Advances in Neural Information
  Processing Systems 38. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[46\] X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu (2024)
  Drivedreamer: towards real-world-drive world models for autonomous
  driving. In European conference on computer vision, pp. 55–72. Cited
  by:
  [§1](#S1.p1.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[47\] X. Wang, Z. Zhu, G. Huang, B. Wang, X. Chen, and J. Lu (2024)
  Worlddreamer: towards general world models for video generation via
  predicting masked tokens. arXiv preprint arXiv:2401.09985. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[48\] H. Wu, Y. Jing, C. Cheang, G. Chen, J. Xu, X. Li, M. Liu, H.
  Li, and T. Kong (2024) Unleashing large-scale video generative
  pre-training for visual robot manipulation. In International
  Conference on Learning Representations, Vol. 2024, pp. 10641–10662.
  Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 World Models for Robot Control ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[49\] K. Wu, C. Hou, J. Liu, Z. Che, X. Ju, Z. Yang, M. Li, Y.
  Zhao, Z. Xu, G. Yang, et al. (2024) Robomind: benchmark on
  multi-embodiment intelligence normative data for robot manipulation.
  arXiv preprint arXiv:2412.13877. Cited by:
  [§3.3](#S3.SS3.p2.1 "3.3 Training Pipeline ‣ 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[50\] W. Wu, F. Lu, Y. Wang, S. Yang, S. Liu, F. Wang, Q. Zhu, H.
  Sun, Y. Wang, S. Ma, et al. (2026) A pragmatic vla foundation model.
  arXiv preprint arXiv:2601.18692. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[51\] J. Xiang, G. Liu, Y. Gu, Q. Gao, Y. Ning, Y. Zha, Z. Feng, T.
  Tao, S. Hao, Y. Shi, et al. (2024) Pandora: towards general world
  model with natural language actions and video states. arXiv preprint
  arXiv:2406.09455. Cited by:
  [§2.1](#S2.SS1.p2.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[52\] H. Yan, Z. Zhong, J. Zhu, J. He, W. Yuan, W. Song, X. Gong, Y.
  Cai, G. Zhao, X. Yan, et al. (2026) S-vam: shortcut video-action model
  by self-distilling geometric and semantic foresight. arXiv preprint
  arXiv:2603.16195. Cited by:
  [§2.2](#S2.SS2.p3.1 "2.2 World Models for Robot Control ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[53\] A. Ye, W. Ke, X. Wang, X. Chen, C. Ni, G. Zhao, B. Wang, Z.
  Zhu, J. Xie, and D. Zhang (2026) HALO-wa: hybrid-attention
  latent-guided online reinforcement learning for world-action models.
  arXiv preprint arXiv:2607.04265. Cited by:
  [§1](#S1.p3.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[54\] A. Ye, B. Wang, C. Ni, G. Huang, G. Zhao, H. Li, H. Li, J.
  Li, J. Lv, J. Liu, et al. (2026) GigaWorld-policy: an efficient
  action-centered world–action model. arXiv preprint arXiv:2603.17240.
  Cited by:
  [§1](#S1.p5.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§2.2](#S2.SS2.p2.1 "2.2 World Models for Robot Control ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§2.2](#S2.SS2.p4.1 "2.2 World Models for Robot Control ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Figure
  2](#S3.F2 "In 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Figure
  2](#S3.F2.5 "In 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§3.1](#S3.SS1.p1.1 "3.1 Overview of GigaWorld-Policy-0.5 ‣ 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§3.1](#S3.SS1.p4.1 "3.1 Overview of GigaWorld-Policy-0.5 ‣ 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§3.2](#S3.SS2.p3.1 "3.2 Model Architecture ‣ 3 Method ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§4.1](#S4.SS1.p2.1 "4.1 Real-World Results ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Table
  1](#S4.T1.5.1.1.6 "In 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Table
  2](#S4.T2.5.1.1.6 "In 4.1 Real-World Results ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Table
  3](#S4.T3.5.1.1.5 "In 4.1 Real-World Results ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Table
  4](#S4.T4.5.5.1 "In Effect of MoT architecture. ‣ 4.2 Ablation Studies ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§4](#S4.p2.1 "4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[55\] A. Ye, Z. Zhang, B. Wang, X. Wang, D. Zhang, and Z. Zhu (2025)
  Vla-r1: enhancing reasoning in vision-language-action models. arXiv
  preprint arXiv:2510.01623. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[56\] S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S.
  Indupuru, Y. L. Tan, C. Zhu, J. Xiang, et al. (2026) World action
  models are zero-shot policies. arXiv preprint arXiv:2602.15922. Cited
  by:
  [§1](#S1.p4.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[57\] T. Yuan, Z. Dong, Y. Liu, and H. Zhao (2026) Fast-wam: do world
  action models need test-time future imagination?. arXiv preprint
  arXiv:2603.16666. Cited by:
  [§1](#S1.p5.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§2.2](#S2.SS2.p4.1 "2.2 World Models for Robot Control ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§4.1](#S4.SS1.p2.1 "4.1 Real-World Results ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§4.2](#S4.SS2.SSS0.Px2.p2.1 "Effect of MoT architecture. ‣ 4.2 Ablation Studies ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Table
  1](#S4.T1.5.1.1.5 "In 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Table
  2](#S4.T2.5.1.1.5 "In 4.1 Real-World Results ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Table
  3](#S4.T3.5.1.1.4 "In 4.1 Real-World Results ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [Table
  4](#S4.T4.5.4.1 "In Effect of MoT architecture. ‣ 4.2 Ablation Studies ‣ 4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"),
  [§4](#S4.p2.1 "4 Experiments ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[58\] A. Zhai, B. Liu, B. Fang, C. Cai, E. Ma, E. Yin, H. Wang, H.
  Zhou, J. Wang, L. Shi, et al. (2025) Igniting vlms toward the embodied
  space. arXiv preprint arXiv:2509.11766. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[59\] J. Zhang, X. Chen, A. Chen, C. Lv, D. Li, G. Zhou, H. Yin, H.
  Yuan, H. Li, J. Li, et al. (2026) Qwen-robotworld technical report:
  unifying embodied world modeling through language-conditioned video
  generation. arXiv preprint arXiv:2606.17030. Cited by:
  [§2.1](#S2.SS1.p2.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[60\] G. Zhao, C. Ni, X. Wang, Z. Zhu, X. Zhang, Y. Wang, G.
  Huang, X. Chen, B. Wang, Y. Zhang, et al. (2025) Drivedreamer4d: world
  models are effective data machines for 4d driving scene
  representation. In Proceedings of the computer vision and pattern
  recognition conference, pp. 12015–12026. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[61\] G. Zhao, X. Wang, C. Ni, Z. Zhu, W. Qin, G. Huang, and X.
  Wang (2025) Recondreamer++: harmonizing generative and reconstructive
  models for driving scene representation. In Proceedings of the
  IEEE/CVF International Conference on Computer Vision, pp. 26718–26728.
  Cited by:
  [§2.1](#S2.SS1.p3.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[62\] G. Zhao, X. Wang, Z. Zhu, X. Chen, G. Huang, X. Bao, and X.
  Wang (2025) Drivedreamer-2: llm-enhanced world models for diverse
  driving video generation. In Proceedings of the AAAI Conference on
  Artificial Intelligence, Vol. 39, pp. 10412–10420. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[63\] G. Zhao, Y. Wang, X. Wang, Z. Zhu, T. Yu, G. Huang, Y. Zai, J.
  Jiao, C. Xue, X. Wang, et al. (2026) UniDriveDreamer: a single-stage
  multimodal world model for autonomous driving. arXiv preprint
  arXiv:2602.02002. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[64\] S. Zhou, Y. Du, J. Chen, Y. Li, D. Yeung, and C. Gan (2024)
  RoboDreamer: learning compositional world models for robot
  imagination. In Proceedings of the 41st International Conference on
  Machine Learning, pp. 61885–61896. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[65\] Y. Zhou, X. Wang, H. Shao, L. Wang, G. Zhao, J. Shao, J.
  Zhu, T. Yu, Z. Zhu, G. Huang, et al. (2026) Drivedreamer-policy: a
  geometry-grounded world-action model for unified generation and
  planning. arXiv preprint arXiv:2604.01765. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[66\] C. Zhu, R. Yu, S. Feng, B. Burchfiel, P. Shah, and A.
  Gupta (2025) Unified world models: coupling video and action diffusion
  for pretraining on large robotic datasets. arXiv preprint
  arXiv:2504.02792. Cited by:
  [§2.2](#S2.SS2.p2.1 "2.2 World Models for Robot Control ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[67\] H. Zhu, Y. Wang, J. Zhou, W. Chang, Y. Zhou, Z. Li, J. Chen, C.
  Shen, J. Pang, and T. He (2025) Aether: geometric-aware unified world
  modeling. In Proceedings of the IEEE/CVF International Conference on
  Computer Vision, pp. 8535–8546. Cited by:
  [§2.1](#S2.SS1.p2.1 "2.1 World Models as Data Engines for Robotics ‣ 2 Related Work ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
- \[68\] Z. Zhu, X. Wang, W. Zhao, C. Min, B. Li, N. Deng, M. Dou, Y.
  Wang, B. Shi, K. Wang, et al. (2024) Is sora a world simulator? a
  comprehensive survey on general world models and beyond. arXiv
  preprint arXiv:2405.03520. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch").
````
