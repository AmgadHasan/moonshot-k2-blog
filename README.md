# moonshot-k2-blog
My recap of Moonshot AI's blog post about the Kimi-K2 LLM.
Moonshot said the technical report is "coming soon".
AI labs don't have a good track record of keeping promises, so I'll make use of what knowledge is available for now instead of waiting for an unspecified (and potentially infinite) amount.
If they do release a detailed technical report, it might be a good chance to flesh this repo even more.

# Architecture
## 1. Design Principles
#### (aka how to design a modern LLM arch.)
Before training even started, they ran a mountain of scaling-law experiments on architecture variants.
Every single variant that differed from DeepSeek V3 failed to beat it — at best they tied.

So the question became: “Should we pick an inferior but different architecture just to be different?”
The answer was no. DSv3’s structure is battle-tested at scale; their “new ideas” weren’t
With two huge variables already in play (Muon optimizer + larger param count), we didn’t want to add an unverified third variable.

Constraint #1: Inherit DSv3’s architecture wholesale, then tune only the structural hyper-params.
Constraint #2: Cost ceiling. DSv3 is at the upper limit of what they can afford; K2 must be similar. 

Hence the design task boiled down to:

**Within the DSv3 skeleton, find parameters that keep train/inference cost flat while pushing loss significantly lower.**

## Changes

### 1. Sparsity scaling law
"With fixed activated params, simply increasing total MoE params still obeys the scaling law—no overfitting observed" (internal research by Moonshot pre-training team)

Action: Increase number of total experts to `384`

Downside: Model is now bigger. If we use split the model across `256` nodes it now needs +2.5 GB per node.
| Model | EP rank load        | MLP weight |
| ----- | ------------------- | ---------- |
| DSv3  | 2 routed + 1 shared | ~7.5 GB    |
| K2    | 3 routed + 1 shared | ~10 GB     |

### 2. Less attention heads
MoE just got 50 % more expensive—can we claw it back elsewhere?
DeepSeek doubled heads vs classic MHA to maximize bandwidth utilization, but that hurt latency in both prefill & decode.

Action: Reduce the number of attention heads to `64`.

Result: 
1. Cutting heads halves the quadratic term—huge win for long sequences (k2's bread & butter: agents, vibe coding).
2. QKVO projection params drop from 10 B → 5 B, shaving FLOPs again.

Ablation showed the negative impact on loss is tiny compared to MoE’s gain. Heads=64 locked in.

### 3. No expert grouping
Grouping helps when multiple experts sit on one GPU, balancing work at device level.
At our scale we must use large EP, so each device holds ≤1 expert.
Balancing moves to the node level, yet worst-case imbalance inside a node still kills latency.
Thus dynamic expert re-allocation + redundancy (EPLB) outweigh grouping.
A freer router also enlarges the combinatorial space → better model quality.


## 2. Architecture Details:
1. Sparse Mixture of Expert (SMoE)
2. 1000-A32: 1 trilion total tokens, 32 of which are activated in each forward pass
3. Very similar to Deepseek V3 & R1
4. Has less attention heads but more total experts
5. Bigger vocab
<img width="680" height="356" alt="image" src="https://pbs.twimg.com/media/Gvqs56pXIAEVI73.jpg?name=orig" />


# Motivation & Prinicples
## 1. Agentic Intelligence

In RL, there are three key components: algorithm, environment, and priors.
Without good priors, the agent is just randomly guessing the action it takes. This results in low rewards overall and very weak feedback signal.
LLM Pre-training is the crucial foundation for establishing the priors that makes reinforcement learning (RL) exploration tractable, efficient, and generalizable.

Caveat: "human data is a finite "fossil fuel" and its growth is lagging far behind the pace of compute.
Token efficiency is important: given a fixed sized dataset, how can we develop "smarter" models? (hints: use better optimizer like muonclip)

## 2.  Post-training
We're in the "Era of Experience" (David Silver, Richard Sutton, 2025): LLMs increasingly learn from their own self-generated interactions, receiving rewards that free them from the limits of human data and enable them to surpass human capabilities. Authors believe this unlocks superhuman intelligence as we aren't bottlenecked by our mere human brains.
Examples:
1. AlphaProof: Inititially trained on ~100k formal proofs by human experts -> generate ~100M more through continual interaction with a formal proving system. 
2. Deepseek R1: Use verifiable problems with RL to let the model learn from its attempted solutions.

# Contributions
### 1. MuonClip Optimizer
For a finite/fixed pretraining dataset and a fixed model configuration, a more token-efficient optimizer generates more intelligence.
There has been non-stop research to improve upon AdamW since 2015-ish.

The most prominent one so far was Muon by [Keller Jordan](https://kellerjordan.github.io/posts/muon/) which broke the record for the most token efficient way to train nanogpt.

Moonshot's Moonlight has demonstrated that the Muon optimizer substantially outperforms the widely-used AdamW optimizer for LLM training.

Kimi K2 was designed to further scale up Moonlight, which employs an architecture similar to DeepSeek-V3. Based on scaling-law analysis, we reduce the number of heads for long-context efficiency, and increase MoE sparsity for greater token efficiency. While scaling up, we encountered a persistent challenge: training instability caused by exploding attention logits, an issue that occurs more frequently with Muon but less with AdamW in our experiments. Existing solutions such as logit soft-capping and query-key normalization were found inadequate.

To address this, we introduce the MuonClip optimizer that improves Muon with our proposed qk-clip technique. Specifically, qk-clip stabilizes training by directly rescaling the weight matrices of the query and key projections after Muon updates, thus controlling the scale of attention logits at the source. Concretely, the query and key projections are scaled as follows:

$$q_i = \eta^\alpha W_q x_i$$
$$k_i = \eta^{1-\alpha} W_k x_i$$

where $\alpha$ is a balancing hyperparameter, so the attention logit becomes:

$$(\eta^\alpha q_i)^\top (\eta^{1-\alpha} k_j) = \eta q_i^\top k_j$$

The adaptive factor $\eta$ (with threshold $t$) is set after every step based on the max attention logit in this step:

$$\eta = \min\left(\frac{t}{\max_{i,j}(q_i^\top k_j)}, 1\right)$$

Our experiments show that MuonClip effectively prevents logit explosions while maintaining downstream task performance.

Kimi K2 (1T-32A) was pre-trained on 15.5T tokens using MuonClip with **zero training spike**, demonstrating MuonClip as a robust solution for stable, large-scale LLM training.

Notice also the second loss dip at ~11T tokens!
<img width="1919" height="1152" alt="image" src="https://github.com/user-attachments/assets/379114e3-cbb2-456a-9087-91bd98f85eb3" />

### 2. Large-scale agentic data synthesis
How to teach the model sophisticated tool-use capabilities? 
They developed a comprehensive pipeline inspired by ACEBench. It simulates real-world tool-using scenarios at scale. 

1. Systematically evolve hundreds of domains with thousands of tools (real MCP and synthetic).
2. Generate hundreds of agents with diverse tool sets.
3. Create simulated environments and user agents (i.e agents that act as a user)
4. Use rubric-based tasks for consistent evaluation.
5. Simulate multi-turn tool-use scenarios with agents interacting in environments and with user agents.
6. Employ an LLM judge to evaluate results against task rubrics, filtering for high-quality training data.

<img width="2482" height="770" alt="workflow-Cqznd7Jl" src="https://github.com/user-attachments/assets/ba8a6071-3d19-4dfc-9a21-805f7de755da" />


# Trivias
Xai [rejected](https://xcancel.com/kellerjordan0/status/1893868235381961140) the idea behind moun and said this is wrong.

<img width="1500" height="783" alt="tweet-1893868235381961140" src="https://github.com/user-attachments/assets/cfb5e1d1-554d-4f4e-a937-a2301984ee01" />

How to short xAI?

# References
1. [Kimi K2 Blog post](https://moonshotai.github.io/Kimi-K2/)
2. [Sebastian Raschka's tweet](https://xcancel.com/rasbt/status/1944056316424577525)
3. English [translation](https://www.kimi.com/share/d1q8l75e09n7its6e7jg) of Moonshot engineer Shaowei Liu's [writing](https://www.zhihu.com/question/1927140506573435010/answer/1927892108636849910) 
