# moonshot-k2-blog
My recap of Moonshot AI's blog post about the Kimi-K2 LLM

## Motivation & Prinicples
### 1. Agentic Intelligence

In RL, there are three key components: algorithm, environment, and priors.
Without good priors, the agent is just randomly guessing the action it takes. This results in low rewards overall and very weak feedback signal.
LLM Pre-training is the crucial foundation for establishing the priors that makes reinforcement learning (RL) exploration tractable, efficient, and generalizable.

Caveat: "human data is a finite "fossil fuel" and its growth is lagging far behind the pace of compute.
Token efficiency is important: given a fixed sized dataset, how can we develop "smarter" models? (hints: use better optimizer like muonclip)

### 2.  Post-training
We're in the "Era of Experience" (David Silver, Richard Sutton, 2025): LLMs increasingly learn from their own self-generated interactions, receiving rewards that free them from the limits of human data and enable them to surpass human capabilities. Authors believe this unlocks superhuman intelligence as we aren't bottlenecked by our mere human brains.
Examples:
1. AlphaProof: Inititially trained on ~100k formal proofs by human experts -> generate ~100M more through continual interaction with a formal proving system. 
2. Deepseek R1: Use verifiable problems with RL to let the model learn from its attempted solutions.

## Solutions
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

# Trivias
Xai [rejected](https://xcancel.com/kellerjordan0/status/1893868235381961140) the idea behind moun and said this is wrong.

<img width="1500" height="783" alt="tweet-1893868235381961140" src="https://github.com/user-attachments/assets/cfb5e1d1-554d-4f4e-a937-a2301984ee01" />

How to short xAI?

# References
1. [Kimi K2 Blog post](https://moonshotai.github.io/Kimi-K2/)
