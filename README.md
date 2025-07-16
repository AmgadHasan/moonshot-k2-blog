# moonshot-k2-blog
My recap of Moonshot AI's blog post about the Kimi-K2 LLM

## Prinicples
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
