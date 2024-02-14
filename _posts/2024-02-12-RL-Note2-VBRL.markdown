---
layout: post
title:  Reinforcement Learning (RL) study note 2 (VBRL)
date:   2024-02-12 00:00:50 +0300
image:  04.jpg
tags:   [RL]
---

![]({{site.baseurl}}/img/RL-Note1-1.png)

# 1. Q* function
Step of value-based RL:
- Estimate the Q* function directly from data.
- With a finite action space, one can make dicisions directly from the Q* function.

## 1.1 Solve for Q* from data
### 1.1.1 Recall Value Iteration (VI):
1. Initialize $$Q^{(0)}$$ arbitrarily.
2. For $$t=1,\dots, T$$
    - $$Q^{(i)}(s,a)=r(s,a)+\gamma \mathbb{E}_{s'\sim P(\cdot \mid s,a)}\left [ \max_{a'} Q^{(i-1)}(s',a') \right ] $$
3. Return $$ Q^{(T)}$$
### 1.1.2 Fitted Q iteration (FQI):
Given a dataset $$D={(s_i,a_i,r_i,s'_i)}^n_{i=1}$$.
1. Initialize $$Q^{(0)}$$ arbitrarily.
2. For $$t=1,\dots, T$$
    - $$Q^{(i)}(s,a)={argmin}_{f \in \mathcal{F}} \sum_{i=1}^n \left ( f(s_i,a_i)-r_i-\gamma \max_{a'} Q^{(i-1)}(s_i^{'},a') \right )^2 $$
3. Return $$ Q^{(T)}$$

$$\mathcal{F}$$ is the fucntion class.
- In the tabular setting, $$\mathcal{F} = \{f: S \times A \to \mathbb{R}\} $$
- More generally, $$\mathcal{F}$$ can be a neural network mapping from $$(s,a)$$ to $$\mathbb{R}$$

**FQI is actually used in offline RL.**


