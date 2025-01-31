---
layout: post
title:  Reinforcement Learning (RL) study note 1
date:   2024-02-01 00:00:50 +0300
image:  04.jpg
tags:   [RL]
usemath: latex
---
# 1 The RL Ontology
![]({{site.baseurl}}/img/RL-Note1-1.png)

# 2 Model-Based RL
## 2.1 What is model-based RL (MBRL)?
- Step 1: Learn an estimated MDP using existing data.
- Step 2: Do planning in the learned MDP (simulator).
- Step 3: Try the new policy out in the real world.
- Go to Step 1 after collecting more data.

So the notation of MBRL can be written as:
- Real world $$M = (P, r)$$
- Learned world $$\hat{M} = (\hat{P}, \hat{r})$$
- Find the optimal policy in $$\hat{M}$$ : $$ \hat{\pi}={argmax}_{\pi} V^{\pi}_{\hat{M}} $$

**Simulation Lemma:**

$$\max_s V^*_M(s)-V^{\hat{\pi}}_M(s) \le \frac{\gamma}{1-\gamma}\max_{s,a} \left \| P(\cdot \mid s,a) - \hat{P}(\cdot \mid s,a)  \right \|_1 \cdot \left \| V^* \right \|_\infty $$

## 2.2 Does it work in practice?
- To some extent, yes
- Heavily used in robotics

In other settings, like in Atari Games, model-based RL have been under-performing until very recently.
![]({{site.baseurl}}/RL-Note1-AtariPerformance.png)

*Why does it work in robotics but not Atari Games?*
- Strong prior knowledge on the physical model.
- data needed scales polynomially with parameters.

## 2.3 Challenges with MBRL

1. Transition model with high-dim observation is hard to learn.
2. How to plan in the learned model?
3. How to effectively encode domain knowledge? (This is a mega challenge for all of ML)

