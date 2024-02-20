---
layout: post
title:  Reinforcement Learning (RL) study note 2 (VBRL)
date:   2024-02-12 00:00:50 +0300
image:  04.jpg
tags:   [RL]
usemath: latex
---

![]({{site.baseurl}}/img/RL-Note1-1.png)

# 1 Q* function
Step of value-based RL:
- Estimate the Q* function directly from data.
- With a finite action space, one can make dicisions directly from the Q* function.

## 1.1 Solve for Q* from data
### 1.1.1 Recall Value Iteration (VI):
1. Initialize $$Q^{(0)}$$ arbitrarily.
2. For $$t=1,\dots, T$$
    $$Q^{(i)}(s,a)=r(s,a)+\gamma \mathbb{E}_{s'\sim P(\cdot \mid s,a)}\left [ \max_{a'} Q^{(i-1)}(s',a') \right ] $$
3. Return $$ Q^{(T)}$$
### 1.1.2 Fitted Q iteration (FQI):
Given a dataset $$D={(s_i,a_i,r_i,s'_i)}^n_{i=1}$$.
1. Initialize $$Q^{(0)}$$ arbitrarily.
2. For $$t=1,\dots, T$$
    $$Q^{(i)}(s,a)={argmin}_{f \in \mathcal{F}} \sum_{i=1}^n \left ( f(s_i,a_i)-r_i-\gamma \max_{a'} Q^{(i-1)}(s_i^{'},a') \right )^2 $$
3. Return $$ Q^{(T)}$$

$$\mathcal{F}$$ is the fucntion class.
- In the tabular setting, $$\mathcal{F} = \{f: S \times A \to \mathbb{R}\} $$
- More generally, $$\mathcal{F}$$ can be a neural network mapping from $$(s,a)$$ to $$\mathbb{R}$$

**FQI is actually used in offline RL.**

## 1.2 Compare with model-based learning
Given a dataset $$D={(s_i,a_i,r_i,s'_i)}^n_{i=1}$$.
Model-based RL:
1. Learn $$\hat{P}(s'\mid s,a)= \frac{N_D(s,a,s')}{N_D(s,a)} $$
2. Return $$\hat{Q} = Q_{\hat{P}}^*  $$, e.g. via Value Iteration (VI)

Compare $$Q^{(T)} $$ from FQI with $$\hat{Q} $$ from MBRL, you will find that **they are the SAME!** i.e. $$\lim_{T \to \infty} Q^{(T)} = \hat{Q} $$  
( $$Q^{(t)}=\hat{Q}^{(t)} $$, where $$\hat{Q}^{(t)}$$ if from VI in $$\hat{P}$$ )

The difference between FQI and MBRL is Computational/Space complexity.

MBRL learn and save the model, which lives in $$\mathbb{R}^{S \times A \times S} $$. Value-based RL learn and save the Q function, which lives in $$\mathbb{R}^{S \times A} $$.

In order to use FQI in online RL, one must store all historical data, of size $$(S + A)^T$$, which sometimes dominates the space complexity.

# 2 Q-Learning
## 2.1 Q-Learning: a streaming algorithm
- At time step $$t$$
- Observes transition tuple $$(s_t,a_t,r_t,s'_t)$$
- Q-Learning: 
    $$Q^{(t+1)}\left(s_{t}, a_{t}\right)=Q^{(t)}\left(s_{t}, a_{t}\right)+\alpha_{t}\left(s_{t}, a_{t}\right)\left(r_{t}+\gamma \max _{a^{\prime}} Q^{(t)}\left(s_{t}^{\prime}, a^{\prime}\right)-Q^{(t)}\left(s_{t}, a_{t}\right)\right)$$
- Recall FQI:
    $$Q^{(t)}(s, a)=\operatorname{argmin}_{f \in \mathcal{F}} \sum_{i=1}^{n}\left(f\left(s_{t}, a_{t}\right)-r_{t}-\gamma \max _{a^{\prime}} Q^{(t)}\left(s_{t}^{\prime}, a^{\prime}\right)\right)^{2}$$
- Q-Learning is taking one gradient step with respect to the FQI objective with step size $$\alpha_t(s_t,a_t)$$

## 2.2 When does Q-Learning converge to Q*
If and only if $$\sum_t^{\infty} \alpha_t(s,a)=\infty$$ and $$ \sum_t^{\infty} \alpha_t^2(s,a)<\infty $$ for all $$(s,a) \in S \times A$$, Q-Learning converge to Q*.

If $$(s,a)$$ is not visited at step t, then $$\alpha_t(s_t,a_t)=0$$.
So $$\sum_t^{\infty} \alpha_t(s,a)=\infty$$ implies that the learning rate must take a diminishing rate at least $$ \alpha_t(s,a) \propto 1/\sqrt{N_t(s,a)} $$ and at most $$1/N_t(s,a)$$.
However, this theorem only works for tabular setting. When $$\{s\}$$ is large or infinite, you can't hope to visit each state infinitely often.

## 2.2 When does Q-Learning not converge to Q*
In fact, Q-Learning is known to diverge under function approximation.

And Q-Learning can fail under function approximation:
![]({{site.baseurl}}/img/RL-Note2-fail.png)

# 3 What happens beyond the tabular setting

**Value-based RL may fail:**
1. They might not converge (algorithm-specific: Q-Learning).
2. They might not converge to the correct solution (all value-based RL).

## Failure examples
### MDP: 2 states, 1 action.
![]({{site.baseurl}}/img/RL-Note2-fail.png)
- **Realizable** Linear Function Approximation: $$Q(s,a) = \phi(s,a)^Tw$$

$$ Q^{(0)}(s,a)=\phi w^{(0)}, w^{(0)}>0, \gamma = 0.9  $$

step 1:$$ (s1, \alpha, s_2, \gamma ) $$

$$ \begin{split} Q^{(1)}(s_1) &= (1-\alpha)(1\cdot w^{(0)})+\alpha (0+\gamma \cdot 2 w^{(0)}) \\
&= (1+0.8\alpha)\cdot w^{(0)} \end{split} $$

$$ w^{(1)} = (1+0.8\alpha)\cdot w^{(0)} $$

step 2:$$ (s1, \alpha, s_2, \gamma ) $$

$$ Q^{(2)} = (1-\alpha)(2\cdot w^{(1)})+\alpha \gamma w^{(1)} $$
$$ w^{(2)} = (1-0.55\alpha)w^{(1)} $$
$$ w^{(1)} = x\cdot w^{(0)} $$

$$ \begin{split} Q^{(2)} &= (1-\alpha) Q^{(1)}(s_1)+\alpha \gamma Q^{(0)}(s_2)\\
&= (1-\alpha) w^{(1)}+\alpha \gamma w^{(0)}\\
&\le w^{(1)}
\end{split}$$
when $$\alpha w^{(1)} > 2\alpha \gamma w^{(0)} $$

![]({{site.baseurl}}/img/RL-Note2-fail2.png)

### Bellman-Completeness
- A Q function class $$\mathcal{F}$$ is **Bellman-Completeness** if
- For any $$f \in \mathcal{F} $$, there exists $$g \in \mathcal{F} $$, such that

$$g(s,a)=(\tau f)(s,a) = r(s,a) + E_{s'\sim P(s'\mid s,a)}\left[ \max_{a'} f(s',a') \right] $$

$$\tau f (s_1) = 0+f(s_2) = 2 \cdot \gamma \cdot w $$

$$\begin{split} \tau f (s_2) &= 0+ [(1-\varepsilon)w+\varepsilon \cdot 2 \cdot w]\gamma\\
&= \gamma (1-\varepsilon)w \end{split}$$

$$\mathcal{F}=\{f(s)\mid f(s) = \phi(s)\cdot w \} , \tau f \notin \mathcal{F},\left(\tau f(2) \not = 2 \tau f(1)\right)$$

- In other words, $$\mathcal{F}$$ is **closed** under Bellman operator $$\tau$$.

- Completeness is **not monotone**, so having a rich function class won't help.

![]({{site.baseurl}}/img/RL-Note2-BC.png)

- Theorem (Foster et al., 2022). Value-based method can fail without Bellman-Completeness.

- They contrasted an failure example, where any algorithm require at least $$\mid S\mid^{1/3}$$ samples to learn a good policy.
- This is an **algorithm-independent** result.

So, to the FQI and Q-learning:
1. FQI requires storing all historical data, which is memory inefficient.
2. Q-learning converges just fine in the tabular setting.

# 4 To make Q-learning converge

## 4.1 Trick 1: Target network (two time-scale update rule)

$$Q^{(t+1)}(s_t,a_t) = Q^{(t)}(s_t, a_t) + \alpha_t(s_t, a_t)\left ( r_t+\gamma \max_{a'}T^{(t)}(s'_t,a') - Q^{(t)}(s_t, a_t) \right) $$

$$T^{(t+1)}(s_t,a_t) =  Q^{(t)}(s_t, a_t) + \beta_t(s_t, a_t)\left ( Q^{(t)}(s_t,a_t)+ T^{(t)}(s_t,a) \right) $$

It is a slowly updating target network.

## 4.2 Trick 2: Double Q-learning

$$Q^{(t+1)}(s_t,a_t) = Q^{(t)}(s_t, a_t) + \alpha_t(s_t, a_t)\left ( r_t+\gamma T^{(t)}(s'_t,a') - Q^{(t)}(s_t, a_t) \right) $$

$$T^{(t+1)}(s_t,a_t) =  Q^{(t)}(s_t, a_t) + \beta_t(s_t, a_t)\left ( Q^{(t)}(s_t,a_t)+ T^{(t)}(s_t,a) \right) $$

$$a'={argmax}_a Q^{(t)}(s'_t,a) $$

## 4.3 Baseline 3: Inverse double Q-learning

$$Q^{(t+1)}(s_t,a_t) = Q^{(t)}(s_t, a_t) + \alpha_t(s_t, a_t)\left ( r_t+\gamma Q^{(t)}(s'_t,a') - Q^{(t)}(s_t, a_t) \right) $$

$$T^{(t+1)}(s_t,a_t) =  Q^{(t)}(s_t, a_t) + \beta_t(s_t, a_t)\left ( Q^{(t)}(s_t,a_t)+ T^{(t)}(s_t,a) \right) $$

$$a'={argmax}_a T^{(t)}(s'_t,a) $$

![]({{site.baseurl}}/img/RL-Note2-Converge.png)

