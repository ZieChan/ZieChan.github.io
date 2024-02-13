---
layout: post
title:  Reinforcement Learning 简单介绍
date:   2023-08-23 16:02:00 +0300
image:  07.jpg
tags:   [Guide, Flowers]
usemath: latex
---

# Reinforcement Learning 理论学习笔记

一般来说，机器学习可以被分为有监督学习、无监督学习和强化学习。而强化学习会被单独列出来是因为它是在某个场景下，与环境进行互动交流而进行学习。
  
举一个例子，假设我们从没有玩过超级马里奥，进入游戏后，我们知道有向前走、向后走、跳和蹲这四个操作，不过我们并不知道这些动作有什么用。那么首先我们进入游戏后，我们像个树桩一样站着不动，时间到了之后，我们就寄了（输了）。那现在我们知道，保持不动带来的回报可能是负的了。

游戏重新开始后，我们一直往前走，走到一个方块下面，我们尝试顶一下这个方块。方块碎了，并且我们得到了一个金币。那么在方块下面进行跳跃这个动作的回报就从0变成1。

强化学习就是不停地在环境中进行操作，学习每个状态（state）下每个动作（action）的回报（reward），最终通过学习得到了每个state的价值（value），或是在给定state的情况下做出各个action的概率，亦或是在这个环境中最佳的行为方式（policy）。
## 1 Markov Process & Markov Decision Processes

要介绍强化学习的原理，就得说到Markov Process和Markov Decision Processes，毕竟它们可以说是强化学习的基础。

### 1.1 Markov Process
简单介绍一下Markov Process。在Markov Process中，state、action都是已知的（暂时忽略reward）。首先我们进行一些数学符号的定义：state的集合为$$S$$，action的集合为$$A$$，当前的state为$$s$$，由当前状态选择的action为$$a$$，做出行为$$a$$后到达的状态为$$s’$$，转移模型（transition model）为$$T$$，转移矩阵（transition matrix）为$$P(s,a,s’)$$。

让我们假设下面这种情景：某个机器人存在三种状态（state），即$$state = S = \{ standing, moving, fallen\}$$，两种动作（action），即$$action = A = \{ slow, fast\}$$。而每个状态（state）下的做出各个动作（action）的概率在Markov Process中是已知的。

![]({{site.baseurl}}/img/20231114222135.jpg)
_Figure 1 A Markov Process defined by a set of states_

在图1中，橙色的线条代表着slow，紫色的线条代表着fast。因此在此情景下的transition matrix 为：

![]({{site.baseurl}}/img/20231114222150.jpg)

因此Markov Process 中下一个状态$$s’$$的选择由上一个状态$$s$$和由上一个状态选择的行为$$a$$决定。

![]({{site.baseurl}}/img/20231114222159.jpg)

### 1.2 Markov Decision Process

Markov Decision Process可以说是Markov Process的升级版，在这里我们就要考虑在当前state做出某个action的value了。$$S$$、$$A$$、$$T$$、$$s$$、$$a$$、$$s’$$的定义同1.1，回报的期望（expected rewards）定义为$$R（s,a)$$，而由当前状态$$s$$做出行为$$a$$得到的回报reward为$$r(s,a)$$。

现在我们升级一下1.1的情景，加入每个状态下做出各个行为后到新的状态的回报，

![]({{site.baseurl}}/img/20231114222206.jpg)
由此可以看出，$$r(standing, slow,moving)=1$$，$$r(standing, fast,moving)=2$$，以此类推。而从fallen状态做出slow行为到standing的状态的概率为$$T(s,a,s’)=p(fallen,slow,standing)=\frac{2}{5}$$，从fallen状态做出slow行为到fallen的状态的概率为$$T(s,a,s’)=p(fallen,slow,fallen)=\frac{3}{5}$$。

那么一对现在的状态与行为的预期回报$$R(s,a)$$为

$$\begin{equation} R(s,a)=E[r_{t+1}|s_t=s,a_t=a]=\sum_r r\sum_{s’}P(s’,r|s,a) \end{equation}$$

而$$\begin{equation}
P(s’,r|s,a)=P(s_{t+1}=s’,r_{t+1}=r|s_t=s,a_t=a)
\end{equation}$$
并且对于所有可能的$$(s,a)$$， $$\sum_{s’}\sum_rP(s’,r|s,a)=1$$。

举个例子，从状态fallen做出slow行为的预期回报

$$\begin{equation} \begin{split} 
R(fallen, slow)&=p(fallen, slow, fallen) \cdot r(fallen, slow, fallen)\\
&\quad +p(fallen, slow, standing) \cdot r(fallen, slow, standing)\\
&=\frac{3}{5}\cdot (-1) +\frac{2}{5}\cdot 1= -\frac{1}{5}
\end{split} \end{equation}$$

因此，在该情景下的$$R(s,a)$$为

![]({{site.baseurl}}/img/20231114222226.jpg)
那么，我们再给出由当前状态与做出的行为，到下一个状态的概率的定义：
$$\begin{equation} P(s’|s,a) = P(s_{t+1} = s’|s_t = s,a_t = a)=\sum_rP(s’,r|s,a) \end{equation}$$
而由当前状态做出一个行为后进入到某一个状态的预期回报为：
$$\begin{equation} R(s,a,s’)=E[r_{t+1}|s_t=s,a_t=a,s_{t+1}=s’]=\frac{\sum_r rP(s’,r|s,a)}{P(s’|s,a)}\end{equation} $$

不过对比现实中的情景，上面的马尔可夫模型就显得非常简单且理想了。现实中的状态和行为必然会更加多样和复杂，而行为的选择也是更具现实中的观察而选择的。(比如玩LOL，你观察到对面没有出现，你聪明的大脑就会选择让你上去补兵了，但可能旁边草丛里面藏着5个你没观察到的老六……而如果这5个b直接出现在你的兵线上等着你，你大概率不会上去补兵了)

![]({{site.baseurl}}/img/20231114222235.jpg)

因此在现实中，行为$$a$$的选择与观察$$o$$和状态$$s$$有关。

## 2 Reinforcement Learning 的相关定义
首先我们定义一个规则(policy)$$\pi:S \to A$$,基于这个规则$$\pi$$,每当我们遇到一个情况$$s$$，我们就会做出行为$$a$$。这样，在每一步(step)$$t$$中，根据规则$$\pi$$与当前情况$$s$$,做出行为$$a$$的概率可以表示为
$$\begin{equation} \pi_t(a|s)=P(a_t=a|s_t=s) \end{equation}$$
而选择的行为$$a$$也可以表示为$$a=\pi_t(s)$$。

而从一个状态$$s$$执行一个行为$$a$$，得到回报$$r$$,进入到下一个状态$$s’$$，再执行下一个行为的过程……这个过程我们经常用一个树表示

![]({{site.baseurl}}/img/20231114222243.jpg)

### 2.1 状态价值函数 State Value Function
那么在强化学习中，我们会想知道在某个情景下的policy $$\pi :S \to A$$是什么。policy的学习程度取决于我们根据这个policy学习的步数$$h$$,也称为horizon。

我们引入状态价值函数$$V_{\pi}(s)$$去衡量在某个规则$$\pi$$下，给定状态$$s$$的的好坏。而这个状态的价值取决于无限步之后的回报。

当$$h=0$$时：

$$\begin{equation} V_{\pi}^0=0 \end{equation}$$

当$$h=1$$时：

$$\begin{equation} V_{\pi}^1=R(s,\pi(s))+V_{\pi}^0 = R(s,a) \end{equation}$$

当$$h=2$$时：

$$\begin{equation} V_{\pi}^2=R(s,\pi(s))+\sum_{s’}T(s,\pi(s),s’)R(s’,\pi(s’)) \end{equation}$$

对于任意$$h$$：

$$\begin{equation} V_{\pi}^h=R(s,\pi(s))+\sum_{s’}T(s,\pi(s),s’)V_{\pi}^{h-1} (s') \end{equation}$$

从上面也可以看出$$V_{\pi}^h$$是由$$V_{\pi}^{h-1}$$递归计算得到的。

而状态价值函数同时也是回报的期望：

$$\begin{equation} V_{\pi}(s)= E_{\pi}[g_t|s_t=s]=E_{\pi}[\sum_k \gamma^k r_{t+k+1}|s_t=s] \end{equation}$$

在这里我们引入一个折扣率(discount rate) $$\gamma$$，当$$\gamma \to 0$$代表越重视近期的影响，并且忽视长期影响，若$$\gamma \to 1$$则代表越重视长期的影响，对近期影响的考虑逐渐下降。而$$g:S\to Y$$代表着根据观察环境，从状态state得到回报的函数。

当$$h=\infty$$时

$$\begin{equation} \begin{split} 
V_{\pi}(s)&=E[\sum_{t=0}^{\infty} \gamma^t R_t|\pi,s_0]\\
&=E[R_0+\gamma R_1+\gamma^2 R_2+\cdots |\pi,s_0=s]\\
&=E[R_0+\gamma(R_1+\gamma (R_2+\cdots))|\pi,s_0=s]\\
&=R(s,\pi(s))+\gamma\sum_{s'}T(s,\pi(s),s')V_{\pi}(s')
\end{split} \end{equation}$$

### 2.2 动作价值函数 Action Value Function
与状态价值函数类似，我们可以用动作价值函数$$Q_{\pi}^h(s,a)$$在某个policy下动作价值的好坏。同样是对于policy $$\pi$$，horizon$$h$$，state $$s$$和action$$a$$。

当$$h=0$$时：

$$\begin{equation} Q_{\pi}^0=0 \end{equation}$$

当$$h=1$$时：

$$\begin{equation} Q_{\pi}^1=R(s,\pi(s))+0 = R(s,a) \end{equation}$$

当$$h=2$$时：

$$\begin{equation} Q_{\pi}^2=R(s,\pi(s))+\sum_{s’}T(s,\pi(s),s’)\max_{a'}R(s’,a') \end{equation}$$

对于任意$$h$$：

$$\begin{equation} Q_{\pi}^h=R(s,\pi(s))+\sum_{s’}T(s,\pi(s),s’))\max_{a'}Q_{\pi}^{h-1} (s',a') \end{equation}$$

当有$$n$$个states，$$|S|=n$$,$$m$$个actions，$$|A|=m$$且horizon为$$h$$时，$$Q_{\pi}^h(s,a)$$的时间复杂度为$$O(nmh）$$。
与State Value Function 类似，当我们考虑$$\gamma$$和$$g$$时：

$$\begin{equation} Q_{\pi}(s,a)= E_{\pi}[g_t|s_t=s,a_t=a]=E_{\pi}[\sum_k \gamma^k r_{t+k+1}|s_t=s,a_t=a] \end{equation}$$

而$$V_{\pi}(s)$$与$$Q_{\pi}(s,a)$$之间是有联系的，它们之间的关系为：

$$\begin{equation} V_{\pi}(s)=\sum_a \pi(a|s)Q_{\pi}(s,a) \end{equation}$$







### h3 标题
#### h4 标题
##### h5 标题
###### h6 标题


## 水平线

___

---

***


## 文本样式

**This is bold text**

__This is bold text__

*This is italic text*

_This is italic text_

~~Strikethrough~~


## 列表

无序

+ Create a list by starting a line with `+`, `-`, or `*`
+ Sub-lists are made by indenting 2 spaces:
  - Marker character change forces new list start:
    * Ac tristique libero volutpat at
    + Facilisis in pretium nisl aliquet
    - Nulla volutpat aliquam velit
+ Very easy!

有序

1. Lorem ipsum dolor sit amet
2. Consectetur adipiscing elit
3. Integer molestie lorem at massa


1. You can use sequential numbers...
1. ...or keep all the numbers as `1.`

Start numbering with offset:

57. foo
1. bar


## 代码

Inline `code`

Indented code

    // Some comments
    line 1 of code
    line 2 of code
    line 3 of code


Block code "fences"

$$`
Sample text here...
$$`

Syntax highlighting

$$` js
var foo = function (bar) {
  return bar++;
};

console.log(foo(5));
$$`R

