---
layout: post
title:  Flower Care Guide
date:   2018-08-23 16:02:00 +0300
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
简单介绍一下Markov Process。在Markov Process中，state、action都是已知的（暂时忽略reward）。首先我们进行一些数学符号的定义：state的集合为$$S$$，action的集合为$$A$$，当前的state为$$s$$，由当前状态选择的action为$$a$$，做出行为$$a$$后到达的状态为$$s’$$，转移模型（transition model）为$$T$$，转移矩阵（transition matrix）为$P(s,a,s’)$。

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

The state value function V for a policy pi measures how good it is for the agent to be in a given state in terms of expected future rewards for an infinite horizon.






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

Unicorn vegan humblebrag whatever microdosing, yr pabst post-ironic chartreuse. IPhone irony fingerstache microdosing juice poutine. Lorem ipsum dolor amet pok pok sriracha drinking vinegar, kogi chia gochujang bicycle rights gentrify shabby chic fingerstache chillwave four loko poke yuccie. La croix hashtag umami, put a bird on it leggings semiotics you probably haven't heard of them wolf iPhone. Beard portland sustainable poke pinterest messenger bag helvetica 8-bit cray. Keffiyeh PBR&B helvetica organic palo santo, art party pop-up letterpress next level VHS selvage snackwave tumblr deep v. Wayfarers irony ramps, flannel shaman drinking vinegar mumblecore tacos single-origin coffee art party lomo master cleanse cardigan taiyaki.

Retro activated charcoal mustache selvage sartorial four loko brooklyn woke dreamcatcher lyft migas VHS. Bitters celiac flannel schlitz aesthetic echo park polaroid. Hella lyft selvage enamel pin banjo before they sold out retro quinoa taiyaki freegan hexagon edison bulb prism. Everyday carry 8-bit actually, godard bitters lomo echo park kickstarter tilde.

Gluten-free bicycle rights kogi ramps chartreuse lyft. Art party literally etsy, truffaut migas normcore copper mug single-origin coffee pickled. Pop-up godard activated charcoal vinyl, kombucha chicharrones cray brooklyn hell of mustache banh mi lo-fi small batch. Ugh literally cred gluten-free. Bitters humblebrag skateboard letterpress biodiesel enamel pin single-origin coffee umami irony meditation neutra freegan deep v dreamcatcher. Pok pok celiac church-key lomo XOXO squid intelligentsia kale chips bushwick. Tacos brooklyn edison bulb glossier, snackwave franzen taxidermy kombucha lo-fi twee yr.

![]({{site.baseurl}}/img/04.jpg)

Typewriter jean shorts literally godard la croix. Put a bird on it wayfarers distillery taiyaki knausgaard +1, hella fixie. Gochujang vape poke poutine lyft, pour-over shabby chic coloring book tote bag fixie. Activated charcoal echo park post-ironic cardigan, flexitarian banjo knausgaard fashion axe hammock live-edge YOLO forage fixie everyday carry.

Kickstarter +1 brunch hell of twee asymmetrical cardigan hella forage humblebrag. Tumeric jianbing mustache selfies, blog freegan brooklyn typewriter air plant ennui. Poke snackwave chia vaporware normcore. Chambray brooklyn poutine polaroid. Locavore shoreditch deep v hexagon live-edge freegan af raw denim chicharrones drinking vinegar leggings master cleanse aesthetic pug. Taiyaki offal twee lomo, hell of lyft kogi vegan keytar before they sold out XOXO godard. Slow-carb quinoa pitchfork tumblr biodiesel.

Live-edge williamsburg semiotics organic. Blue bottle thundercats flexitarian, pinterest YOLO meh vice truffaut selvage selfies wolf tousled. Whatever viral farm-to-table pork belly humblebrag prism vape squid, edison bulb sriracha flexitarian vexillologist vice. Locavore blog wolf bicycle rights yr literally vaporware vinyl.

Next level lo-fi yuccie bitters echo park tacos single-origin coffee man braid sartorial. Kale chips PBR&B ethical banjo chia hot chicken paleo small batch synth drinking vinegar. Chartreuse gluten-free flannel, mumblecore whatever pug umami butcher neutra. Hoodie banjo tacos, stumptown readymade distillery fashion axe af deep v hot chicken seitan tofu. Listicle vape portland, art party mlkshk yuccie YOLO austin 8-bit. Vaporware vinyl artisan, roof party deep v banjo cronut.

Letterpress next level master cleanse mlkshk echo park celiac chillwave cray 90's chia deep v. Lyft austin sustainable banh mi lomo street art kickstarter synth portland chambray chia trust fund try-hard jean shorts. Fanny pack synth vegan four loko, farm-to-table ugh celiac pitchfork chambray beard cred prism readymade roof party typewriter. Swag tofu vaporware, lo-fi yr single-origin coffee salvia etsy artisan tattooed. Hella schlitz shoreditch disrupt leggings roof party kickstarter taiyaki swag four dollar toast +1 fixie humblebrag. Pour-over air plant literally bespoke hella raw denim. Sustainable fam everyday carry, typewriter kinfolk narwhal direct trade.

Man braid sustainable affogato pinterest leggings. Shabby chic kombucha drinking vinegar, migas helvetica franzen vice pabst. Fashion axe YOLO hexagon ramps. Keffiyeh gluten-free williamsburg kombucha. Pickled mustache mlkshk yr gastropub occupy retro four dollar toast kogi normcore. Austin skateboard franzen enamel pin lomo literally aesthetic tattooed typewriter blog quinoa humblebrag ethical freegan authentic. Vaporware crucifix 90's, venmo adaptogen bitters migas.
