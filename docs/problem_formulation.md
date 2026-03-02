# 钓鱼控制问题形式化表述

基于 [`src/gym/fishing_env.py`](../src/gym/fishing_env.py)，该问题建议采用“双层表述”：

- 主表述（用于求策略）：POMDP
- 辅助表述（用于机理分析）：随机控制系统

原因是策略实际依赖观测 `FishingObservation` 而非完整内部状态，因此从求解策略角度应以 POMDP 为主。

## 1. 时间与参数

- 时间步：$t=0,1,\dots,H-1$，其中 $H=\texttt{max\_steps}$。
- 步长：$\Delta t=\texttt{dt}$（默认 $1/60$ 秒）。
- 一个回合内固定参数：难度、装备加成、FPS/VR 相关参数。

## 2. 主表述：POMDP（推荐用于策略求解）

一个离散时间 POMDP 可写成
$$
\mathcal{M}=(\mathcal{X},\mathcal{A},\mathcal{O},P,\Omega,r,\gamma)
$$
其中策略为输出反馈形式：
$$
\pi:\mathcal{O}\to\Delta(\mathcal{A})
$$
或带记忆形式 $\pi(a_t\mid o_{0:t},a_{0:t-1})$。

### 2.1 状态、观测与动作

令完整马尔可夫状态为
$$
x_t=(f_t,p_t,v_t,c_t,\tau_t,q_t,\eta_t)
$$
其中：

- $f_t\in[0,1]$：鱼中心位置（`fish_position`）
- $p_t\in[0,1]$：玩家浮标中心位置（`player_position`）
- $v_t\in\mathbb{R}$：玩家速度（`player_velocity`）
- $c_t\in[0,1]$：进度条（`catch_progress`）
- $\tau_t\ge 0$：鱼方向计时器（`fish_direction_timer`）
- $q_t\in[0,1]$：鱼目标位置（`fish_target_position`）
- $\eta_t$：其余内部变量（如 `total_fight_time`, `step_count`）

策略可见观测为
$$
o_t=(f_t,p_t,\Delta t,d)
$$
对应 `FishingObservation`，因此这是“部分可观测”控制问题。

动作为二值输入：
$$
a_t\in\{0,1\}
$$
其中 $a_t=1$ 表示按住输入键，$a_t=0$ 表示松开。

### 2.2 状态转移

#### 2.2.1 鱼运动

- 方向计时器累加：$\tau_t\leftarrow \tau_t+\Delta t$。
- 当 $\tau_t$ 超过有效换向时间阈值时，采样新目标并截断单次跳变幅度（代码中由难度/FPS/VR 决定）。
- 鱼位置按一阶指数平滑追踪目标：
$$
f_{t+1}=\mathrm{clip}_{[0,1]}\!\left(f_t+\alpha_t(q_t-f_t)\right),\quad
\alpha_t=1-e^{-\lambda_t\Delta t}
$$
$\lambda_t$ 为有效衰减率（受难度、装备、FPS/VR 影响）。

#### 2.2.2 玩家运动

令 $g=\texttt{gravity}, s=\texttt{player\_speed}$，则
$$
v_t' = v_t - g\Delta t + a_t s\Delta t
$$
$$
p_t'=\mathrm{clip}_{[0,1]}(p_t+v_t'\Delta t)
$$
若触边界（$p_t'=0$ 或 $1$），速度反弹并衰减：
$$
v_{t+1}=-0.3\,v_t'
$$
否则 $v_{t+1}=v_t'$，且 $p_{t+1}=p_t'$。

#### 2.2.3 进度条更新

定义命中事件
$$
h_t=\mathbf{1}\{|f_{t+1}-p_{t+1}|<\delta_t\}
$$
其中 $\delta_t$ 为重叠阈值（受难度、装备、FPS/VR 影响）。

- 若 $h_t=1$：$c_{t+1}= \mathrm{clip}_{[0,1]}(c_t+r_c\Delta t)$
- 若 $h_t=0$：$c_{t+1}= \mathrm{clip}_{[0,1]}(c_t-r_{\text{lose},t}\Delta t)$

其中 $r_{\text{lose},t}$ 随对局时间增加（含宽限期与上限倍数）。

### 2.3 初始条件与终止

`reset()` 后：

- $f_0=0.5,\;p_0=0.5,\;v_0=0,\;c_0=0.1,\;\tau_0=0$
- $q_0\sim \mathrm{Uniform}(0.3,0.7)$

终止条件：

- 成功：$c_t\ge 1$
- 失败：$c_t\le 0$
- 截断：达到 $H$ 且未成功

定义成功事件 $S$ 与结束时刻 $T$（对应 `total_fight_time`）。

### 2.4 优化目标（成功前提下最短时间）

你提出的目标可写成约束优化：
$$
\min_{\pi}\ \mathbb{E}_{\pi}[T\mid S]
\quad\text{s.t.}\quad
\mathbb{P}_{\pi}(S)\ge p_0
$$
其中 $p_0$ 取接近 1（理想为 1）。

等价地，也可用词典序目标表达：

1. 先最大化 $\mathbb{P}_{\pi}(S)$；
2. 在最优成功率策略集合内最小化 $\mathbb{E}_{\pi}[T\mid S]$。

这正对应“先确保钓上来，再尽可能快”。

### 2.5 与当前环境 reward 的关系

当前 `step()` 的即时奖励是：

$$
r_t=(c_{t+1}-c_t)+\mathbf{1}_{\text{success}}-\mathbf{1}_{\text{fail}}
$$
它鼓励进度增长并区分成功/失败，但并不直接等价于“条件最短时间”。若训练时要严格对齐目标，建议采用上面的约束/词典序定义（或其拉格朗日近似）。

## 3. 辅助表述：控制论视角（状态空间）

同一系统可写成随机离散时间控制系统：
$$
x_{t+1}=f(x_t,u_t,w_t),\quad y_t=h(x_t)
$$
其中：

- $x_t$：完整状态（含 $f_t,p_t,v_t,c_t,\tau_t,q_t$ 等）
- $u_t\in\{0,1\}$：控制输入
- $w_t$：随机扰动（鱼目标采样与相关随机性）
- $y_t=o_t$：可测输出（观测）

终端集合可写为：
$$
\mathcal{X}_{\mathrm{succ}}=\{x\mid c\ge 1\},\quad
\mathcal{X}_{\mathrm{fail}}=\{x\mid c\le 0\}
$$
其目标是“高概率到达 $\mathcal{X}_{\mathrm{succ}}$ 且到达时间最短”。  
由于实际只能观测 $y_t$ 而非完整 $x_t$，实现时仍应回到输出反馈策略（即上面的 POMDP 求解）。
