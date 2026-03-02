# 控制论视角（非RL训练式）参考清单

> 目标对齐：针对“成功钓鱼前提下最短时间”的策略设计，以下资料**刻意避开需要训练策略网络**的方法，优先选择动态规划、最优控制、输出反馈MPC、belief-space规划等路线。

## 1. 基础理论：部分可观测随机最优控制（POMDP/信息状态）

### 1.1 Smallwood & Sondik (1973)
- 标题：The Optimal Control of Partially Observable Markov Processes over a Finite Horizon
- 类型：经典论文（Operations Research）
- 链接：https://ideas.repec.org/a/inm/oropre/v21y1973i5p1071-1088.html
- 摘要改写：将“状态不可直接观测”的离散马尔可夫控制问题形式化，证明有限时域下最优价值函数在信念空间上是分段线性凸函数，并给出相应求解框架。
- 对你问题的启发：你的环境可先做有限时域近似（例如剩余步数 N 的滚动优化），在 belief 空间做策略改进，而不是训练黑箱策略。

### 1.2 Sondik (1978)
- 标题：The Optimal Control of Partially Observable Markov Processes over the Infinite Horizon: Discounted Costs
- 类型：经典论文（Operations Research）
- 链接：https://ideas.repec.org/a/inm/oropre/v26y1978i2p282-304.html
- 摘要改写：把无限时域折扣最优控制扩展到部分可观测场景，使用“占据概率单纯形”作为状态，发展了可实现的近似策略改进思路并讨论收敛。
- 对你问题的启发：若要做长期策略（不限固定步长），可用 discounted 目标近似“尽快成功”，并在信息状态上迭代。

### 1.3 Borkar (2003)
- 标题：Dynamic programming for ergodic control with partial observations
- 类型：理论论文（Stochastic Processes and their Applications）
- 链接：https://www.sciencedirect.com/science/article/pii/S0304414902001904
- 摘要改写：在部分观测条件下建立动态规划原理，连接离散马尔可夫控制与连续扩散过程，给出平均代价（ergodic）问题的DP基础。
- 对你问题的启发：虽然你的目标不是平均代价，但该文给出“部分观测 + DP”的严格基础，可支持你把现问题写成信息状态动态规划。

### 1.4 Bandini et al. (2019)
- 标题：Randomized filtering and Bellman equation in Wasserstein space for partial observation control problem
- 类型：理论论文（SPA）
- 链接：https://www.sciencedirect.com/science/article/pii/S0304414918300553
- 摘要改写：针对部分观测随机控制，给出随机化滤波与DPP，并把价值函数刻画为 Wasserstein 空间上的HJB粘性解；还讨论了非高斯LQ特例。
- 对你问题的启发：如果你后续需要“连续状态 + 严格理论保证”，这条线是高阶但严谨的基础。

## 2. 可操作方法：belief-space规划（不训练策略网络）

### 2.1 Platt et al. (RSS 2010)
- 标题：Belief space planning assuming maximum likelihood observations
- 类型：会议论文（RSS）
- 链接：https://www.roboticsproceedings.org/rss06/p37.html
- 摘要改写：把部分可观测控制转为 belief-space 下的随机控制，用“最大似然观测”构造可规划的近似动力学，再配合 LQR/局部规划做在线重规划。
- 对你问题的启发：你可以对鱼位置观测噪声或隐藏状态做滤波，然后在 belief-space 做短视域重规划（receding horizon）。

### 2.2 van den Berg et al. (IJRR 2012)
- 标题：Motion Planning under Uncertainty using Iterative Local Optimization in Belief Space
- 类型：期刊论文（IJRR）
- 链接：https://robotics.cs.unc.edu/publications/vandenBerg2012_IJRR.pdf
- 摘要改写：在连续POMDP下，使用EKF近似 belief 动力学，并用 belief-space iLQG 做局部二阶优化，得到局部最优线性反馈策略。
- 对你问题的启发：这和你的环境高度匹配：连续状态、二值输入、短时动作优化。可直接尝试“估计器 + iLQG局部策略”。

### 2.3 Nishimura & Schwager (IJRR 2021)
- 标题：SACBP: Belief Space Planning for Continuous-Time Dynamical Systems via Stochastic Sequential Action Control
- 类型：期刊论文（IJRR）
- 链接：https://harukins.github.io/publication/2021_ijrr
- 摘要改写：把 belief 系统视作混合动力系统，基于扰动理论扩展 sequential action control，在连续时间下实现近实时策略合成，避免显式离散化。
- 对你问题的启发：如果你强调实时性（例如每帧快速决策），可借鉴其“anytime + 局部改进”思想。

## 3. 输出反馈MPC：工程上最容易落地的一类

### 3.1 Sehr & Bitmead (Automatica 2018)
- 标题：Stochastic output-feedback model predictive control
- 类型：期刊论文（Automatica）
- 链接：https://www.sciencedirect.com/science/article/abs/pii/S0005109818301985
- 摘要改写：把随机输出反馈最优控制转写为滚动优化，显式传播条件状态分布（信息状态），并给出相对无限时域最优控制的性能界思路。
- 对你问题的启发：非常适合“观测不全 + 风险约束 + 追求性能”的场景，可做你策略设计主线之一。

### 3.2 Yan, Cannon, Goulart (Automatica 2022)
- 标题：Stochastic output feedback MPC with intermittent observations
- 类型：期刊论文（Automatica）
- 链接：https://www.sciencedirect.com/science/article/abs/pii/S0005109822001285
- 摘要改写：考虑随机扰动、噪声和观测丢包，用“未来观测仿射参数化”把问题转成凸LQ优化，给出约束满足与代价有界性结论。
- 对你问题的启发：若你将来引入“偶发观测缺失/延迟”（例如模拟视觉丢帧），这篇很直接可用。

### 3.3 Homer & Mhaskar (ACC 2015)
- 标题：Output Feedback Model Predictive Control of Stochastic Nonlinear Systems
- 类型：会议论文（ACC）
- 链接：https://experts.mcmaster.ca/scholarly-works/150942
- 摘要改写：研究随机非线性系统的输出反馈预测控制，给出概率意义下可行性与稳定性，并显式刻画吸引域。
- 对你问题的启发：可借鉴“吸引域 + 概率稳定”评估思路来定义策略安全边界。

## 4. 最短时间控制：与你目标最直接

### 4.1 Tedrake《Underactuated Robotics》动态规划章节
- 标题：Ch. 7 - Dynamic Programming（含 double integrator 最短时间 bang-bang 例）
- 类型：讲义/在线教材
- 链接：https://underactuated.csail.mit.edu/dp.html
- 摘要改写：展示最短时间问题如何转化为可计算的最优控制问题，并以双积分器给出 bang-bang 结构和切换曲面直觉。
- 对你问题的启发：你的玩家竖直运动本质接近“受限加速度系统”，bang-bang/切换面思想可直接用于手工策略设计。

### 4.2 Sklyar et al. (JOTA 2015)
- 标题：Time-Optimal Control Problem for a Special Class of Control Systems
- 类型：期刊论文（开源）
- 链接：https://link.springer.com/article/10.1007/s10957-014-0607-6
- 摘要改写：研究一类非线性系统到原点的最短时间控制，证明最优控制取值常落在离散集合（-1/0/+1）且切换点有限。
- 对你问题的启发：为“二值动作 + 最短时间”的策略结构提供理论支持，与你的按键动作空间非常一致。

### 4.3 Bemporad et al. (IFAC 2011)
- 标题：Model predictive control for time-optimal point-to-point motion control
- 类型：会议论文（IFAC）
- 链接：https://www.sciencedirect.com/science/article/abs/pii/S1474667016439820
- 摘要改写：提出面向点到点最短时间运动的MPC实现，利用问题结构达到实时求解，并展示相较传统控制的时间性能优势。
- 对你问题的启发：给出“最短时间目标 + 在线优化 + 实时可算”的工程模板。

## 5. 讲义与教程（补齐理论到实现）

### 5.1 MIT 6.231 讲义（Bertsekas）
- Lecture 10（SSP）：https://ocw.mit.edu/courses/6-231-dynamic-programming-and-stochastic-control-fall-2015/resources/mit6_231f15_lec10/
- Lecture 17（SSP病态与弱条件）：https://ocw.mit.edu/courses/6-231-dynamic-programming-and-stochastic-control-fall-2015/resources/mit6_231f15_lec17/
- 摘要改写：系统讲解随机最短路（SSP）的Bellman方程、proper/improper policy、收敛与病态情形。
- 对你问题的启发：你的“成功为终止、失败为吸收、时间最短”可以严谨落在 SSP 框架里。

### 5.2 POMDPs for Dummies
- 链接：https://pomdp.org/tutorial/
- 类型：教程（非RL训练导向）
- 摘要改写：几何化解释POMDP值函数与经典求解器（如 witness / incremental pruning）直觉，适合理解 belief 更新和策略结构。
- 对你问题的启发：有助于你把“观测不全”从工程直觉转为可计算对象。

## 6. 对你项目的建议阅读顺序（非训练式策略优先）

1. Underactuated Ch.7（先拿到最短时间 bang-bang 直觉）
2. MIT 6.231 Lecture 10/17（把目标严谨化为 SSP）
3. Smallwood-Sondik 1973 + Sondik 1978（建立部分观测DP基础）
4. Platt 2010 + van den Berg 2012（形成可实现的 belief-space 局部最优策略）
5. Sehr-Bitmead 2018 + Yan-Cannon-Goulart 2022（工程化到输出反馈MPC）

## 7. 可直接迁移到 fishing_env 的三条“非训练”策略路线

- 路线A：Bang-bang + 切换面
  - 先用解析/数值方法构造“追鱼-刹车”切换规则，再加进度条风险修正项。
- 路线B：Belief-space 局部优化（iLQG / 局部LQR）
  - 用滤波器估计隐藏状态（速度、目标跳变），每步滚动求局部最优控制。
- 路线C：输出反馈随机MPC
  - 直接优化“预测窗口内成功概率约束 + 时间代价”，在线求解二值近似控制序列。

---

如果你愿意，我下一步可以把上面三条路线进一步细化成你仓库可执行的版本（状态估计器、优化变量、代价函数、约束、每步伪代码）。
