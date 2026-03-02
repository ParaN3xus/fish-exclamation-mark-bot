# TimeOptimalBangBangPolicy 策略说明

本文解释 [`src/control/time_optimal_bangbang_policy.py`](../src/control/time_optimal_bangbang_policy.py) 中 `TimeOptimalBangBangPolicy` 的设计与表现来源。

## 1. 策略目标

这个策略不是单纯“追鱼中心”，而是两层目标：

1. 优先保证 `catch_progress` 不掉到失败区间（success-first）。
2. 在可行前提下用最短时间控制思想快速回到重叠区。

动作空间是二值：`a in {0, 1}`，分别对应“松开/按下”。

## 2. 动力学抽象（最核心）

策略把相对运动写成近似双积分系统：

- 相对位置：`y = fish - player`
- 相对速度：`r = fish_v - player_v`
- 控制只影响玩家加速度，进而影响 `r_dot`

在这个抽象下，`_min_time_first_action` 对两种一开关序列做解析比较：

- 先按后松：`(1 -> 0)`
- 先松后按：`(0 -> 1)`

它通过求解二次方程估计到达 `y=0` 的时间，选首个动作最优的那一支。这就是 bang-bang 的“最短时间首动作”来源。

关键实现位置：
- `_min_time_first_action`（解析最短时间首动作）
- `_quadratic_roots`（根求解）

## 3. 仅靠 bang-bang 不够：必须有状态重建

环境观测只给 `fish_center/player_center`，看不到真实内部状态。策略自己重建：

- 玩家速度估计 `_v_est`
- 鱼速度估计 `_fish_v_est`
- 鱼隐藏目标位置 `_fish_target_est`
- 距离下次换向的时间估计 `_time_since_target_change`
- 进度条估计 `_progress_est`

这些量分别由：
- `_estimate_kinematics`
- `_update_progress_estimate`

持续在线更新。

没有这一步，最短时间控制会在高难度频繁换向时失效。

## 4. 预测与风险机制

### 4.1 预测鱼位置

`_predict_fish_center` 用当前难度、鱼衰减率、换向不确定性，预测短时未来鱼中心，不直接用当前鱼点做控制目标。

作用：减小“追着尾迹跑”的滞后。

### 4.2 风险指标

`danger = clamp((0.4 - progress_est)/0.4)`

进度越低，danger 越高，策略越保守，体现在：
- 更宽恢复带 `recovery_band`
- 更偏向保守目标偏置 `target_bias`
- 更重视鲁棒评分而非局部最短时间

### 4.3 一步风险覆盖

当 `danger > 0.2`，策略会比较两动作下一步是否更容易掉出重叠阈值，必要时覆盖掉当前动作。

这一步经常能避免“本来最短时间，但下一帧直接丢杆”的失败。

## 5. 高难度强项：鲁棒首动作评估

高难度（或 danger 偏高）时，会调用 `_evaluate_first_action_robust`：

- 对首动作 `0/1` 分别做短视域滚动仿真
- 构造鱼换向场景（低/中/高目标）
- 使用 `worst + mean` 混合打分，`risk_weight` 随难度和 danger 增大

最后选鲁棒分更高的首动作。

这使它在 `difficulty >= 7` 时，比单纯追踪或纯局部最短时间更稳定。

## 6. 行为结构总结（执行顺序）

`act()` 的实际流程可概括为：

1. 更新估计状态（速度、进度、鱼目标）。
2. 用预测鱼位置做名义最短时间首动作。
3. 远离目标时走恢复逻辑（快速回收）。
4. 边界保护（防撞墙反弹损失时间）。
5. 近目标区用滞回抑制抖振。
6. danger 高时做一步风险覆盖。
7. 高难/高风险时用鲁棒短视域仿真二选一。

## 7. 为什么它目前总体最强

结合当前实现，它的优势是“解析最短时间 + 鲁棒风险控制”双重结构：

- 解析解带来快速收敛能力（追得快）。
- 场景鲁棒打分带来高难稳定性（不容易掉进度）。

很多策略只能做到其中一项。

## 8. 局限与下一步

已知局限：

- `difficulty=9` 仍接近不可解区（多个策略都很低）。
- 评分函数参数较多，存在场景依赖。

可继续优化方向：

1. 针对 `difficulty=9` 增大鲁棒视域和场景覆盖。  
2. 在 `_evaluate_first_action_robust` 中加入“触底/触顶反弹惩罚”项。  
3. 对装备参数（strength/expertise）做分档参数集，而不是一套通用参数。
