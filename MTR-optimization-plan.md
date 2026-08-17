# MTR++ 优化方案

> 基于 WOMD 格式赛事约束，围绕 MTR++ baseline 的瓶颈分析与优化路线
>
> 更新时间：2026-08-17

---

## 一、MTR++ 四个核心瓶颈

### ① Agent-centric 计算冗余

- 每个目标 agent 独立编码整个场景（周围 agent + map polylines）
- N 个待预测 agent = N 次重复编码，虽然场景高度重叠
- WOMD 密集场景 190 agent → 190 次冗余计算
- 官方需 8×A100，batch size 才到 80
- 对比 QCNet：场景编码只算一次、所有 agent 共享

### ② Intention Query 数量固定

- 预定义 K 个可学习 query（如 64 或 128）覆盖运动意图
- 城区路口意图密集：64 个可能不够
- 高速直行意图简单：64 个大量冗余 → 多个 query 抢同一意图 → 模式坍塌
- 换数据集/换场景需重新调参

### ③ 置信度排序弱于位移精度

- minADE / minFDE 很强，但 mAP 43.29 非最高（ModeSeq 46.65）
- GMM 置信度头不够精准：能预测出贴真实轨迹的那条，但不确定哪条最可能
- 直接影响下游规划：拿到错误的"最可能"轨迹，决策就废了
- mAP 是比赛主指标，这是最需要优化的点

### ④ 闭环性能未验证

- 只在开环 benchmark（WOMD/AV2）上验证
- 闭环场景下预测误差沿规划链路累积放大
- 无闭环训练机制（对比 HiP-AD/ResAD）
- 开环 SOTA ≠ 闭环好用

---

## 二、六大优化创新点

按"投入产出比"排序，P0 为必做项，P1 为推荐项，P2 为进阶项。

### 优化 1：置信度校准（P0 · 半天）

**问题**：MTR++ 原始 confidence 排序不够准，能预测准但不知道哪条最可能。

**方案 A — 温度缩放**：

```python
import torch.nn as nn

class TemperatureScaling:
    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def __call__(self, logits):
        return logits / self.temperature

def fit_temperature(model, val_loader):
    model.eval()
    scaling = TemperatureScaling()
    optimizer = torch.optim.LBFGS([scaling.temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = 0
        for batch in val_loader:
            logits = model(batch)['cls_logits']
            scaled = scaling(logits)
            loss += nn.CrossEntropyLoss()(scaled, batch['best_mode_idx'])
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaling
```

**方案 B — Platt Scaling**：用逻辑回归拟合 confidence → accuracy 映射。

**方案 C — Label Smoothing**：训练时把 hard label 换成 soft label，防止过拟合高置信度。

**预期收益**：mAP +1-2 点。

---

### 优化 2：aWTA Loss（P0 · 1 天）

**问题**：标准 Winner-Takes-All loss 只优化最佳匹配的轨迹，其他 5 条不管，容易模式坍塌。

**方案**：退火温度策略，softmax 加权所有 mode 的 loss。

```python
import torch
import torch.nn.functional as F

def aWTA_loss(pred_trajs, pred_scores, gt_trajs, temperature, num_modes=6):
    """
    pred_trajs:  (B, 6, 80, 2)  模型预测的6条轨迹
    pred_scores: (B, 6)          每条置信度
    gt_trajs:    (B, 80, 2)      真实轨迹
    temperature: float           退火温度，从高到低
    """
    # 1. 计算每条轨迹和 GT 的距离 (B, 6)
    dist = torch.norm(pred_trajs - gt_trajs.unsqueeze(1), dim=(-1, -2)).mean(dim=-1)

    # 2. aWTA：用温度做 softmax 加权
    weights = F.softmax(-dist / temperature, dim=1)  # (B, 6)

    # 3. 加权回归 loss
    reg_loss = (weights * dist).sum(dim=1).mean()

    # 4. 分类 loss（预测哪条最好）不变
    best_mode = dist.argmin(dim=1)
    cls_loss = F.cross_entropy(pred_scores, best_mode)

    return reg_loss + cls_loss
```

**退火调度器**：

```python
class AnnealingScheduler:
    def __init__(self, start_temp=10.0, end_temp=0.1, total_epochs=30):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.total_epochs = total_epochs

    def get_temperature(self, epoch):
        ratio = epoch / self.total_epochs
        return self.start_temp * (self.end_temp / self.start_temp) ** ratio
```

**训练循环**：

```python
scheduler = AnnealingScheduler(start_temp=10.0, end_temp=0.1, total_epochs=30)

for epoch in range(30):
    temp = scheduler.get_temperature(epoch)
    for batch in train_loader:
        pred = model(batch)
        loss = aWTA_loss(
            pred['trajs'], pred['scores'],
            batch['gt_trajs'],
            temperature=temp
        )
        loss.backward()
        optimizer.step()
```

**退火过程**：

| 阶段 | 温度 | 行为 |
|---|---|---|
| 前期（epoch 1-10） | 高（10→2） | 所有 6 条轨迹都参与训练，防模式坍塌 |
| 中期（epoch 10-20） | 中（2→0.5） | 逐渐聚焦，最贴近 GT 的几条权重变大 |
| 后期（epoch 20-30） | 低（0.5→0.1） | 接近标准 WTA，只优化最优那条，精度收敛 |

**实测效果（aWTA 论文，WOMD + MTR）**：

| 指标 | 标准 WTA | aWTA | 改善 |
|---|---|---|---|
| minFDE | 3.22 | 1.34 | -58% |
| MissRate | 0.58 | 0.18 | -69% |
| minADE | 1.27 | 0.63 | -50% |

---

### 优化 3：运动学后处理（P1 · 1 天）

**问题**：MTR++ 可能输出物理不可行轨迹（急转弯、瞬移）。

**方案**：对 6 条轨迹做自行车模型约束过滤。

```python
import numpy as np
import math

def kinematic_filter(trajs, agent_state):
    """
    trajs: (6, 80, 2)  6条预测轨迹
    agent_state: 当前时刻 agent 的状态 (x, y, heading, speed)
    """
    max_steer = math.radians(35)   # 最大转向角
    max_accel = 8.0                # 最大加速度 m/s²
    max_speed = 30.0               # 最大速度 m/s

    feasible = []
    for traj in trajs:
        # 从 (x, y) 序列反算速度/加速度/曲率
        velocities = np.diff(traj, axis=0) * 10  # 10Hz
        speeds = np.linalg.norm(velocities, axis=1)
        headings = np.arctan2(velocities[:, 1], velocities[:, 0])

        # 检查约束
        speed_ok = speeds.max() < max_speed
        accel_ok = np.abs(np.diff(speeds)).max() < max_accel * 0.1
        steer_ok = np.abs(np.diff(headings)).max() < max_steer

        if speed_ok and accel_ok and steer_ok:
            feasible.append(traj)

    if not feasible:
        return trajs  # 全部不满足则不过滤

    return np.array(feasible)
```

**效果**：直接改善 Miss Rate 和 Overlap Rate，高速长时程场景尤其有效。

---

### 优化 4：TTA 测试增强（P1 · 半天）

**方案**：对输入场景做 4 种增强，分别推理，聚类融合。

```python
import numpy as np

def tta_inference(model, scene):
    """4 种增强：原图 + 水平翻转 + 垂直翻转 + 180°旋转"""
    augmentations = [
        lambda x: x,                          # 原图
        lambda x: flip_horizontal(x),         # 水平翻转
        lambda x: flip_vertical(x),           # 垂直翻转
        lambda x: rotate_180(flip_horizontal(x)),  # 180°旋转
    ]

    all_preds = []
    for aug in augmentations:
        aug_scene = aug(scene)
        pred = model(aug_scene)
        # 反变换回原始坐标系
        pred = inverse_transform(pred, aug)
        all_preds.append(pred)

    # 聚类融合 6 条轨迹
    return cluster_and_merge(all_preds, k=6)


def cluster_and_merge(all_preds, k=6):
    """从 4×6=24 条轨迹中聚类出 6 条"""
    from sklearn.cluster import KMeans

    all_trajs = np.concatenate([p['trajs'] for p in all_preds], axis=0)  # (24, 80, 2)
    all_scores = np.concatenate([p['scores'] for p in all_preds], axis=0)  # (24,)

    # 展平用于聚类
    flattened = all_trajs.reshape(24, -1)  # (24, 160)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(flattened)

    # 每个聚类取分数最高的那条
    final_trajs = []
    final_scores = []
    for i in range(k):
        mask = labels == i
        cluster_scores = all_scores[mask]
        best_idx = cluster_scores.argmax()
        final_trajs.append(all_trajs[mask][best_idx])
        final_scores.append(cluster_scores[best_idx])

    # 归一化置信度
    final_scores = np.array(final_scores)
    final_scores = final_scores / final_scores.sum()

    return {
        'trajs': np.array(final_trajs),    # (6, 80, 2)
        'scores': final_scores,             # (6,)
    }
```

**效果**：mAP +1-2 点。代价：推理时间 ×4，离线评测可接受。

---

### 优化 5：Intention Query 动态调整（P2 · 2-3 天）

**问题**：固定 64 query 在不同场景不够灵活。

**方案**：加轻量级场景复杂度估计器，动态选 query 子集。

```python
def estimate_complexity(scene):
    """根据 agent 数量和交互密度估计场景复杂度"""
    num_agents = len(scene['tracks_to_predict'])
    num_interactions = count_interactions(scene)  # 50m 内有交互的 agent 对数
    avg_speed = np.mean([get_speed(a) for a in scene['agents']])

    # 高速 + 密集 = 复杂
    complexity = num_agents * 0.3 + num_interactions * 0.5 + avg_speed * 0.2
    return complexity


def select_query_count(complexity):
    """根据复杂度动态选 query 数量"""
    if complexity < 10:       # 高速直行
        return 16
    elif complexity < 30:     # 普通城区
        return 64
    else:                     # 密集路口
        return 128


# 推理时
complexity = estimate_complexity(scene)
num_queries = select_query_count(complexity)
# 只激活前 num_queries 个 intention query
model.decoder.num_active_queries = num_queries
```

| 场景 | query 数量 | 效果 |
|---|---|---|
| 高速直行 | 16-32 | 防冗余，减少模式坍塌 |
| 普通城区 | 64 | 默认 |
| 密集路口 | 128-256 | 防漏意图 |

---

### 优化 6：场景编码共享（P2 · 1 周+）

**问题**：Agent-centric N 个 agent = N 次重复编码。

**方案**：借鉴 QCNet，scene-level 共享编码。

```python
# MTR++ 原版: agent-centric
class MTREncoder(nn.Module):
    def forward(self, scene, center_agent_idx):
        scene_features = self.encode_scene(scene, center=center_agent_idx)
        return scene_features  # 对每个 agent 要跑一遍

# 借鉴 QCNet: scene-level 共享
class SharedSceneEncoder(nn.Module):
    def forward(self, scene):
        # 1. 编码一次
        agent_features = self.encode_agents(scene['tracks'])
        map_features = self.encode_map(scene['map_infos'])

        # 2. Factorized attention 融合（三路解耦）
        agents = self.temporal_attn(agent_features)               # 时间维度
        agents = self.agent_map_cross_attn(agents, map_features)   # agent-map 交互
        agents = self.agent_agent_attn(agents)                     # agent-agent 交互
        return agents  # 所有 agent 共享

    def encode_agents(self, tracks):
        # 极坐标 + 傅里叶特征（QCNet 风格）
        pos = tracks[:, :, :2]  # (N, T, 2)
        r, theta = self.to_polar(pos)
        feat = torch.cat([
            self.fourier_embed(r),
            self.fourier_embed(theta)
        ], dim=-1)
        return feat
```

**效果**：
- 计算量从 O(N × scene_size) 降到 O(scene_size + N × query_size)
- 显存大幅降低，batch size 可从 4 提到 10+
- 推理速度提升 5-10 倍

**代价**：改动较大，需重写 encoder。但 decoder 不用改。

---

## 三、优化优先级总表

| 优先级 | 优化 | 投入 | 收益 | 阶段 |
|---|---|---|---|---|
| **P0** | 置信度校准 | 半天 | mAP +1-2 | 训练后 |
| **P0** | aWTA Loss | 1 天 | minFDE -58%，MR -69% | 训练中 |
| P1 | 运动学后处理 | 1 天 | Miss Rate 改善 | 推理后 |
| P1 | TTA 增强 | 半天 | mAP +1-2 | 推理时 |
| P2 | Query 动态调整 | 2-3 天 | 场景适应性 | 推理时 |
| P2 | 场景编码共享 | 1 周+ | 显存/速度 | 架构改造 |

**P0 两项必做**：置信度校准直接改善 mAP（比赛主指标），aWTA Loss 在 WOMD 上实测 minFDE 降 58%、MissRate 降 69%。投入 1-2 天，收益最大。

---

## 四、模型结构优化（借鉴 QCNet）

### 可借鉴的方向

| QCNet 设计 | 能否借鉴 | 原因 |
|---|---|---|
| 场景编码共享 | 能 | 直接解决 MTR++ 最大瓶颈 |
| 极坐标 + 傅里叶 | 能 | 提升泛化性，改动小 |
| 流式推理复用 | 看场景 | 比赛离线评测用不上，车端有用 |
| QCNet 的解码器 | 不能借鉴 | 两套设计哲学不同，混用打折扣 |

**关键原则**：借鉴编码器，不借鉴解码器。MTR++ 的 intention query + refinement 两阶段解码器是核心优势，保持不动。

### 改造后架构

```
MTR++ 原版:
  Agent-centric Encoder × N → Intention Query Decoder × N

改造后（MTR + QCNet 编码器）:
  Shared Scene Encoder（QCNet 风格）→ Intention Query Decoder × N（MTR++ 风格）
```

### 改造优先级

| 改造 | 工作量 | 收益 | 风险 | 建议 |
|---|---|---|---|---|
| 极坐标 + 傅里叶 | 2-3 天 | 泛化性提升 | 低 | 先做 |
| 场景编码共享 | 1-2 周 | 显存/速度大幅改善 | 中 | 第二步 |
| 流式推理 | 1 周 | 推理 13ms | 高 | 暂缓 |

---

## 五、预期效果汇总

| 阶段 | mAP | 怎么达到 |
|---|---|---|
| 直接复现 | 42-43 | 官方配置训练，不改任何东西 |
| 加置信度校准 | 43-44 | 温度缩放 / Platt Scaling |
| 加 aWTA Loss | 44-45 | 退火 WTA 替换标准 WTA |
| 加 TTA | 45-46 | 4 种增强推理 + 聚类融合 |
| 加 ensemble | 46+ | 多模型集成（工程量大） |

**底线**：先用 MTR++ 原版 + P0 优化拿到 baseline（mAP 44-45），有时间再叠加 P1/P2。结构改造（借鉴 QCNet）主要改善显存和速度，不是直接提升 mAP——比赛 mAP 天花板还是靠 loss 优化和 TTA。
