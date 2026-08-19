# MTR 优化方案：融合 QCNet + DiffSemanticFusion 创新点

> 目标指标：**minADE ↓, minFDE ↓, mAP ↑**
> Baseline: 3s/5s/8s mAP=0.420/0.405/0.328, minADE=0.323/0.472/0.687, minFDE=0.589/0.909/1.400

---

## 一、三项目架构对比与核心差异

| 维度 | MTR (基线) | QCNet (借鉴) | DiffSemanticFusion (借鉴) |
|------|-----------|-------------|--------------------------|
| **坐标编码** | 绝对坐标(centered) | Query-centric相对坐标(SE(2)等变) | 同QCNet |
| **轨迹参数化** | 绝对位置GMM (mu, std, rho) | 累积位移+单调增长不确定性 | 扩散去噪 |
| **损失函数** | 硬argmin选mode + GMM NLL | **软logsumexp选mode + Laplace NLL** | MSE扩散loss |
| **Mode选择** | 最近意图点hard assignment | Mixture NLL logsumexp (soft) | N/A |
| **地图编码** | Polyline单token | **点/多边形双层级** | 扩散去噪地图特征 |
| **解码器** | 6层迭代精炼 | **两阶段propose-refine** | 同QCNet |
| **多模态** | 64查询→NMS→6 | 可学习mode embedding + m2m竞争 | 扩散采样多样性 |
| **不确定性** | GMM固定log_std范围 | cumsum单调增长scale | 金字塔噪声多尺度 |

---

## 二、优化方案（按优先级排序）

### P1: Mixture NLL 损失 — 软模式选择（最高优先级）

**问题**：MTR使用 `center_gt_positive_idx = argmin(dist)` 硬选最近意图点为正样本。
- 硬选导致梯度只流向一个mode，其他mode无学习信号
- 意图点离GT稍远时，正样本选择不稳定
- 这直接恶化 minFDE（终点误差）因为模式选择不准

**QCNet方案**：Mixture NLL 用 `logsumexp` 软选择：
```python
# MTR当前: 硬选
loss_reg_gmm = nll_loss_gmm_direct(..., pre_nearest_mode_idxs=center_gt_positive_idx)

# QCNet方案: 软选
nll_per_mode = compute_nll_per_mode(pred_trajs, gt_trajs)  # (N, num_modes)
log_pi = F.log_softmax(pred_scores, dim=-1)  # (N, num_modes)
loss = -torch.logsumexp(log_pi - nll_per_mode, dim=-1)  # soft min over modes
```

**改动位置**：`mtr/utils/loss_utils.py` 新增 `mixture_nll_loss` 函数；`mtr/models/motion_decoder/mtr_decoder.py` 的 `get_decoder_loss` 方法

**预期收益**：minFDE 降低 3-5%（软选择使所有mode都有梯度，提升模式覆盖）

---

### P2: EMA 权重 + 多Checkpoint选优（零风险）

**问题**：训练末期权重震荡，单checkpoint可能不是最优点

**方案**：
1. **EMA (Exponential Moving Average)**：维护权重滑动平均，推理时用EMA权重
   ```python
   ema_decay = 0.999
   ema_model = copy.deepcopy(model)
   # 每step更新
   for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
       p_ema.data.mul_(ema_decay).add_(p_model.data, alpha=1 - ema_decay)
   ```
2. **多checkpoint评估**：保存最后5个epoch的checkpoint，分别评估minADE/minFDE，选最优

**改动位置**：`tools/train_utils/train_utils.py` 加EMA逻辑；`tools/test.py` 加多ckpt评估

**预期收益**：minADE/minFDE 各降低 1-2%（平滑权重减少方差）

---

### P3: 累积位移轨迹参数化（中等风险）

**问题**：MTR直接回归80帧绝对位置，远端帧（8s）误差累积大

**QCNet方案**：预测逐步位移，cumsum还原位置：
```python
# 当前MTR: 直接回归绝对位置
pred_trajs = motion_reg_head(query_content)  # (N, 64, 80, 7) 绝对xy + GMM

# QCNet方案: 回归位移再cumsum
per_step_delta = motion_reg_head(query_content)  # (N, 64, 80, 2) 每步Δx,Δy
pred_positions = torch.cumsum(per_step_delta, dim=2)  # 累积求和
# 不确定性也cumsum（单调增长 = 远端不确定性自然增大）
scale = torch.cumsum(F.elu(scale_raw) + 1, dim=2) + 0.1
```

**改动位置**：`mtr/models/motion_decoder/mtr_decoder.py` 的 `motion_reg_heads` 输出处理

**预期收益**：8s minFDE 降低 5-8%（位移预测更稳定，远端不确定性更合理）

---

### P4: Query-Centric 相对地图特征（中等风险）

**问题**：MTR用绝对centered坐标编码地图，对全局旋转敏感

**QCNet方案**：所有空间关系在目标节点局部坐标系下表达：
```python
# 相对特征 = [距离, 相对角度(到自身heading), Δheading]
rel_pos = pos_src - pos_dst
r = torch.stack([
    torch.norm(rel_pos[:, :2], dim=-1),                    # 距离
    angle_between_2d_vectors(orient_dst, rel_pos[:, :2]),  # 相对角度
    wrap_angle(orient_src - orient_dst)                     # Δheading
], dim=-1)
```

**改动位置**：`mtr/models/context_encoder/mtr_encoder.py` 的位置编码；`mtr/models/utils/transformer/position_encoding_utils.py`

**预期收益**：提升旋转鲁棒性，minADE 降低 2-3%

---

### P5: 扩散轨迹精炼（创新点，高收益高风险）

**问题**：MTR的6条轨迹由确定性decoder生成，多样性不足，尤其minFDE依赖轨迹终点覆盖

**DiffSemanticFusion方案**：用DDPM扩散模型对MTR预测进行精炼
1. **条件扩散**：以MTR的确定性预测为 `local_cond`（条件）
2. **金字塔噪声**：多分辨率噪声代替i.i.d.高斯，产生时间相关扰动
3. **少步cosine DDPM**：20步，推理可行

```python
# 金字塔噪声（来自DiffSemanticFusion）
def pyramid_noise_like(x, discount=0.9):
    b, c, t = x.shape
    noise = torch.zeros_like(x)
    for i in range(4):  # 多尺度
        scale = t // (2 ** i)
        if scale > 0:
            coarse = torch.randn(b, c, scale, device=x.device)
            noise += F.interpolate(coarse, size=t) * (discount ** i)
    return noise / noise.std()

# 扩散精炼
# 1. MTR输出 6 条轨迹作为条件
# 2. 扩散模型学习 "给定MTR预测，生成更优/更多样的轨迹"
# 3. 推理时多次采样增加多模态覆盖
```

**改动位置**：新增 `mtr/models/diffusion/` 模块；`mtr/models/model.py` 集成

**预期收益**：minFDE 降低 5-10%（扩散采样增加终点覆盖），mAP可能小幅提升

---

### P6: Laplace NLL 替换 GMM NLL（低风险）

**问题**：MTR用双变量高斯GMM，参数多（5维: mu_x, mu_y, log_std_x, log_std_y, rho），rho收敛慢

**QCNet方案**：Laplace分布更简单（loc + scale），对异常值更鲁棒：
```python
# Laplace NLL = log(2*scale) + |y - loc| / scale
nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
```

**改动位置**：`mtr/utils/loss_utils.py`

**预期收益**：minADE 降低 1-2%（L1-based损失对异常值更鲁棒）

---

## 三、实施路线图

```
Phase 1 (安全, 1-2天):
  ├── P2: EMA权重 + 多ckpt选优     [零代码风险]
  ├── P6: Laplace NLL替换GMM       [改loss_utils.py]
  └── P1: Mixture NLL软模式选择    [改loss_utils.py + decoder]

Phase 2 (核心创新, 3-5天):
  ├── P3: 累积位移参数化           [改decoder输出]
  ├── P4: Query-centric相对特征    [改encoder位置编码]
  └── P1+P3+P6 联合训练验证

Phase 3 (高收益创新, 5-7天):
  ├── P5: 扩散轨迹精炼模块          [新增diffusion模块]
  ├── P5集成测试
  └── 最终提交: P1+P2+P3+P5联合
```

---

## 四、创新点总结（用于赛事文档）

1. **软模式选择的Mixture NLL损失**：借鉴QCNet，用logsumexp替代hard argmin，使所有候选轨迹模式都有梯度信号，提升多模态覆盖率和minFDE
2. **累积位移轨迹参数化**：借鉴QCNet，预测逐步位移再cumsum还原，每步只学小delta，不确定性随时间单调增长，改善远端(8s)预测精度
3. **EMA权重平滑+多checkpoint选优**：减少训练末期权重震荡，直接降低minADE/minFDE方差
4. **Laplace分布NLL**：比高斯GMM更鲁棒的回归损失，减少异常值影响
5. **金字塔噪声扩散轨迹精炼**：借鉴DiffSemanticFusion，用多分辨率噪声DDPM对确定性预测进行扩散精炼，增加轨迹终点覆盖度，直接降低minFDE

---

## 五、风险控制

- 所有改动保持向后兼容（config开关控制）
- 每个优化点独立可验证（可单独ablation）
- 保留原始baseline分支，改动在新branch上
- 每次改动后用小数据集smoke test验证不崩溃
