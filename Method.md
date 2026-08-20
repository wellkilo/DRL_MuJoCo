# DRL MuJoCo 方法契约

本文记录训练内部方法、输入输出和必须保持的数学契约。若修改 Actor、GAE、归一化或多 GPU 拓扑，必须同步更新本文与 `tests/`。

## 训练调用链

```text
main.py <config.yaml>
  -> load_config
  -> ray.init
  -> resolve_training_topology
  -> MuJoCoActor.sample
  -> ReplayBuffer.add
  -> Learner.train_step
  -> ParameterServer 参数/观测统计同步
  -> CSV metrics + PyTorch checkpoint
```

## 配置加载

### `load_config(path: str) -> Config`

- 实现：`drl/config_loader.py`
- 输入：YAML 文件路径。
- 输出：不可变 `Config`。文件未提供的字段使用 dataclass 默认值。
- 错误：未知字段或类型不能构造 `Config` 时抛出异常，由 `main.py` 记录并退出。

关键拓扑字段：

| 字段 | 类型 | 契约 |
|---|---|---|
| `num_actors` | `int` | 配置请求的 Actor 总数，必须为正数 |
| `num_gpus` | `int` | 配置请求的 Learner/GPU 数，必须为正数 |
| `actors_per_gpu` | `int | null` | 省略时由 `num_actors / num_gpus` 推导；显式设置时三者必须相等 |
| `param_sync_interval` | `int` | Learner 参数平均间隔，必须为正数 |

### `resolve_training_topology(...) -> TrainingTopology`

- 实现：`drl/training_utils.py`
- 保留旧单机配置语义：`num_actors: 1`、`num_gpus: 1` 必须启动 1 个 Actor。
- 多 GPU 配置中，`num_actors` 必须等于 `num_gpus * actors_per_gpu`。
- 可用 GPU 少于请求值时降低 Learner 数量，每个有效 Learner 的 Actor 数保持不变。
- 没有 GPU 时使用一个 CPU/MPS Learner。

## Rollout 数据契约

### `prepare_action(sampled_action, action_low, action_high)`

- 实现：`drl/rollout_utils.py`
- 返回 `(policy_action, env_action)`。
- `policy_action` 是 Normal 分布直接采样值，写入轨迹并用于 PPO log-prob。
- `env_action` 仅用于 `env.step`，裁剪到 MuJoCo 动作空间。
- 禁止使用裁剪后的动作搭配裁剪前的旧 log-prob，否则同策略 PPO ratio 不为 1。

### `MuJoCoActor.sample(...)`

输入：

```text
state_dict: 当前所属 Learner 的参数
rollout_length: 本轮环境步数
gamma: 折扣因子
gae_lambda: GAE lambda
obs_rms_state: 上轮结束后的全局观测统计
```

输出：

```text
trajectory: [{obs, act, logp, value, adv, ret}, ...]
stats: {actor_id, episodes, episode_return_sum, episode_returns,
        episode_len_sum, obs_rms_moments}
```

整个 rollout 使用同一份 `obs_rms_state`。Actor 只上报本轮原始观测的增量 moments，不上报包含全局历史的累计状态。

### `compute_gae(...) -> (advantages, returns)`

- 实现：`drl/rollout_utils.py`。
- `terminated=True`：任务自然终止，bootstrap mask 为 0。
- `truncated=True`：TimeLimit 等外部截断，使用截断前 `next_obs` 的 value bootstrap。
- 终止和截断都会切断 GAE 跨 episode 传播，防止 reset 后的新 episode 影响上一条轨迹。

## 观测统计契约

### `compute_batch_moments(samples)`

返回一批新观测的 `mean`、`var`、`count`；输入不得为空或包含非有限值。

### `merge_rms_states(states)`

只允许合并互不重叠的数据 moments。主进程先合并同轮所有 Actor 增量，再由 `ParameterServer.update_obs_rms` 与全局历史合并一次。

## Learner 与同步

### `Learner.train_step(updates)`

1. 读取所属 ReplayBuffer 的完整 on-policy 数据。
2. 全局标准化 advantage。
3. 执行 PPO mini-batch 更新、梯度裁剪和 KL 早停。
4. 清空 Buffer。
5. 返回 `state_dict` 和训练指标。

当 `param_sync_interval > 1` 时，非同步轮的 Actor 必须使用其所属 Learner 的本地参数采样；同步轮平均所有 Learner 参数并广播。展示指标通过 `average_metrics` 聚合所有 Learner，而不是只读取第一个 Learner。

## 输出契约

- Metrics：由配置 `metrics_path` 指定，训练启动时覆盖旧文件。
- 当前模型：`output/model_<config>.pt`。
- 最佳模型：`output/model_<config>_best.pt`。
- Checkpoint JSON 等价结构：`{"actor": state_dict, "obs_rms_state": state | null}`。
- `buffer_size` 在 Learner 清空前采集，表示本轮真实训练样本量。

## 验证命令

```bash
python -m unittest discover -s tests -v
python -m compileall -q main.py drl web/server.py scripts tests
bash -n scripts/slurm/*.sh
cd web && npm run build
```
