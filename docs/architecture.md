# 代码结构与阅读顺序

## 从训练入口开始

1. `configs/reach_then_close.yaml`：任务和训练的默认参数。
2. `tienkung_rl/train.py`、`tienkung_rl/_cli.py`：解析命令、加载配置并分派运行。
3. `tienkung_rl/learning/runner.py`：创建策略、采样、PPO 更新、记录和保存。
4. `tienkung_rl/task/vec_env.py`：将多个任务环境转换为批量 CPU Tensor 接口。
5. `tienkung_rl/task/environment.py`：组织单环境重置与每步任务逻辑。
6. `tienkung_rl/sim/parallel.py`、`mujoco_backend.py`：执行物理步骤和读取状态。

## 按需求定位修改位置

| 需求 | 文件 |
|---|---|
| 调整默认超参数 | `configs/reach_then_close.yaml` |
| 增加配置字段 | `tienkung_rl/config/models.py`、`load.py` |
| 改动作映射 | `tienkung_rl/task/actions.py` |
| 改观测 | `tienkung_rl/task/observations.py`、`contracts.py` |
| 改自动闭手条件 | `tienkung_rl/task/admission.py` |
| 改闭手、抬升时序 | `tienkung_rl/task/option.py` |
| 改奖励和结束条件 | `tienkung_rl/task/reward.py`、`termination.py` |
| 改课程及采样范围 | `tienkung_rl/task/curriculum.py`、`randomization.py` |
| 改模型或 PPO | `rsl_rl/rsl_rl/modules/actor_critic.py`、`algorithms/ppo.py` |
| 改训练恢复 | `tienkung_rl/learning/checkpoint.py`、`runner.py` |
| 改评估 | `tienkung_rl/learning/evaluation.py`、`manifests.py` |
| 改日志和报表 | `tienkung_rl/telemetry/` |
| 改显示 | `tienkung_rl/visualization/` |

当前命令行训练使用 `tienkung_rl/learning/runner.py`。
`rsl_rl/rsl_rl/runners/on_policy_runner.py` 是独立 runner，其工具函数仍由算法测试覆盖，
不是当前命令行训练的入口。

## 数据与产物

- `tienkung_rl/assets/tienkung2_pro/source/` 是当前资产的独立源文件，包含 URDF、网格和来源许可。
- `patches/` 定义转换补丁，`mjcf/` 保存生成的模型，资产验证检查关节、碰撞和物理行为，不再比较文件哈希或要求重新生成结果完全一致。
- `evaluation/manifests/` 提供仓库中的评估用例；`learning/manifest_data/` 用于打包分发。
- `tests/fixtures/` 中的历史数值参考用于回归比较，独立于已移除的 Isaac 实现。
- 新训练结果写入 `runs/`，评估及诊断结果写入 `outputs/` 或 `reports/`；这些目录均被 Git 忽略。

## 模块边界

- `paths.py`：统一项目、配置与资产路径，不依赖仿真器或资产生成器。
- `config/`：读取 YAML、应用命令行覆盖、检查参数并保存配置。
- `sim/`：MuJoCo 状态读取和物理步进，不负责奖励或训练。
- `task/`：动作、观测、奖励、终止、课程和自动抓取流程。
- `learning/`：PPO 训练编排、权重读写和评估。
- `telemetry/`、`visualization/`：记录结果和显示场景。
- `assets/pipeline.py`：离线生成资产；运行时模块不应为了路径常量导入它。

## 简化后的保存和检查

checkpoint 的 `.pt` 是完整训练状态；JSON 是便于阅读的导出，不再作为加载门槛。
保存时使用临时文件和原子替换，避免中断写坏权重；不再重新加载验证、计算源代码哈希、
依赖哈希或 payload/JSON 交叉哈希。旧 `.pt` 的 JSON 仍可提供历史元数据。

保留参数范围、Tensor 尺寸、动作语义、NaN/Inf、关节限制和物理模型检查，
以及避免覆盖已有训练结果的保护。它们直接影响训练正确性或数据保存。
资产配置中的旧 `*_sha256` 字段为可选历史信息，默认配置不再填写。
离线生成清单和日志中仍有用于追溯的摘要，但不作为运行时文件完整性门槛。

## 物理参数维护

`configs/reach_then_close.yaml` 是接触与执行器参数的配置入口：
`physics.contact` 设置普通碰撞接触，`physics.cylinder_contact` 设置物体侧的柔和接触响应，
省略后使用普通接触参数。手指网格不额外增加碰撞边距。
修改这些值后，运行 `python -m tienkung_rl.assets.pipeline --generate` 更新两个 MJCF，
再运行 `python -m tienkung_rl.validate_asset` 检查物理行为。
手指位置刚度当前为 10 N·m/rad，力矩上限仍为 5 N·m；这些仍属于仿真初始标定值。

评估用例已按当前方柱尺寸范围（外接圆半径 0.010–0.015 m）更新，仓库与打包数据保持一致。
更新前后的最终评估用例不同，结果应注明使用的评估集，不能直接比较。
