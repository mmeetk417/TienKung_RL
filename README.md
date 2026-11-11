# 天工机器人强化学习抓取

基于 **MuJoCo + PyTorch + PPO**，在 CPU 上训练天工 2 Pro 固定基座右臂与右侧灵巧手完成桌面抓取，无需 Isaac Sim 或 CUDA。

策略使用 106 维状态，学习控制右臂的 7 个关节；到达抓取位姿并稳定停留后，环境自动执行预设的闭手、保持、抬升和保持流程。**手指动作由固定流程控制，不是端到端学习。**

## 1. 安装

支持 Ubuntu 22.04 x86-64，需先安装 Conda。在项目根目录执行：

```bash
bash install.sh
conda activate tienkung-rl
```

已安装过环境时，只需执行 `conda activate tienkung-rl`。以下命令均在项目根目录运行。
首次安装获取 Conda 索引和解析依赖可能较慢；`environment.lock.yml` 是带版本约束的环境文件，仍需要依赖求解。

## 2. 训练

直接使用默认配置训练，不需要额外参数：

```bash
python -m tienkung_rl.train
```

默认参数：128 个环境、8 个仿真工作线程、种子 42、1000 次迭代，每 10 次保存权重。训练不打开窗口；CPU 训练可能耗时较长，一次迭代包含每个环境 1280 步采样。

输出位于 `runs/reach_then_close/`：

| 路径 | 内容 |
|---|---|
| `checkpoints/model_final.pt` | 训练完成后的权重 |
| `checkpoints/model_000010.pt` 等 | 中间权重，可用于提前查看效果 |
| `tensorboard/` | 训练曲线 |
| `config/` | 本次运行配置快照 |

查看曲线：

```bash
tensorboard --logdir runs/reach_then_close/tensorboard
```

重新从头训练需换一个输出目录，已有非空目录不会被覆盖：

```bash
python -m tienkung_rl.train --output-dir runs/experiment_02
```

可选的无窗口运行检查（不训练策略）：

```bash
python -m tienkung_rl.smoke --num-envs 2
```

## 3. 推理可视化

不加载权重，先查看场景：

```bash
python -m tienkung_rl.viewer
```

训练完成后，加载策略并打开 MuJoCo 窗口：

```bash
python -m tienkung_rl.viewer --mode policy --checkpoint runs/reach_then_close/checkpoints/model_final.pt
```

可将权重路径替换成中间权重或其他实验权重。当前 `.pt` 文件已包含元数据，可单独复制；旧版本权重可能仍需同名 JSON 文件。

常用可选参数：`--hide-debug-geometry --hide-overlays` 隐藏诊断显示，`--pause-on-terminal` 停留在回合结束姿态。`--mode scripted` 用于查看预设抓取流程，不代表策略推理效果。

## 4. 无窗口评估

评估入口需要显式指定权重和评估集：

```bash
python -m tienkung_rl.evaluate \
  --checkpoint runs/reach_then_close/checkpoints/model_final.pt \
  --manifest evaluation/manifests/validation.json
```

报告保存到 `runs/reach_then_close/evaluation/`。正式评估可换用同目录下的 `final_standard_256.json` 或 `final_randomized_256.json`；日常调参使用验证集。

## 5. 修改参数

默认读取 [`configs/reach_then_close.yaml`](configs/reach_then_close.yaml)，不传参数即可使用默认值。常用设置如下：

| YAML 字段 | 用途 / 默认值 |
|---|---|
| `platform.num_envs` | 环境数，128 |
| `platform.worker_threads` | 仿真工作线程数，8 |
| `run.maximum_iterations` | 训练迭代数，1000 |
| `run.save_interval_iterations` | 保存间隔，10 |
| `run.seed` / `run.output_dir` | 种子 42 / 输出目录 |
| `ppo` | 网络、学习率、每轮采样步数等 |
| `task.reward` | 奖励权重 |
| `task.admission` | 自动闭手的位置、姿态、停留门槛 |
| `task.option` | 闭手与保持时间、抬升高度等 |
| `physics` | 仿真、接触与执行器参数 |
| `randomization` / `curriculum` | 随机化范围 / 课程难度 |

命令行优先于 YAML，临时调整示例：

```bash
python -m tienkung_rl.train --num-envs 16 --max-iterations 100
python -m tienkung_rl.train --set ppo.actor_learning_rate=0.0002
```

`--set 字段路径=值` 可重复使用，修改不同字段。常用快捷参数还有 `--seed`、`--worker-threads`、`--save-interval` 和 `--output-dir`。

长期实验建议复制配置，并修改其中的 `run.output_dir`：

```bash
cp configs/reach_then_close.yaml configs/my_experiment.yaml
# 编辑新配置后运行
python -m tienkung_rl.train --config configs/my_experiment.yaml
```

推理和评估时也追加 `--config configs/my_experiment.yaml`，使用训练对应的配置；网络尺寸和动作定义须与权重匹配。可在运行输出的 `config/` 中查阅实际使用的配置。

完整参数见各入口的 `--help`。修改机器人资产后可执行 `python -m tienkung_rl.validate_asset` 检查；代码修改位置见 [工程结构](docs/architecture.md)。历史 Isaac 权重不适用于当前 MuJoCo 入口。
