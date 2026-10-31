# Windows 本机多窗口训练

一个中央 PPO 为所有游戏收集数据并更新同一模型，各游戏由独立 worker 控制。启动多个训练器不能代替多环境采样，并会争用游戏。

## 游戏目录

准备两份已安装 BepInEx 和当前 Mod 的游戏目录，各自包含 `hollow_knight.exe` 与 `BepInEx/plugins/HollowRL/HollowRL.dll`。游戏目录隔离日志和配置，同一 Windows 账户的存档目录仍共享。先备份存档，使用训练模板并关闭 Steam Cloud。启动器提供 `steam_appid.txt`，检查游戏进程和监听端口。

```powershell
.\launch_multi_games.ps1 `
  -GameDirs 'D:\HollowRL-instances\env1','D:\HollowRL-instances\env2' `
  -Ports 9001,9002
```

默认 640×360、30 FPS、1 倍速，后台继续运行。可用 `-Fps 60 -Width 1280 -Height 720` 调整显示。各窗口加载 1 号存档，站到叛徒领主雕像前；训练器自动进战并等待可战斗状态。

## 中央训练

```powershell
.\run_multi_windows.ps1 -Ports 9001,9002 `
  -RunDir python\runs\traitor_local -Lr 1e-4 -EntCoef 0.005 `
  -EvalEvery 25 -EvalEpisodes 10
```

此命令使用默认难度和奖励；显式辐辉使用 [README](README.md) 的 `main_multi.py` 命令。`-NoMonitor` 可禁用嵌入式 GUI，`-StepsPerEnv` 调整每环境采样长度。默认双环境各 1024 步、四环境各 512 步，合计每次更新 2048 步。

同一运行目录自动恢复最新完整 checkpoint；新实验使用新目录。推理专用权重不包含完整续训的优化器状态。

## 日志与性能

- `training_metrics.jsonl`：完整训练回合、更新和评估汇总。
- `eval_metrics.jsonl`：固定贪心策略逐局评估。
- `ppo_checkpoint.pth`：最近的完整训练状态。
- `best_eval_checkpoint.pth`、`best_eval.json`：评估选出的模型及成绩。
- `run_config.json`：实际参数。

```powershell
.\python\.venv\Scripts\python.exe python\watch_training.py --run-dir python\runs\traitor_local
```

先观察总 step/s、内存、各窗口 FPS、重置耗时与重试数，再增加实例。某窗口重置时其他 worker 可继续采样，但所有游戏仍共享硬件资源。重置重试持续增加时先处理故障。

rollout 期间权重固定，GAE 沿各环境时间轴计算后合并，归一化、优化器与 checkpoint 由中央维护。评估打断的未完成训练回合记为 `episode_interrupted`，不混入完整回合平均值。
