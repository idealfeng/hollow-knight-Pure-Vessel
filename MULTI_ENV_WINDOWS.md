# Windows 本机多窗口训练

中央 PPO 为所有实例收集数据并更新同一模型，各游戏由独立 worker 控制。多个训练器争抢端口不等于正确的多环境采样。

## 启动游戏

准备两份安装同版 BepInEx 和 Mod 的游戏目录，各含 `hollow_knight.exe` 和 `BepInEx/plugins/HollowRL/HollowRL.dll`。同一 Windows 账户仍共享存档目录；先备份，安装标准模板并重新加载。

```powershell
.\launch_multi_games.ps1 `
  -GameDirs 'D:\HollowRL-instances\env1','D:\HollowRL-instances\env2' `
  -Ports 9001,9002 -Fps 60 -GameSpeed 1
```

通用启动器默认 640×360、60 FPS、1 倍速，后台继续运行。可用 `-Width 1280 -Height 720` 调整显示。端口已有进程时检查目录、Boss 和启动帧率/速度，参数不同拒绝复用，不自动关闭或重启。改脚本不修改已运行进程。

各窗口加载骨钉 21、无战斗护符、三法术存档，站到雕像前，不必手动选难度。

## 从完整 650 续训

```powershell
.\train_spells.ps1 `
  -Endpoints 127.0.0.1:9001,127.0.0.1:9002 `
  -InitCheckpoint models\traitor_lord_radiant.pth `
  -RunDir python\runs\traitor_local `
  -FrameStack 16 -RewardProfile radiant_spells_dodge_v1 `
  -Lr 0.0001 -EntCoef 0.005 -FreezeObsNorm `
  -EvalEvery 25 -EvalEpisodes 10 -NoMonitor
```

该预设为原生辐辉、1 倍速、名义 10 Hz、每环境 512 步；双窗每次 1024 条，四窗每次 2048 条。固定评估每窗 10 局。完整模型恢复 Adam 和归一化，同目录恢复最新状态，另开实验使用新目录。

通用 `run_multi_windows.ps1` 支持更多参数，但其新运行帧堆叠/学习率默认值不等于三法术 650 预设；复用 650 优先使用上述命令并记录实际设置。

## 日志与性能

```powershell
.\python\.venv\Scripts\python.exe python\watch_training.py --run-dir python\runs\traitor_local
```

面板只读日志。`training_metrics.jsonl` 含完整回合、更新与评估摘要，`eval_metrics.jsonl` 为固定评估逐局结果，`run_config.json` 为实际配置。`ppo_checkpoint.pth` 是最新完整训练状态，`best_eval_checkpoint.pth` 和 `best_eval.json` 为评估选出的模型与成绩。

10 Hz 与游戏 FPS 独立：控制器等待约 100 ms，另外叠加命令、推理和调度开销，并非严格每六个游戏帧决策一次。优先保持各实例实际帧率稳定、训练与评估条件一致，再考虑增加窗口。

某窗口重置时其他 worker 可继续采样，但同机共享硬件资源。采集期间模型固定，GAE 沿各环境时间轴计算后合并；评估打断的未完成训练回合单列 `episode_interrupted`，不混入完整回合平均值。

原生难度和重置见 [FAST_RESET.md](FAST_RESET.md)，跨机器见 [MULTI_MACHINE.md](MULTI_MACHINE.md)。不要让两个控制器接管相同端口；运行中从日志读状态，独立端点检查需等控制器退出。
