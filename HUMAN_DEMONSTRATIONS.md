# 真人示范与行为克隆

Mod 逐帧记录游戏内部状态与逻辑按键，支持键盘和手柄，用于离线行为克隆。这不是屏幕录像。七按键策略覆盖移动、攻击、跳跃和冲刺；记录器还能记录其他能力，但当前策略不会执行全部游戏按键。

## 录制

先停止所有控制目标窗口的训练或试打程序。快捷准备入口会备份旧 Mod，并以 1280×720、60 FPS、1 倍速重启指定游戏：

```powershell
.\prepare_human_recording.ps1 -GameDir 'D:\Games\Hollow Knight' -Port 9001
.\record_human_windows.ps1 -Port 9001 -Episodes 5
```

已经安装当前 Mod 并启动游戏时可跳过准备入口。看到 `Recording to:` 后加载训练存档并手动进战，死亡后手动重试。先录一局检查再补多局，各局保持难度、护符、资源一致。真人录制期间 Mod 拒绝自动输入。

达到指定局数自动结束；终端 `Ctrl+C` 保留已写数据，输出目录中的 `STOP` 文件也可请求停止。

## 数据与训练

默认输出 `python/runs/human_demos/human_9001_日期_时间/`：

- `frames.jsonl`：Unity 帧、时间、按住/按下/松开位掩码与完整状态。
- `episodes.jsonl`：逐局命中、受伤、最低 Boss 血量、胜负和时长。
- `session.json`：格式、按键顺序、帧数、录制状态和丢帧数。

先确认 `dropped_frames` 为 0，保留完整原始逐帧文件。转换器按控制频率下采样，短于一个控制周期的按键可能不被保留；不能视为逐帧操作的无损压缩。

```powershell
.\python\.venv\Scripts\python.exe python\build_human_bc_dataset.py `
  --sessions python\runs\human_demos\<会话目录> `
  --output python\runs\traitor_demos.npz
.\python\.venv\Scripts\python.exe python\train_bc.py `
  --demos python\runs\traitor_demos.npz `
  --run-dir python\runs\traitor_bc --ports 9001 9002
```

行为克隆使用完整战斗，按整局划分训练与验证。检查输出后，从相同目录接 PPO：

```powershell
.\run_multi_windows.ps1 -Ports 9001,9002 -RunDir python\runs\traitor_bc
```

重做预训练使用新目录，避免覆盖策略。示范预训练不保证无伤通关，仍需固定策略实机评估。
