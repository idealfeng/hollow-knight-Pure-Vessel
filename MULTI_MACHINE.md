# 跨机器采样（中央训练器 + 远程游戏）

中央 PPO 训练器运行在主机 A，可以同时连接本机游戏和主机 B 上的游戏。主机 B 只
运行游戏和 HollowRL Mod，不运行 PPO。所有环境共享同一个中央策略，经验进入同一个
rollout buffer，由同一个 optimizer 更新。PPO、奖励函数、观测特征、策略网络、GAE、
PPO 超参数和 checkpoint 格式均不变。

## 端点

每个环境用 `HOST:PORT` 表示。旧 `--ports` 参数仍然可用，自动映射到 `127.0.0.1`。

```text
主机 A 本机游戏   127.0.0.1:9001
主机 B 远程游戏   192.168.1.50:9002
```

## 主机 B：让 Mod 监听局域网地址

在主机 B 上，用可选 `BindIp` 参数启动游戏，让 Mod 监听 `192.168.1.50`：

```powershell
.\launch_multi_games.ps1 -GameDirs "D:\Games\HollowKnightB" -Ports 9002 `
  -BindIp 192.168.1.50 -BossName "Mantis Traitor Lord" `
  -BossScene GG_Traitor_Lord -BossFsm Mantis
```

不传 `-BindIp` 时保持原有行为，监听 `127.0.0.1`。主机 B 的 HollowRL Mod 必须重新
编译并部署（`connection.cs` 现在支持 `--hollowrl-bind-ip=`）。

## 主机 A：启动中央训练器

本机游戏照常用 `launch_multi_games.ps1`（不传 BindIp 即监听 127.0.0.1）启动，然后：

```powershell
.\run_multi_windows.ps1 `
  -Endpoints 127.0.0.1:9001,192.168.1.50:9002 `
  -RunDir python\runs\traitor_remote_9001_9002 `
  -Lr 1e-4 -EntCoef 0.005 -EvalEvery 10 -EvalEpisodes 1
```

或者直接调用：

```powershell
.\python\.venv\Scripts\python.exe python\main_multi.py `
  --endpoints 127.0.0.1:9001 192.168.1.50:9002 `
  --run-dir python\runs\traitor_remote_9001_9002
```

旧的本机双环境命令仍然有效：

```powershell
.\run_multi_windows.ps1 -Ports 9001,9002 -RunDir python\runs\traitor_local
```

## 训练诊断与评估

运行中的训练器已经载入旧代码；更新 Python 文件后，在一次 `upd` 完成并保存 checkpoint
后正常停止，再运行原启动命令，即可加载改动并续训。不要同时启动两个训练器。

```powershell
.\python\.venv\Scripts\python.exe python\summarize_run.py python\runs\traitor_4env
```

四个环境默认各收集 512 步，合计每次 PPO 更新 2048 步。`training_step/s` 包含采样、重置
和 PPO 更新，但不含周期评估；新日志额外记录采样、更新和评估耗时。

周期贪心评估在所有环境中并发执行。评估中断的训练回合记作 `episode_interrupted`，不计入
完整回合平均值或命中早停窗口。2026-10-28 此修复之前的评估边界可能把中断回合统计与
下一局拼接；旧 `eval_metrics.jsonl` 的独立贪心评估成绩仍可使用。

每次启动会保存 `run_config.json` 和一条 `run_start` 日志。`ppo_checkpoint.pth` 保存最新
权重；`best_eval_checkpoint.pth` 保存评估中表现最好的权重，优先比较胜率，再比较平均
Boss 剩余血量和平均回报。`best_eval.json` 记录其对应成绩。少量评估战斗仍有波动，
它表示本轮已测 checkpoint 中的最佳观测值。

## 连通性测试（只读）

`test_remote_endpoints.py` 只发送一次 `pos` 并校验返回帧格式，不发送 KILL、SPEED
或任何输入动作：

```powershell
.\python\.venv\Scripts\python.exe python\test_remote_endpoints.py 127.0.0.1:9001 192.168.1.50:9002
```

## 防火墙

Mod 监听显式指定的 IP（默认 127.0.0.1）。远程模式必须在主机 B 的防火墙放行对应
端口，例如：

```powershell
New-NetFirewallRule -DisplayName "HollowRL 9002" -Direction Inbound `
  -Action Allow -Protocol TCP -LocalPort 9002
```

同时确保 `HOLLOWRL_HOST` / `HOLLOWRL_PORT` 环境变量不冲突。

## 故障排查

- **连不上远程游戏**：先跑连通性测试脚本；检查主机 B 防火墙、Mod 是否用
  `--hollowrl-bind-ip=192.168.1.50` 启动、IP 是否可达。
- **端点格式错误**：`HOST:PORT` 只有一个冒号，端口在 1024..65535；重复端点会报错。
- **远程被静默转成本机**：不会发生——`--endpoints` 逐字保留 host，只有旧
  `--ports` 才映射到 127.0.0.1。
- **断连**：worker 的 state 轮询线程会自动重连；训练器按现有 reset 逻辑处理。
