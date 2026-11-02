# 两台电脑共同采样

A 运行中央 PPO、推理和本机游戏，B 只运行游戏与 Mod，不需要 Python 训练环境。所有端点共享一个模型和优化器。下列 `192.168.1.50` 是示例，请换成 B 实际局域网地址。

## B 主机

在两份游戏目录安装同版 Mod，显式以 60 FPS、1 倍速启动：

```powershell
.\launch_multi_games.ps1 `
  -GameDirs 'D:\Games\HollowKnight3','D:\Games\HollowKnight4' `
  -Ports 9003,9004 -BindIp 192.168.1.50 -Fps 60 -GameSpeed 1
```

不指定 `BindIp` 时只监听 `127.0.0.1`，A 无法通过局域网访问。B 防火墙放行相应 TCP 端口并限制在可信局域网。各窗口加载同一标准存档，站到雕像前。

本项目收尾发现 B 旧进程仍请求 30 FPS，A 请求 60。现在启动器默认 60，已有进程参数不符时拒绝复用；**B 旧 30 帧进程必须关闭后再启动，改脚本或更新 DLL 不会追溯修改运行参数**。游戏菜单可再次修改上限，应核对游戏内帧率和垂直同步；启动目标不等于实测帧率。

## A 主机

按 [本机多窗口指南](MULTI_ENV_WINDOWS.md) 启动 A 的 9001、9002，然后执行：

```powershell
.\train_spells.ps1 `
  -Endpoints 127.0.0.1:9001,127.0.0.1:9002,192.168.1.50:9003,192.168.1.50:9004 `
  -InitCheckpoint models\traitor_lord_radiant.pth `
  -RunDir python\runs\traitor_remote `
  -FrameStack 16 -RewardProfile radiant_spells_dodge_v1 `
  -Lr 0.0001 -EntCoef 0.005 -FreezeObsNorm `
  -EvalEvery 25 -EvalEpisodes 10 -NoMonitor
```

四环境各 512 步，每次更新 2048 条；固定评估各 10 局，共 40 局。`Endpoints` 包含 host 与 port，旧 `Ports` 只指本机端口。相同完整目录重启恢复最新检查点，不要同时启动两个训练器。

## 检查与更新

运行中从 A 的日志或 `watch_training.py` 查看状态。独立端点检查会新建并关闭连接，可能触发 Mod 清理输入，先停止控制这些游戏的任务再运行：

```powershell
.\python\.venv\Scripts\python.exe python\check_endpoints.py 127.0.0.1:9001 192.168.1.50:9003
```

Python 奖励、网络或动作策略修改通常只需更新 A；修改 Mod 命令、导出字段或原生处理时，需要关闭两台游戏、更新 DLL、重新启动。更换存档也要关闭游戏并重新加载，不能只复制磁盘文件就认为窗口已生效。

同一权重的历史有效决策间隔在 A 约 104 ms、B 约 110 ms；同步串行命令、网络和游戏 `Update` 相位都会改变执行时序。帧率启动配置差异已证实，但没有完成统一实际 FPS 的交叉试验，不能单归因于某台硬件更差。

后续复现应固定模型 SHA、输入、执行器、存档、原生难度、速度和实际 FPS，分别报告各端口固定局数结果。详细证据与首步观测新鲜度限制见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)，重置见 [FAST_RESET.md](FAST_RESET.md)。
