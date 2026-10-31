# 跨电脑采样

主机 A 运行中央 PPO 和本机游戏，主机 B 只运行游戏与 Mod。所有端点共享一个策略和 optimizer，B 不需要 Python 训练环境。

## 主机 B

安装同一版本 Mod，用 B 的局域网 IP 启动，例如：

```powershell
.\launch_multi_games.ps1 `
  -GameDirs 'D:\Games\HollowKnight3','D:\Games\HollowKnight4' `
  -Ports 9003,9004 -BindIp 192.168.1.50
```

把示例地址换成实际 IP。不指定 `BindIp` 时只监听 `127.0.0.1`。B 防火墙放行选定 TCP 端口，并限制在可信局域网。各窗口加载训练存档，站到叛徒领主雕像前。

## 主机 A

本机窗口按 [本机多窗口指南](MULTI_ENV_WINDOWS.md) 启动，再执行：

```powershell
.\run_multi_windows.ps1 `
  -Endpoints 127.0.0.1:9001,127.0.0.1:9002,192.168.1.50:9003,192.168.1.50:9004 `
  -RunDir python\runs\traitor_remote -Lr 1e-4 -EntCoef 0.005 `
  -EvalEvery 25 -EvalEpisodes 10
```

`Endpoints` 保留各项 host 与 port，旧 `Ports` 只表示本机端口。上述四环境各收集 512 步，每次更新 2048 步，固定评估各 10 局、合计 40 局。显式辐辉可将 [README](README.md) 完整命令的 `--ports` 换成 `--endpoints` 和上述四个端点。

## 检查与更新

状态从 A 的日志读取，使用 `watch_training.py --run-dir <目录>`。需要独立连通性检查时，先停止所有控制目标游戏的程序：

```powershell
.\python\.venv\Scripts\python.exe python\check_endpoints.py 127.0.0.1:9001 192.168.1.50:9003
```

检查器发送 `pos`，但它仍会新建并关闭连接；Mod 的连接清理可能释放注入输入，不能在采样期间当作无干扰探测。

Python 改动只更新 A；Mod 改动需关闭各游戏后更新 DLL 并重启。相同完整运行目录重启会恢复 checkpoint，不要同时启动两个训练器。

远程失联检查 B 是否开机、游戏是否加载、监听地址、IP 和防火墙。“难度设置无响应”本身不能证明 Mod 不支持难度。确认恢复后再续训，快速重置见 [FAST_RESET.md](FAST_RESET.md)。
