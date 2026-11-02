# 原生难度与快速场景重置

快速重置第一次从雕像进入 Boss 场景并捕获合法训练资源，之后直接重载目标场景，等待骑士、Boss、摄像机和画面进入可战斗状态。它减少神居往返等待，不把改血量或跳过伤害判定当作加速方式。

## 难度与日常操作

| `--boss-level` | 原生难度 | 叛徒领主初始 HP |
|---:|---|---:|
| 0 | 调谐 Attuned | 800 |
| 1 | 升天 Ascended | 1300 |
| 2 | 辐辉 Radiant | 1300 |

程序同时核对原生目标/实际等级、Boss HP、骑士 9 格生命和初始魂量；1300 HP 本身不能区分升天和辐辉。使用骨钉 21、无战斗护符、三法术升级版的训练存档。

按 [README](README.md) 启动已安装 Mod 的窗口，加载 1 号存档并站到雕像前，不必手动选难度。不要同时运行控制相同端口的另一任务。

固定打完两局辐辉：

```powershell
.\python\.venv\Scripts\python.exe python\play.py --port 9001 --boss-level 2 --episodes 2 --stop-after episodes
```

固定打完两局调谐：

```powershell
.\python\.venv\Scripts\python.exe python\play.py --port 9001 --boss-level 0 --episodes 2 --stop-after episodes
```

录到一局无伤后停止用 `.\record_radiant.ps1 -Port 9001 -Episodes 0`。固定局数评估显式使用 `--stop-after episodes`，避免中途胜利就退出而误报完整批次。

## 协议与所有权

内部命令包括 `RESET_FIGHT ENABLE`、`RESET_FIGHT LEVEL <0|1|2>`、`RESET_FIGHT TARGET`、`RESET_FIGHT`、`RESET_FIGHT STATUS`、`RESET_FIGHT VISUAL`、`RESET_FIGHT DISABLE`。唯一控制连接获取重置所有权，Python 按 generation 核对本次结果，只有协议和状态流都就绪才继续。

重置恢复捕获的资源，关闭调试锁血，保留原生黑冲、法术和受伤规则。加载其他存档或更新 Mod 后应重新启动并捕获；不能假定磁盘文件更新已改变运行中实例。

## 故障和时序边界

场景/视觉未就绪、断线或原生等级不匹配会报错，不把重置失败算成有效战斗，也不静默降难度。查看相应任务的日志，确认原控制器退出后再启动任务。

不要在采样中另开短连接发送 `pos`：关闭连接可能触发输入清理。运行中优先读日志或只读面板；端点检查工具只在没有其他控制器接管时使用。

游戏 FPS 与名义 10 Hz 控制不同。当前没有引擎帧序号或同步推进协议，恢复速度稳定后首步仍可能使用保留的暂停观测。名义 150 秒预算按决策次数计算，实际墙钟可能更长。这些时序限制尚未修复，详见 [实验纪实](docs/EXPERIMENTS.md)。
