# Windows 多环境训练

当前实现是真正的中央 PPO：每个游戏实例由一个独立 Python worker 控制，
所有 worker 把样本交给同一个模型更新。不要手动启动多个 `main.py`，那只会产生
互不共享数据的模型，并争用同一个 checkpoint。

```text
Hollow Knight :9001 -> worker 0 --+
                                  +-> central PPO -> one checkpoint
Hollow Knight :9002 -> worker 1 --+
```

## 1. 准备两个游戏目录

先完整退出 Steam 和游戏，备份：

```text
%USERPROFILE%\AppData\LocalLow\Team Cherry\Hollow Knight
```

将已经安装 BepInEx 和 HollowRL Mod 的游戏目录复制为两个训练目录，例如：

```text
D:\HollowRL-instances\env1
D:\HollowRL-instances\env2
```

每个目录都必须存在：

```text
hollow_knight.exe
BepInEx\plugins\HollowRL\HollowRL.dll
```

启动脚本会在每个游戏目录放入 `steam_appid.txt`（内容为 Hollow Knight 的
Steam App ID `367520`），用于从复制目录直接启动。没有这个文件时，副本可能
在日志中报告 `Application was not launched through Steam! Shutting down...`，
随后退出或被 Steam 重启到原安装目录。脚本现在会检查游戏进程仍在运行，且指定
端口确实由该进程监听，然后才输出 `Ready`。

独立目录隔离 BepInEx 日志和配置，但 Hollow Knight 的存档目录仍由同一 Windows
账户共享。因此必须使用专门的训练存档、事先备份，并关闭 Steam Cloud。不要用
正式游玩存档做多实例实验。

本机已经准备为：

```text
环境 1: D:\Program Files\Steam\steamapps\common\Hollow Knight
环境 2: D:\HollowRL-instances\env2
存档备份: D:\Paper\hollow knight\save_backup_before_multienv_20261026
```

编译后的最新 Mod 位于：

```text
unity_mod\HollowRL\bin\Release\HollowRL.dll
```

复制它覆盖到两个实例各自的 `BepInEx\plugins\HollowRL\HollowRL.dll`。

## 2. 启动游戏实例

仓库根目录执行：

```powershell
.\launch_multi_games.ps1 `
  -GameDirs "D:\HollowRL-instances\env1","D:\HollowRL-instances\env2" `
  -Ports 9001,9002
```

这台机器也可以直接使用已经填好路径的快捷脚本：

```powershell
.\launch_two_local_games.ps1
```

脚本默认给每个实例设置：

- 独立 TCP 端口；
- 640x360 窗口；
- 30 FPS 上限；
- 2 倍游戏速度；
- 后台继续运行（由 Mod 设置 `Application.runInBackground`）。

新启动的游戏窗口会正常显示，方便分别加载训练存档；重复运行脚本会识别已
监听对应端口的实例，不会再额外启动同一路径的游戏。

两个窗口都需要加载训练存档并站到 Pure Vessel 雕像前。训练器启动时会对两边
同时执行 `up`、`submit`，检查 Boss 确实出现，再暂时冻结游戏时间以完成
模型初始化；开始采样时恢复指定的训练速度。此自动进入仅适用于随仓库提供的
Pure Vessel 训练存档和已知雕像坐标。若位置不符，训练器会明确报出端口和坐标。

## 3. 启动中央训练器

```powershell
.\run_multi_windows.ps1 -Ports 9001,9002
```

如果监控窗口因本机 Tcl/Tk 配置无法打开，训练器会自动改用终端日志继续运行。也可以明确禁用图形监控，并限制本次更新次数：

```powershell
.\run_multi_windows.ps1 -Ports 9001,9002 -NoMonitor -TotalUpdates 100
```

每次 PPO 更新后会原子保存检查点；再次使用同一 `-RunDir` 会从该检查点继续。

默认两个环境各收集 1024 步，每次 PPO 更新仍是 2048 个 transition。checkpoint
和曲线写入独立目录：

```text
python\runs\body_center_v3_9001_9002\
```

旧实验 `python\runs\multi_9001_9002\` 保留，仅供对照：当时每局约 13 游戏秒的 Boss
开场动画被算作训练步数，而且碰撞盒常缺失。新实验会等到 Boss 的 Control FSM 离开
`Intro` 且真实身体碰撞盒出现后才开始采样；Mod 不再把技能探测范围的 trigger 当作身体框。
传送期间身体框可能暂时缺失，此时距离奖励跳过该帧。新目录也不会覆盖单环境的
`python\ppo_checkpoint.pth`。

`python\runs\combat_ready_bodybox_9001_9002\` 也保留：它已经跳过开场并读到身体框，
但策略的相对 Boss 坐标仍来自根节点。v2 改为使用身体框中心，传送期间才回退到根节点；
观测含义改变，因此单独开始新 checkpoint。

三个环境可用：

```powershell
.\run_multi_windows.ps1 -Ports 9001,9002,9003 -StepsPerEnv 768
```

当前这台 16 GB 机器双开后实测仅剩约 1.5 GB 可用内存，两份游戏的工作集共约
2.4 GB；显存使用约 2.7/8.2 GB。因此当前建议最多 **2 个游戏实例**，不要在
这个内存占用水平下再开第 3 个。只有先关闭其他高内存应用、留出至少约 4 GB
可用内存，再逐步测试 3 个实例的稳定性与总吞吐。

## 4. 如何判断是否值得多开

更新日志会输出：

```text
step/s | reset 31.2s/2 retries
```

先分别记录单环境和双环境至少 30 分钟：

- 双环境总 `step/s` 应达到单环境的约 1.7 倍以上；
- 总内存建议持续低于 13--14 GB；
- 不应持续使用页面文件；
- 每个窗口应保持至少 30 FPS；
- reset retries 不应持续增加；
- 两个端口都应不断产生 episode。

若重置重试很多，优先修正菜单时序，不要继续增加环境数。异步 worker 会让另一个
环境在某个实例重置期间继续采样，但无法消除重置本身浪费的机器时间。

确认机器上死亡动画更早结束后，可以小步测试重置等待时间，例如：

```powershell
.\run_multi_windows.ps1 -Ports 9001,9002 -DeathSettleWait 6 -PostLeftWait 1.5
```

一次只减少 0.5 秒，并观察 `reset retries`；重试增加就恢复默认值。默认值仍为
7 秒和 2 秒，不会因为启用多环境而冒险加快菜单输入。

## 实现约束

当前 checkpoint 格式为 5：在身体框中心坐标修复的基础上，修正了“Boss 是否朝向小骑士”
的左右判断。格式 4 的示范可由 `python/train_bc.py` 按原始位置和朝向迁移；旧网络不能直接续接。
模仿学习与本机对照实验的结果、路径和续训命令见 [TRAINING_DIAGNOSIS.md](TRAINING_DIAGNOSIS.md)。

- 每个 worker 在导入 `game_state` 前设置自己的 `HOLLOWRL_PORT`。
- rollout 期间模型参数固定，收集结束后中央进程才更新。
- GAE 沿每个环境自己的时间轴计算，之后才合并 batch。
- observation normalization、optimizer、checkpoint 只存在于中央进程。
- 某个环境死亡后可独立重置，不会阻塞另一个环境继续 step。
