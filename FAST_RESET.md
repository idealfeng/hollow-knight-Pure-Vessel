# 叛徒领主直接重开战斗

中央 Python 训练器会探测新的 `RESET_FIGHT` 接口，支持时自动启用。第一次仍需加载训练
存档并从雕像进入战斗一次，以记录该实例的场景、入口、难度、血量和魂量。此后的训练死亡
会停留在 Boss 场景，Python 记录终局后直接请求重载同一场景，跳过返回神居和雕像操作。

## 已验证

9001、9002 均已通过自然死亡后重开与连续活着重开，每轮约 4.69–4.73 秒。骑士恢复
9 血、0 魂，Boss 恢复 800 血，Python 每局命中/受伤计数清零。修复后记录在
`python/runs/fast_reset_visual_*.json`，游戏窗口截图在 `artifacts/game_*_verified.png`。

早期版本只验证了游戏状态，直接重载后镜头仍停在 `FadingOut`，导致黑屏。当前版本
显式调用游戏原生 `FadeSceneIn()`，等待画面恢复后才报告就绪；两份游戏均已检查重载后
的实际战斗画面。独立验证器还会检查主镜头启用且淡入状态为 `Normal`。

9001 此前额外通过控制连接意外断开测试：死亡拦截自动释放，角色按原游戏流程返回
雕像，未留在 0 血冻结状态。Python 回归检查共 44 项通过。

场景原有的 Boss 出场仍保留，避免更改开局条件；节省的是死亡返回、神居菜单和雕像交互
等待。场景载入速度取决于每台电脑，4.7 秒是本机测量值。

## 同学电脑上的 9003、9004

### 2026-10-29 受伤与锁血修复

快速重载停止 HeroController 协程后，现在会清空旧协程句柄与受伤计时器，避免下一局
无法重新开启受伤无敌帧。快速重置训练模式也会清除并禁用 F8 调试锁血，防止角色受伤
后自动回到 9 血。独立验收遇到无操作回血会直接报错；60 秒超时的强制死亡会明确记作
超时，不再误报成 Boss 秒杀。

本次 9001、9002 各通过三轮实机验收：首轮受伤后立即重置，后两轮自然死亡并重置。
未出现无操作回血或瞬间连续扣血，相邻受伤最短间隔约 1.30 秒；六次直接重置耗时
4.68–4.74 秒，均恢复 9 血、0 魂、Boss 800 血，镜头与淡入状态正常。
结果保存在 `python/runs/fast_reset_iframes_9001.json` 和
`python/runs/fast_reset_iframes_9002.json`。本次 DLL 编译成功，44 项 Python 回归测试通过。
9003、9004 已安装修复包，并各通过相同的三轮实机验收：受伤后立即重置、两轮自然
死亡、资源恢复与镜头检查均通过，直接重置耗时 4.67–4.72 秒，未出现无操作回血或
瞬间连续扣血。结果保存在 `python/runs/fast_reset_iframes_9003.json` 和
`python/runs/fast_reset_iframes_9004.json`。四窗已冻结待训，中央训练器保持停止。

本机此前安装的无敌帧修复 DLL 和旧 `fast_reset_mod.zip` 并非同一版本。请使用本次
重新生成的安装包，四个窗口均需重启才能加载新 DLL。原有黑屏与自然死亡验收记录
不能替代此次受伤修复的实机验收。验收命令如下，每个端点分别执行：

```powershell
.\python\.venv\Scripts\python.exe -u python\verify_fast_reset.py `
  --endpoint 127.0.0.1:9001 --cycles 3 --natural-death-each-cycle `
  --reset-after-first-hit --output python\runs\fast_reset_iframes_9001.json
```

当前 `traitor_4env/ppo_checkpoint.pth` 为 30,720 步，包含异常受伤期间的训练。
异常前备份 `ppo_before_fast_reset_20261028_235318.pth` 为 14,336 步。修复验收通过后，
建议先在修复后的环境里评估该备份，再决定恢复训练；本次实机验收没有评估模型强弱。
备份保留了真人示范预训练基础。日志中异常前最后 40 局平均命中 6.3 次，22,528 步
之后的 272 局平均命中约 0.56 次；这些是训练阶段的统计，不是相同条件的模型对比评估。
确认恢复后，可用下面命令在新目录接续备份，保留原始运行与日志。已有目录的 `ppo_checkpoint.pth`
优先于 `-InitCheckpoint`，因此恢复时必须使用新的运行目录：

```powershell
.\run_multi_windows.ps1 `
  -Endpoints 127.0.0.1:9001,127.0.0.1:9002,192.168.0.184:9003,192.168.0.184:9004 `
  -InitCheckpoint python\runs\traitor_4env\ppo_before_fast_reset_20261028_235318.pth `
  -RunDir python\runs\traitor_4env_iframes_fixed `
  -Lr 1e-4 -EntCoef 0.005 -EvalEvery 10 -EvalEpisodes 1
```

### 本次恢复训练

14,336 步备份已在修复后的四窗各完成两局贪心评估，共 8 局，平均命中 11 次、Boss
最低血量平均 437、胜率 0；最好一局命中 18 次、Boss 降至 206 血。评估期间没有
断连或战斗场景丢失，未更新权重。完整结果为
`python/runs/traitor_4env_iframes_fixed/baseline_eval.json`，记录了源备份 SHA256。
这是一小批恢复前基准，不能直接与随机采样训练的平均命中数比较。

中央训练器已按用户确认从该备份的 update 7、14,336 步启动，运行目录为
`python/runs/traitor_4env_iframes_fixed`。每环境 512 步、每轮合计 2048 步，学习率
`1e-4`、熵系数 `0.005`，每 10 次更新进行四窗贪心评估，沿用备份中固定的观测归一化。
`trainer_stdout.log`、`trainer_stderr.log` 和 `launch.json` 保存在新运行目录中。
原始 `traitor_4env` 目录与备份保留。

该轮停止后续训时仍使用上方命令中的运行目录；已有的新 checkpoint 会优先载入。
`artifacts/start_recovery_training.ps1` 仅供首次恢复，已有新 checkpoint 时会拒绝再次初始化。

把本项目生成的 `artifacts/fast_reset_mod.zip` 复制到同学电脑并解压。停止中央训练器，
关闭同学电脑上的两份游戏。在解压目录执行（替换为两份游戏的实际目录）：

```powershell
.\install_reset_mod.ps1 -GameDirs 'D:\Games\HollowKnight3','D:\Games\HollowKnight4'
```

安装器会先备份旧 DLL，并校验新 DLL。它不需要同学电脑安装 .NET SDK，也不会改存档。
然后用原来的启动命令重新打开 9003、9004，保留原 `--hollowrl-bind-ip`、端口和 Boss 参数。
两份游戏各加载一次训练存档，放在叛徒领主雕像前即可。中央 Python 文件只需在训练电脑更新。

四个端点验收通过后，使用上方新运行目录与异常前备份接续训练。
日志中 `reset_mode=scene` 表示直接重载；`menu` 表示该实例仍走旧雕像流程。新旧 DLL 可以
混合运行，但只有新版能加速。

## 独立实机验证

先停止 PPO 训练器，目标游戏加载训练存档并位于雕像前：

```powershell
.\python\.venv\Scripts\python.exe python\verify_fast_reset.py `
  --endpoint 192.168.0.184:9003 --cycles 2 --natural-death `
  --output python\runs\fast_reset_validation_9003.json
```

此命令会进入战斗，第一轮让 Boss 击败无操作角色，再检查重载资源和回合计数。成功后
游戏冻结待训；不读取模型或进行训练。不要同时对同一窗口启动验证器和 PPO。

## 协议与回退

- `RESET_FIGHT ENABLE`：当前控制连接取得快速重置所有权，才会拦截训练死亡。
- `RESET_FIGHT`：返回已排队的重置编号，重载在 Unity 主线程执行。
- `RESET_FIGHT STATUS`：返回 `reset_v1;状态;编号;场景;说明`。
- `RESET_FIGHT VISUAL`：请求 Unity 主线程采样镜头状态，稍后再次查询取得结果。
- `RESET_FIGHT DISABLE`：恢复游戏原有死亡流程。控制连接断开也会释放所有权。

Python 同时确认新编号已 `ready`、状态流已显示可战斗后才开始下一局；旧 DLL 不支持时
自动使用雕像流程。新版若报错或超时，会关闭快速死亡拦截后尝试旧流程，并打印失败原因。
每轮重置成功时游戏先冻结，由 worker 的 `start` 命令恢复训练速度。

如需暂时关闭快速模式，在启动中央训练器之前设置：

```powershell
$env:HOLLOWRL_FAST_RESET = '0'
```

移除此环境变量或设为 `auto` 可恢复自动探测。切换难度、护符或初始资源后，重启对应
游戏以重新捕获基准。本轮接口不改变观测、动作或 checkpoint 格式。
