# HollowRL：叛徒领主 PPO

通过 BepInEx Mod 读取《空洞骑士》的角色、Boss 和弹幕状态，在游戏内注入按键，用 PPO 训练叛徒领主策略。控制频率 10 Hz，四帧堆叠的结构化观测，不依赖屏幕识别。支持固定权重试打、本机多窗口和跨电脑共享一个中央模型。

## 环境与安装

需要 Windows PowerShell、正版 Hollow Knight、Python 3.11+、.NET SDK 和 BepInEx 5 x64。Python 监控面板需要 Tk。

1. 关闭游戏，把仓库 `BepInEx/` 的内容复制到游戏根目录，使 `winhttp.dll`、`doorstop_config.ini` 与 `hollow_knight.exe` 同级。运行游戏一次，再关闭。
2. 构建 Mod，把示例路径替换成实际安装目录：

   ```powershell
   dotnet build unity_mod/HollowRL.slnx -c Release '-p:HKDir=D:\Games\Hollow Knight'
   ```

   将 `unity_mod/HollowRL/bin/Release/HollowRL.dll` 复制到每份游戏的 `BepInEx/plugins/HollowRL/HollowRL.dll`。
3. 备份 `%USERPROFILE%\AppData\LocalLow\Team Cherry\Hollow Knight`，再复制 `HK_save/user1.dat`。它会替换 1 号存档；同一 Windows 账户的多个游戏实例共享存档目录，训练时关闭 Steam Cloud。
4. 在仓库根目录建立 Python 环境：

   ```powershell
   python -m venv python\.venv
   .\python\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

游戏程序集从已安装的游戏中引用，仓库不包含游戏本体。

## 用已训练策略打辐辉并录屏

`models/traitor_lord_radiant.pth` 是历史更新 1000 的七按键模型，包含匹配的观测归一化。**核查发现，历史存档将基础骨钉伤害设为 33，且未装备力量护符；此前的 4/40 和无伤验收只能代表这个修改后的角色配置，不能作为标准 21 伤害成绩。**该模型保留用于迁移和复现历史结果；当前 `HK_save/user1.dat` 已恢复满级骨钉 21、无战斗护符，旧模型在新配置下的表现需重新评估。

先启动 9001 游戏：

```powershell
.\launch_multi_games.ps1 -GameDirs 'D:\Games\Hollow Knight' -Ports 9001 -Fps 60 -Width 1280 -Height 720
```

加载 1 号训练存档，站到叛徒领主雕像前，开启自己的录屏软件，再执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\record_radiant.ps1
```

按终端提示继续。入口只控制本机指定端口，使用固定贪心策略，不更新模型与归一化参数；校验原生辐辉难度 2、初始 Boss 血量 1300，以 1 倍速试打。每局上限 120 秒，观测时间特征仍使用来源模型的 60 秒尺度。失败自动重开，默认最多 100 局，无伤通关一次后停止重开。录屏软件需手动停止；终端 `Ctrl+C` 可中止试打。

可用 `.\record_radiant.ps1 -Port 9001 -Episodes 20` 改端口与局数。试打结果写入 `python/runs/recordings/`。继续训练应使用完整 PPO checkpoint，以保留优化器状态；发布的推理模型用于固定策略运行。

## 训练与监控

每个窗口加载训练存档并位于雕像前，只启动一个中央训练器。显式辐辉示例：

```powershell
.\python\.venv\Scripts\python.exe python\main_multi.py `
  --ports 9001 9002 --run-dir python\runs\traitor_radiant `
  --boss-level 2 --reward-profile radiant_spacing_v1 --eval-objective nohit `
  --episode-time-limit 120 --observation-time-limit 60 `
  --lr 0.0001 --ent-coef 0.005 --eval-every 25 --eval-episodes 10
```

已有运行目录时优先恢复其 `ppo_checkpoint.pth`；用完整旧模型开启新目录时增加 `--init-checkpoint <路径>`。需要保留迁移模型的归一化时增加 `--freeze-obs-norm`。每次更新保存最新状态，周期固定评估单独保存最优 checkpoint 与成绩。`--eval-episodes` 是每环境局数，四环境各 10 局合计 40 局。

```powershell
.\python\.venv\Scripts\python.exe python\watch_training.py --run-dir python\runs\traitor_radiant
.\python\.venv\Scripts\python.exe python\summarize_run.py python\runs\traitor_radiant
```

监控只读文件，不接管游戏。训练回合使用随机动作；判断水平应结合固定权重评估的胜率、无伤局数和 Boss 最低血量，损失下降不保证胜率提高。

## 奖励与策略

七按键动作覆盖左右、上下、攻击、跳跃和冲刺，相反方向互斥。当前 `radiant_spacing_v1` 主要原始奖励：Boss 每损失 33 血 +1、击败 +100、每损失一格血 -3、死亡 -20、空挥 -0.03、无伤结束危险动作阶段 +0.05、每秒时间成本 -0.10；中立/恢复阶段距离过远有有界成本。冲击波 ACTIVE 初始水平边缘距离为 3～6 且阶段无伤结束时，每次动作额外 +0.20。辐辉致命受伤通常损失 9 格血，受伤与死亡合计 -47。PPO 内部使用原始奖励的 0.05 倍。

## 目录与检查

- `python/`：PPO、环境、评估、试打、监控及真人行为克隆。
- `unity_mod/HollowRL/`：状态协议、输入注入、快速重置和真人录制 Mod。
- `models/`：发布用推理权重。
- `HK_save/`、`BepInEx/`：训练存档模板与 Mod 加载器。
- `tools/`：保留的维护工具。
- `python/runs/`：本地模型、日志、示范与试打结果，不提交 Git。

更多操作见 [快速重置](FAST_RESET.md)、[本机多窗口](MULTI_ENV_WINDOWS.md)、[跨电脑采样](MULTI_MACHINE.md)、[真人示范与行为克隆](HUMAN_DEMONSTRATIONS.md)。

离线检查不会操作游戏：

```powershell
.\python\.venv\Scripts\python.exe -m unittest discover -s python/tests -t python -p 'test_*.py'
```

实机试打另行启动 `record_radiant.ps1`。不要让训练器、真人录制器与试打入口同时控制同一端口。

## 21 伤害与三法术训练

新配置固定满级骨钉基础伤害 21、无战斗护符、已解锁黑波/黑吼/黑砸；保留零槽位虚空之心，不加入剑技。每台机器都要加载这一配置，不能混用旧 33 伤害存档采样。需要保留各机器自身存档时，先备份、关闭当地游戏，再生成独立副本：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\configure_training_loadout.ps1 -SavePath 'C:\path\user1.dat' -OutputPath 'C:\path\user1_21.dat'
```

工具只修正骨钉伤害和护符装备字段，不授予法术或移动能力；复制新副本到对应存档位置后重新加载。原存档和历史权重保留。

用历史七按键 Actor 初始化独立的新法术模型：

```powershell
.\python\.venv\Scripts\python.exe python\migrate_spell_policy.py models\traitor_lord_radiant.pth python\runs\spell_initial\initial_policy.pth
powershell -ExecutionPolicy Bypass -File .\train_spells.ps1
```

默认只连接本机 9001；多机可通过 `-Endpoints` 显式传入端点。该入口固定原生辐辉、10 Hz/1 倍速、120 秒局限、60 秒观测尺度、LR 0.0001、熵系数 0.005、固定旧观测归一化。迁移保留旧 Actor，重置 Critic/优化器，新法术头初始不施法 91%、每种法术各 3%；不是从零学习移动躲避。初始化文件拒绝覆盖；继续同一运行则从其完整 checkpoint 恢复。

策略保留原七个按键，并增加一个分类选择：不施法/黑波/黑吼/黑砸。每个非零决策发一个正常快速施法请求，并设置相应上下方向；不补魂、不绕过冷却、不自动回血。当前法术模式限定不超过 10 Hz 且 1 倍速，让短按之间有释放间隔。

新奖励 `radiant_spells_v1`：实际 Boss 掉血/21，击杀 +100，受伤每格 -3，死亡 -20，时间每秒 -0.10。不按施法、耗魂、近身或 FSM 阶段结束发正奖励。停用旧冲击波间距/穿波代理奖励及空挥惩罚。只在中立/恢复且附近没有残留脱离 Boss 的危险碰撞体时，距离超过 5 开始线性小额惩罚，到 12 封顶每秒 -0.15；危险物缺失/不可信时保守豁免。阈值为待验证假设，不保证比旧配置更强。

每局记录三法术请求次数及观察到的净魂量消耗；请求可能因资源/动画被游戏拒绝，不能当作成功施法数。选择模型仍看固定权重无伤通关和剩余 Boss HP，不能只看新旧奖励曲线。

```powershell
.\python\.venv\Scripts\python.exe python\watch_training.py --run-dir python\runs\traitor_radiant_spells
.\python\.venv\Scripts\python.exe python\play.py --checkpoint python\runs\traitor_radiant_spells\best_eval_checkpoint.pth
```

三法术实机验收从零魂开始，通过正常骨钉命中攒魂，分别确认三法术消耗 33 魂；黑波也观察到独立的 30 点伤害。这是功能验收，不代表模型已经学会何时施法。

## 对照结束后的本机自动续训

可选新奖励试验 `radiant_spells_dodge_v1` 在 `radiant_spells_v1` 基础上，增加通用动作无伤结束奖励和极微弱空挥惩罚：每次无伤完成动作 +0.1，没有每局次数上限，同一招只结算一次；游戏确认挥出但未命中的攻击每次 -0.005，冷却拒绝的按键请求不算空挥。它使用 Boss FSM 动作状态，合并跳起与俯冲为同一招；整招掉血或伤害计数增加、未知状态、死亡、超时及截断都不发躲避奖励。此近似不保证残留冲击波或飞镰已经消失，也不要求使用特定闪避方式。冲击波站位和真正穿波两项专项奖励均为 0。21 点实际 Boss 伤害仍为 +1、击杀 +100，其余奖励、学习率和网络保持一致。

为自动续训入口增加 `--dodge-trial` 时，先完成原熵系数对照，再用其选出的同一完整权重分别做原奖励/新奖励各 20 次更新、各 40 局固定评估；候选通过筛选后另做独立复测，确认改善才用于后续长训，否则保留原奖励和来源权重。所有分支、权重备份、参数与评估分别保存，不在运行中的旧分支上热改奖励。

`python/auto_continue_training.py --experiment-dir <对照目录>` 每 30 秒检查一次已创建的有限实验。只有实验完成并释放四窗后，才从完整备份权重另开训练目录：低熵候选必须通过筛选和独立复测，否则沿用已核实的源模型和原熵系数。奖励、角色数值、学习率保持不变；最新权重定期保存，已评估最优额外按版本归档。训练进程异常退出最多重试两次，游戏本身无法恢复时记录错误，不无限重启。

这是本机规则脚本，不会定时唤醒聊天模型，也不会自动创造新奖励函数。两台电脑、网络和四个游戏窗口需要持续运行且不休眠。查看 `<对照目录>/auto_continue_status.json` 和 `continuation_decision.json` 可追溯当前状态与选择依据。若要接回人工控制，先在对照目录创建 `STOP_AUTO_CONTINUE` 文件禁用监督，再停止相应训练器；停止标记本身不会强杀训练。
