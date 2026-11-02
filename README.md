# HollowRL：叛徒领主 PPO

通过 BepInEx Mod 读取碰撞盒、速度、生命、魂量和 Boss FSM，用游戏内输入接口执行动作，再以 PPO 训练。支持单窗口、本机多窗口和两台电脑共同采样；推理和参数更新由一台中央训练主机完成。

项目已录成标准配置下的叛徒领主辐辉无伤视频，但**没有得到稳定高胜率的无伤策略**。默认发布的是在 A 主机两次录屏成功的原组合动作、16 帧、第 650 次权重。35 动作离散分支保留为对照，本轮没有通关。

- [实验经过、失败探索与证据边界](docs/EXPERIMENTS.md)
- [结构化实验结果](docs/experiment_results.json)
- [清理后的离线与实机验收](docs/VALIDATION.md)

## 安装

以下命令在项目根目录的 Windows PowerShell 执行。需要 Hollow Knight、Python 3.11/3.12、支持 `.slnx` 的 .NET SDK 和 BepInEx 5 x64；GUI 面板需要可用的 Tk。

1. 安装 BepInEx，使 `winhttp.dll`、`doorstop_config.ini` 与 `hollow_knight.exe` 同级，先运行一次确认加载正常，再关闭游戏。
2. 编译并安装 Mod，替换自己的游戏路径：

   ```powershell
   .\install_mod_windows.ps1 -GameDir 'D:\Games\Hollow Knight'
   ```

   也可只编译，将输出的 `unity_mod/HollowRL/bin/Release/HollowRL.dll` 放入游戏的 `BepInEx/plugins/HollowRL/`：

   ```powershell
   dotnet build unity_mod/HollowRL.slnx -c Release '-p:HKDir=D:\Games\Hollow Knight'
   ```

3. 游戏关闭时，先备份 `%USERPROFILE%\AppData\LocalLow\Team Cherry\Hollow Knight`，再按需将 `HK_save/user1.dat` 安装到 1 号槽位。同一 Windows 账户的游戏实例共享存档目录；使用训练存档时关闭 Steam Cloud，避免旧文件同步回来。
4. 创建 Python 环境：

   ```powershell
   python -m venv python\.venv
   .\python\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

游戏本体及其原生程序集需要自行安装；构建时引用实际游戏目录。

## 角色配置与难度

| 项目 | 标准配置 |
|---|---|
| 骨钉 | 满级，普通命中 **21 伤害** |
| 护符 | 无战斗护符；允许保留虚空之心，编号 36 |
| 资源 | 初始 9 格生命、0 蓝血、0 SOUL |
| 法术 | 黑波、黑吼、黑砸均为升级版，正常耗魂；不使用回血和剑技 |
| 调谐 | 原生 `boss-level 0`，Boss 初始 **800 HP** |
| 升天 | 原生 `boss-level 1`，Boss 初始 **1300 HP** |
| 辐辉 | 原生 `boss-level 2`，Boss 初始 **1300 HP**，一次有效受伤即死 |

1300 HP 本身不能区分升天与辐辉，程序还会校验原生难度等级和初始资源。早期曾使用实际单刀 33 伤害的旧配置，其成绩不混入标准 21 伤害结果；旧模型已备份，不再默认使用。

离线检查仓库训练模板：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\inspect_training_loadout.ps1 -RequireStandardLoadout
```

检查自己的磁盘存档时添加 `-SavePath 'D:\Saves\user1.dat'`。该工具只读磁盘，不连接游戏或写存档；通过检查不证明运行中的窗口已经重新加载。`tools/configure_training_loadout.ps1` 可通过 `-SavePath`、`-OutputPath` 生成独立的 21 伤害无战斗护符副本，拒绝覆盖源文件，也不授予尚未获得的法术。安装副本前关闭游戏，之后重新加载。

## 单窗口录屏

启动已安装 Mod 的游戏：

```powershell
.\launch_multi_games.ps1 -GameDirs 'D:\Games\Hollow Knight' -Ports 9001 -Fps 60 -GameSpeed 1 -Width 1280 -Height 720
```

加载 1 号训练存档，站到叛徒领主雕像前，不必手动选难度，然后执行：

```powershell
.\record_radiant.ps1 -Port 9001 -Episodes 0
```

默认使用 `models/traitor_lord_radiant.pth`，即原组合动作、16 帧、第 650 次权重。按提示启动录屏软件，再按 Enter 开始；贪心回放不更新模型与归一化。`-Episodes 0` 表示失败继续重试，首次有效无伤后停止；可用 `-Episodes 100` 限制最大尝试数，或 `Ctrl+C` 中止。退出时只释放输入并暂停所选端口。**录像软件仍需手动停止。**

每局与汇总写入 `python/runs/recordings/session_*/`。首次成功即停得到的录像不能替代固定局数胜率测试。

### 固定局数与难度切换

用相同的默认 650 权重完整打完 20 局调谐，不因中途胜利提前停止：

```powershell
.\python\.venv\Scripts\python.exe python\play.py --port 9001 --boss-level 0 --episodes 20 --stop-after episodes
```

完整打完 2 局辐辉，检查原生辐辉进战与回放流程：

```powershell
.\python\.venv\Scripts\python.exe python\play.py --port 9001 --boss-level 2 --episodes 2 --stop-after episodes
```

`--boss-level 0/1/2` 对应调谐/升天/辐辉。`--stop-after flawless` 首次无伤通关后停止，`win` 首次任意通关后停止，`episodes` 打满给定的正数局数。`--episodes 0` 不能与 `--stop-after episodes` 同用。少量辐辉进战检查不等于证明无伤胜率。

只检查模型能否加载，不连接游戏：

```powershell
.\python\.venv\Scripts\python.exe python\play.py --check-only
```

清理后的实际测试结果和固定局数单列于 [验收记录](docs/VALIDATION.md)，不把历史成功视频当作本次兼容性验收。

## 发布模型

| 文件 | 用途 | 帧堆叠 / 更新编号 | 大小 |
|---|---|---|---:|
| `models/traitor_lord_radiant.pth` | 默认推荐回放与续训，原组合动作 | 16 / 650 | 6.39 MiB |
| `models/traitor_lord_radiant_4f_550.pth` | 原组合动作历史对照，A 录屏不可靠 | 4 / 550 | 2.22 MiB |
| `models/traitor_lord_discrete_16f_825.pth` | 35 动作离散对照，本轮未通关 | 16 / 825 | 6.45 MiB |

三份均为**完整训练检查点**，包含模型、Adam 和观测归一化。发布时只移除个人路径、端点等元数据；模型、优化器与归一化逐项相同。重新序列化会改变容器 SHA，各模型旁的 JSON 同时记录发布 `sha256` 与原始 `source_sha256`，不应把文件哈希变化误认成神经网络权重变化。

650 的两次 A 录屏分别在第 23、31 局成功；550 历史的两局无伤均在 B，后来 A 单窗累计 409 局未成功。不同模型、机器和首次成功即停的流程不能混成稳定胜率，详见 [实验文档](docs/EXPERIMENTS.md)。

## 从 650 继续训练

先让指定窗口加载标准存档并站到雕像前。以下从完整 650 状态恢复 Adam 和归一化，使用 16 帧、三法术、原生辐辉与原奖励：

```powershell
.\train_spells.ps1 `
  -Endpoints 127.0.0.1:9001 `
  -InitCheckpoint models\traitor_lord_radiant.pth `
  -RunDir python\runs\traitor_continue `
  -FrameStack 16 -RewardProfile radiant_spells_dodge_v1 `
  -Lr 0.0001 -EntCoef 0.005 -FreezeObsNorm `
  -EvalEvery 25 -EvalEpisodes 10 -NoMonitor
```

新目录从第 651 次更新继续编号；这不等于完成了 650 次新训练。同目录已有 `ppo_checkpoint.pth` 时，入口恢复该目录最新完整状态及保存设置；另开实验使用新目录。`TotalUpdates` 是更新编号上限。项目不会自动恢复已停止的长训，只有执行命令才启动训练。

该预设每环境收集 512 步，一个环境每次更新 512 条，四环境 2048 条。每次固定评估每环境 10 局，双窗共 20，四窗共 40。训练随机采样与评估贪心选择的分布不同，不要直接混比平均剩血。

```powershell
.\python\.venv\Scripts\python.exe python\watch_training.py --run-dir python\runs\traitor_continue
```

面板只读日志，不连接游戏。`run_config.json` 是实际配置，`training_metrics.jsonl` 是训练/更新/评估摘要，`eval_metrics.jsonl` 是固定评估逐局记录；最近完整模型为 `ppo_checkpoint.pth`，评估选出的模型为 `best_eval_checkpoint.pth`。

多窗口和跨机器见 [MULTI_ENV_WINDOWS.md](MULTI_ENV_WINDOWS.md)、[MULTI_MACHINE.md](MULTI_MACHINE.md)。不要让两个控制器同时接管同一端口，也不要在采样中运行会新建并关闭游戏连接的检查。

## 奖励与网络

默认每帧 118 个状态特征，16 帧为 1888 维，隐层 `256 → 256`。组合策略的方向互斥，跳跃、冲刺、攻击与法术等可按规则组合；离散版另用一个 35 分类头，见 [MACRO_ACTIONS.md](MACRO_ACTIONS.md)。

`radiant_spells_dodge_v1` 的原始奖励如下，进入 PPO 前统一乘 `0.05`：

| 事件 | 奖励 |
|---|---:|
| Boss 实际掉血，包含法术 | 掉血量 ÷ 21 |
| 击败 Boss | +100 |
| 被跟踪招式动作结束，期间未受伤 | +0.1，无每局次数上限 |
| 确认骨钉空挥 | −0.005 |
| 时间成本 | −0.1 / 名义秒 |
| 安全接近阶段离 Boss 过远 | 最多 −0.15 / 名义秒，危险阶段及邻近危险物有豁免 |
| 损失生命 | −3 / 格 |
| 死亡 | −20 |

21 伤害一刀为 +1；法术按实际伤害计分，不因“请求施法”加分。辐辉死亡常见 9→0，损血和死亡合计约 −47。小额闪避项检查的是动作结束，不能证明飞出的冲击波已消失。默认未开启冲击波专项奖励、攻击距离潜势或 TTC 安全潜势。

其余参数为 `γ=0.9955`、GAE `λ=0.95`、学习率 `1e-4`、熵系数 `0.005`。奖励与网络属于 A 上的 Python 逻辑；改变游戏命令、观测导出或原生处理时才需要更新各游戏的 Mod。都应保留旧模型和实际参数。

## 失败探索与可复现性

本轮尝试过低熵、冲击波专项奖励、闪避/空挥、潜势、新黑冲字段、4→16 帧、冲刺重发、低魂保护和 35 动作离散化，没有形成稳定提高无伤率的路线。潜势和新增输入曾同时变化，不能单独归因；650 的成功也不能证明帧数越多越好。

离散分支从旧编号 550 到最终 845，新增 **295 次更新、529,920 条采样**；1,335 局已结束训练和 374 局已结束固定评估均无通关。最后四窗分支的最佳评估是 825，40 局平均 Boss 剩 525 HP。发布该模型用于保留对照，不替代成功的原版 650。

另已确认 A 启动请求 60 FPS、B 曾请求 30 FPS，同一权重的历史有效决策间隔约 104/110 ms。启动目标不是实测 FPS，游戏菜单仍可修改上限。通用启动器现在默认 60；运行实例帧率/速度参数不符时拒绝复用，不自动重启。B 的旧 30 帧进程须关闭后再以 60 启动；改脚本不会修改已运行进程。游戏 FPS 与 AI 名义 10 Hz 不同，命令往返、推理与输入处理开销还会叠加。

## 未来展望

后续可探索原子状态发布、帧序号和输入执行确认，修正恢复后约 150 ms 首步仍可能使用旧观测的问题；还可研究容错到无伤的课程、人类示范和严格的跨机器复现。这些是展望，**尚未实现或证明有效**，没有混入发布模型。详见 [实验文档](docs/EXPERIMENTS.md)、[快速重置](FAST_RESET.md) 和 [潜势实验](COMBAT_POTENTIAL.md)。

### 失败前回退与分阶段课程（未实现）

- **受伤后回到失败前的局面重练。** 将“回退约10秒”视为失败附近重启的训练课程，让模型多练冲击波和后段局面，减少重复打熟练前段的成本。固定10秒未必最佳，应比较不同回退长度并混入正常开局，最终评估始终从真正开局开始、禁止回退。这与 [Backtracking Restarts](https://journals.flvc.org/FLAIRS/article/view/128557) 和 [Go-Explore](https://arxiv.org/abs/2004.12919) 的重访状态思路有关，不代表在本游戏中已经验证加速收敛。
- **回退必须保持训练语义正确。** 保留本次真实受伤/死亡的失败样本；把恢复视为新片段，不能把时间跳回前后的状态当成普通连续转移，也不能重复灌入旧轨迹冒充当前策略采样。快照需恢复Boss血量与FSM、骑士位置/资源/冷却、危险物、计时器和必要的随机状态，并同步恢复观测历史、清理输入。回退的是游戏局面，不是网络和Adam已学参数；仅恢复坐标和血量不够。
- **优先补后段样本，再比较独立阶段模型。** 单一策略已经能读取BossHP和FSM，可以先用同一模型混合正常开局、低血量阶段和失败快照采样,分阶段采样不等于必须为每阶段另训一个网络。如要比较前后段专家网络，须验证阶段触发条件、切换边界、共享动作/输入以及两模型的衔接，最终仍测整场无伤率。
- **机制与瓶颈需要分别验证。** [叛徒领主机制资料](https://hollowknight.wiki/w/Traitor_Lord)描述冲击波在低血量时加入招式，并可能在先前攻击的冷却期间使用，也会后撤调整站位；没有据此确认所有旧招式在低血量时统一加速。不能把1300HP直接按半血650切分。现有轨迹支持冲击波相关局面是难点，但不足以证明它就是全部瓶颈；详见[实验补充](docs/EXPERIMENTS.md#8-后续设想失败前回退与阶段化训练)。

## 目录与检查

| 目录 / 入口 | 内容 |
|---|---|
| `unity_mod/` | 观测、输入、原生难度与场景重置 Mod |
| `python/main_multi.py`、`python/play.py` | 中央 PPO 与固定权重回放 |
| `python/agent/`、`python/tests/` | 动作、奖励及核心回归检查 |
| `models/` | 三份选定完整模型及来源 JSON |
| `HK_save/`、`tools/` | 训练存档模板与离线检查/生成工具 |
| `docs/` | 实验纪实、结果摘要与验收 |
| `python/runs/`、`artifacts/` | 本机历史、日志和备份，不作为正式入口发布 |

保留人类示范和行为克隆工具，见 [HUMAN_DEMONSTRATIONS.md](HUMAN_DEMONSTRATIONS.md)。一次性驱动和机器专用启动脚本已清理，必要的网络、动作、奖励、重置和迁移回归检查仍保留。

```powershell
.\python\.venv\Scripts\python.exe -m unittest discover -s python/tests -t python -p 'test_*.py'
```

离线检查验证接口与计算一致性，不能代替实际战斗能力评估。具体检查数量、清理后的调谐固定局数胜率和辐辉进战结果见 [docs/VALIDATION.md](docs/VALIDATION.md)。
