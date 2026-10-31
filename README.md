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

`models/traitor_lord_radiant.pth` 是从更新 1000 的最优模型导出的推理权重，包含匹配的观测归一化参数。该模型曾在单批 40 局固定评估中无伤通关 4 局；这是一次评估结果，不能当作稳定 10% 胜率。

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

????????2026-10-31??130 ? Python ?????Mod ?????? 9001 ???????? 3 ??? 3 ????? Boss????????????????????????? `models/traitor_lord_radiant.json`?
