# 真人示范录制

录制叛徒领主真人操作，用来预训练当前的七按键策略。记录的是游戏内每一帧的逻辑按键和
内存状态，不依赖键位、不录屏；支持键盘和手柄。左右移动、跳跃、攻击等组合完整保留。

## 开始

在项目目录执行（已经安装新版 Mod 时跳过第一条）：

```powershell
.\prepare_human_recording.ps1
.\record_human_windows.ps1 -Port 9001 -Episodes 5
```

先正常停止 PPO 训练器；准备脚本在训练器运行时会拒绝重启游戏。第一条会备份旧 Mod、
关闭并重新打开主游戏（9001），以 1280×720、60 FPS、1 倍速运行，并指定叛徒领主目标。
先保存需要保留的游戏进度。第二条开始录制；看到 `Recording to:` 后，在主游戏加载 1 号
训练存档，手动进入叛徒领主战斗。开始时确认使用相同难度和装备，不要在局间换护符。

先打 1 局检查数据，再补到 5～10 局。不要求无伤，正常争取打好，死亡后手动重试即可。
第一批优先演示骨钉、移动、跳跃和冲刺；法术也会记录，但旧训练动作尚不能执行法术。
不要在录制时启动 PPO 或其他自动操作程序。

每局结束显示 `Fight ... saved`。菜单和开场动画保存到原始文件，但不标记为战斗。
达到指定局数后自动结束；在录制终端按 Ctrl+C 会结束并保留已录数据。
也可以在本次输出目录创建名为 `STOP` 的文件，让录制器在下一批数据写入后正常退出。

## 输出

默认写入 `python/runs/human_demos/human_9001_日期_时间/`：

- `frames.jsonl`：Unity 帧号、游戏时间、实际时间、按住/按下/松开的位掩码、完整状态协议。
- `episodes.jsonl`：每局命中、受伤、最低 Boss 血量、胜负、时长及按键使用次数。
- `session.json`：格式、按键顺序、录制状态、帧数和丢帧数。

如果使用旧版录制器留下的会话，先用 `reindex_human_demo.py` 重建
`episodes_verified.jsonl`；新录制会话直接使用 `episodes.jsonl`。

`session.json` 中 `dropped_frames` 应为 0。录制器断开时会停止采集；Mod 队列有上限，
发生丢帧会明确报错，保留文件供检查。真人录制期间 Mod 拒绝自动输入命令。

同一帧的状态来自 Unity Update，输入来自 LateUpdate。转换脚本将观测与随后执行的动作
配对，并把奖励对齐到该动作之后的状态变化。记录完成后，先检查 `session.json` 的
`dropped_frames` 为 0，再执行：

```powershell
.\python\.venv\Scripts\python.exe python\build_human_bc_dataset.py `
  --sessions python\runs\human_demos\<会话目录> `
  --output python\runs\traitor_human_demos.npz
.\python\.venv\Scripts\python.exe python\train_bc.py `
  --demos python\runs\traitor_human_demos.npz `
  --run-dir python\runs\traitor_bc_v1_9001_9002 --ports 9001 9002
```

行为克隆只使用已录下的完整战斗，并按整局划分训练与验证。预训练结束后，两个游戏实例
均准备好叛徒领主战斗，再运行：

```powershell
.\run_multi_windows.ps1 -Ports 9001,9002 -RunDir python\runs\traitor_bc_v1_9001_9002
```
