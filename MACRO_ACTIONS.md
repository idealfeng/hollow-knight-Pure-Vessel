# 35 动作离散分支

该分支与原组合动作并存，策略名为 `discrete_macros_v1`。每次从一个 35 分类头选择一项，再映射到普通游戏按键；每项只执行一次控制决策，不是自动连招。

默认录屏仍用原组合动作 650。`models/traitor_lord_discrete_16f_825.pth` 保留为实验对照，本轮未通关。

## 动作表

“当前朝向”表示不指定左右移动；“向左/向右”同时请求相应横向按键。上/下劈通过方向和攻击组合产生。

| 编号 | 动作 |
|---:|---|
| 0 | 不操作，释放持续按键 |
| 1、2 | 左移、右移 |
| 3、4、5 | 当前朝向跳跃、左跳、右跳 |
| 6、7、8 | 当前朝向：水平攻击、上劈、下劈 |
| 9、10、11 | 左移并水平攻击、上劈、下劈 |
| 12、13、14 | 右移并水平攻击、上劈、下劈 |
| 15、16、17 | 当前朝向跳跃并水平攻击、上劈、下劈 |
| 18、19、20 | 左跳并水平攻击、上劈、下劈 |
| 21、22、23 | 右跳并水平攻击、上劈、下劈 |
| 24、25、26 | 当前朝向冲刺、左冲刺、右冲刺 |
| 27、28、29 | 当前朝向跳冲、左跳冲、右跳冲 |
| 30、31、32 | 当前朝向黑波、向左黑波、向右黑波 |
| 33 | 黑吼 |
| 34 | 黑砸 |

该表限制了攻击+冲刺、攻击+法术、冲刺+法术等组合，不保证保留旧策略所有有效行为。普通冲刺/黑冲由原生能力和冷却决定，施法取决于魂量与动画。请求动作不等于原生执行成功。

## 初始化和续训

每帧保留旧的 118 项，16 帧共 1888 维，隐层仍为 `256 → 256`。四帧迁移时最近四帧对应旧权重，新历史列初始为 0，复制归一化。分类头通过旧策略在 35 个规范向量上的联合概率条件化构造。

它继承部分旧行为，既不是随机从零开始，也未完整保留旧组合分布。初次迁移重置 Adam；从完整离散检查点继续训练则保留模型、Adam 和归一化。奖励仍为 `radiant_spells_dodge_v1`，不含潜势。

只检查发布模型能否加载：

```powershell
.\python\.venv\Scripts\python.exe python\play.py --checkpoint models\traitor_lord_discrete_16f_825.pth --check-only
```

固定试打两局辐辉：

```powershell
.\python\.venv\Scripts\python.exe python\play.py --checkpoint models\traitor_lord_discrete_16f_825.pth --port 9001 --boss-level 2 --episodes 2 --stop-after episodes
```

继续研究时显式从完整离散检查点另开目录，以下会启动训练：

```powershell
.\python\.venv\Scripts\python.exe python\main_multi.py `
  --init-checkpoint models\traitor_lord_discrete_16f_825.pth `
  --run-dir python\runs\discrete_continue --ports 9001 `
  --action-distribution discrete_macros_v1 --frame-stack 16 `
  --observation-schema legacy_v1 --execution-profile legacy_v1 `
  --reward-profile radiant_spells_dodge_v1 --boss-level 2 `
  --control-hz 10 --game-speed 1 --episode-time-limit 150 --observation-time-limit 60 `
  --lr 0.0001 --ent-coef 0.005 --freeze-obs-norm --steps-per-env 512 `
  --eval-objective nohit --eval-every 25 --eval-episodes 10 --no-monitor
```

保留的 `python/migrate_macro_policy.py` 只从兼容的完整四帧、21 伤害三法术检查点生成独立初始化文件，拒绝覆盖已存在目标；发布的默认 16 帧模型不能直接作为四帧迁移输入。日常回放无需迁移。

## 本轮结果

离散版继承旧编号 550，最终到 845，实际新增 295 次 PPO 更新、529,920 条已用于更新的采样。完整训练日志 1,335 局、固定评估已结束 374 局，均未通关。最后四窗分支最佳评估为 825，40 局平均 Boss 剩 525 HP。

这次预算内没有出现预期的无伤突破，不能宣称更换动作空间就立即有效，也不能否定所有离散设计。原组合 650 的成功不能算成离散成绩。完整记录见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。
