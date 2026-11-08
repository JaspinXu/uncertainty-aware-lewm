# CS5478 期末展示讲稿：Improving Latent World Model Planning for Vision-Based Robotic Grasping

Team 16: Zhaobin Xu, Liwei Yuan, Junping Li  ·  Fri 13 Nov 2026, 10 minutes (upload slides to Canvas "Course Project Final Presentation" by 12 Nov)

幻灯片：`CS5478_final_presentation.pptx`。下面的英文讲稿和每页的演讲者备注是同一份文字，放映时在演示者视图里也能看到。

## 时间和分工

| 页 | 时间 | 讲者 | 内容 | 英文词数 |
|---|---|---|---|---|
| 1 | 0:00–0:20 | A | Title | 34 |
| 2 | 0:20–1:15 | A | Problem and motivation | 119 |
| 3 | 1:15–2:10 | A | Approach | 108 |
| 4 | 2:10–2:55 | A | Implementation | 97 |
| 5 | 2:55–3:35 | B | Results 1/5: at a glance | 84 |
| 6 | 3:35–4:30 | B | Results 2/5: where plans fail | 113 |
| 7 | 4:30–5:25 | B | Results 3/5: levers 1 and 2 | 112 |
| 8 | 5:25–6:15 | B | Results 4/5: lever 3 | 102 |
| 9 | 6:15–7:05 | C | Results 5/5: lever 3 | 100 |
| 10 | 7:05–7:50 | C | Demonstration | 81 |
| 11 | 7:50–8:50 | C | Discussion | 109 |
| 12 | 8:50–9:30 | C | Conclusion | 76 |

全文约 1135 个英文词，按每分钟 130 词约 8:44，加上翻页、换人和动图播放约 9 分 30 秒，留出半分钟余量。

三人分讲：A 讲第 1–4 页（0:00–2:55，约 358 词）；B 讲第 5–8 页（2:55–6:15，约 411 词）；C 讲第 9–12 页（6:15–9:30，约 366 词）。每次换人都在翻页处，谁讲哪段由组里定；一个人讲就从头念到尾。

超时怎么办：讲稿里 *[斜体方括号]* 的句子可以整句删掉，全部删掉约省 32 秒（70 词）。

## 对照课程要求

| 课程要求 | 对应页 |
|---|---|
| 首页写队号和全部成员姓名 | 1（主标题是项目名，副标题是结论） |
| Problem and motivation | 2 |
| Approach and implementation | 3、4 |
| Results and demonstration | 5–9（结果，先给全貌再逐个解释），10（两段真实 rollout 动图，放映时自动循环） |
| Discussion and conclusions：成果、局限、要点 | 11（成果四个数字、局限四条），12（回答研究问题、三条要点） |
| 个人贡献、后续计划 | 课程没有要求；放在附录备用页 13，不讲，答问时翻过去 |
| 10 分钟 | 讲稿约 9 分 30 秒 |

## 读法

- σ²：sigma squared；σ_p：sigma-p
- CEM：C-E-M（cross-entropy method）
- LeWM：第一次说全称 LeWorldModel，之后说 “L-E-W-M” 即可
- MuJoCo：moo-JO-ko；McNemar：mik-NEE-mer；DAgger：dagger；DINO-WM：DEE-no W-M

## 逐页讲稿

### 1 · Title（0:00–0:20，A）

**页面标题：** Improving Latent World Model Planning for Vision-Based Robotic Grasping  
**指：** 项目名、队号、三人名字，最后指副标题

> Hi everyone, we are Team 16: Zhaobin, Liwei and Junping. Our project is Improving Latent World Model Planning for Vision-Based Robotic Grasping. In one line, our answer is: fix the search, not the score.

**中文提示：** 报队号和三个人名字，念项目名；一句话结论：要修的是“搜索”，不是给方案“打分”。

### 2 · Problem and motivation（0:20–1:15，A）

**页面标题：** LeWM plans fast, but often fails to pick up the cube  
**指：** 先指右边三个数字，再指左图第 10 步，最后指底部研究问题

> Why does this matter? A world model lets a robot plan toward a new goal without training a new policy for it. LeWorldModel, or LeWM, learns one straight from pixels and plans in under a second, forty-eight times faster than DINO-WM.
>
> But on OGBench-Cube, a 3D pick-and-place task, it succeeds only 74 percent of the time in the paper, and 65.6 percent in our reproduction. *[A behaviour-cloning policy that does not plan at all gets 84.]* Here is a typical failure from the same start: the expert lifts the cube by step ten; LeWM never picks it up.
>
> So we asked: which lever limits planning with a frozen LeWM? Its uncertainty signal, its latent representation, or the search itself?

**中文提示：** 先讲为什么重要：世界模型让机器人不用为每个新目标重训策略就能规划。LeWM 直接从像素学，规划快（1 秒内，比 DINO-WM 快 48 倍）。问题：Cube 上论文 74%，我们复现 65.6%；不做规划的行为克隆反而有 84%。指左图：专家第 10 步就把方块拿起来，LeWM 一直没拿起来。最后念研究问题，三种颜色对应后面三个方向。

### 3 · Approach（1:15–2:10，A）

**页面标题：** One planning loop, three places to intervene  
**指：** 从左到右指四个方框，再指上面三个方向和虚线

> We keep the pretrained model frozen and change only its planner. LeWM plans with CEM, a four-step loop: propose 300 action sequences, imagine each one in latent space, score it by how close its final latent is to the goal, then keep the best 30 and repeat.
>
> Each of our levers changes one step. Uncertainty changes the score: we estimate the model’s uncertainty, sigma squared, and use it to penalise, gate or re-rank plans. Representation changes the imagination: we retrain the latent. Search changes the proposals: an action prior learned from expert data tells CEM where to start, and its spread, sigma-p, can set how wide to search.

**中文提示：** 模型冻结，只改规划器。CEM 四步：提出 300 个动作序列 → 在潜空间里想象 → 按终点离目标多近打分 → 留前 30 个再来一轮。三个方向各改一步：不确定性改“打分”（σ² 来自 MC-dropout 或 5 个成员的 ensemble，用来惩罚、门控或重排）；表征改“想象”（重训潜变量）；搜索改“提出”（从专家数据学的动作先验告诉 CEM 从哪开始，先验自己的 σ_p 决定搜多宽）。

### 4 · Implementation（2:10–2:55，A）

**页面标题：** How we built it: exact replays and paired panels  
**指：** 左边 Record → Replay → Label 和三个数字，右边表格

> To know why a plan fails, we built ground truth. We recorded 807 planning calls and replayed over 41 thousand candidate plans in the MuJoCo simulator, from the exact saved state. *[The hard part: a mid-episode replay is exact only if it also restores the physics solver’s warm start and the controller’s cached gripper pose. With those, all 600 checks matched exactly.]*
>
> For fair comparisons, every method runs on the same start states as its own LeWM baseline, with paired tests. Our three test sets have different baselines, so we always compare gains, not raw success rates.

**中文提示：** 要知道计划为什么失败，先造真值：记录 807 次规划，在 MuJoCo 里从保存的状态精确重放 4 万多个候选方案。难点：回合中途重放还要恢复物理求解器的热启动和控制器缓存的夹爪位姿，做到之后 600 次检查全部一致。公平比较：每个方法都和自己的 LeWM 基线在相同起点上配对检验；表里三个测试集的基线不同，所以只比提升。三人分讲：A 到这里交给 B，比如 “Now B will walk you through the results.”

### 5 · Results 1/5: at a glance（2:55–3:35，B）

**页面标题：** Only changing where the search starts moved success  
**指：** 从上到下指四行，最后指最下面的对照行

> Here are all our results in one picture. Each dot is one planner or model, shown as its gain over its own LeWM baseline. Scoring plans by uncertainty stays between minus 3 and plus 5 points. Retraining the latent stays between minus 4 and plus 2.
>
> Only one lever moved success a lot: starting the search from an action prior, plus 19 to plus 44 points. The bottom row is a control, the prior executed alone without planning. The next four slides explain why.

**中文提示：** 一张图看全部结果：每个点是一个规划器或模型，横轴是相对各自 LeWM 基线的提升（百分点）。σ² 打分在 −3 到 +5；重训潜变量在 −4 到 +2；只有“从动作先验开始搜索”大幅提升，+19 到 +44。最下面一行是对照：只执行先验、不做规划。

### 6 · Results 2/5: where plans fail（3:35–4:30，B）

**页面标题：** LeWM fails at the grasp, and the search is why  
**指：** 先指左图橙色 22% 的柱子，再指右上 83% 和示意图（灰点都在绿区外），最后两张小卡

> First, where do plans fail? We split the held-out tasks by what they ask for. When the cube is already in the gripper, LeWM puts it down 82 percent of the time. When it must pick the cube up from the table, it succeeds only 22 percent of the time.
>
> And these are search failures. In 83 percent of failed plans, none of the 51 candidates we replayed reaches the goal, like the grey dots in the sketch, so no scoring rule could pick a good one. The cost is fine: under LeWM’s own cost, the expert’s plan beats everything CEM found in two thirds of the states. CEM just never proposes it.

**中文提示：** 按任务要求分三类（看方块开始和结束时在桌上还是在夹爪里）：拿起 22%，一直拿着 64%，放下 82%，失败集中在“拿起”。右边：83% 的失败计划里，重放的 51 个候选一个都到不了目标，就像示意图里灰点全在绿色区域外，换什么打分都挑不出好方案。67%：按 LeWM 自己的打分，专家方案（星号）比 CEM 找到的所有候选都好，打分没问题，是 CEM 没提出来。6.6 vs 170：CEM 选中的方案模型以为离目标很近，实际很远，在钻模型误差的空子。

### 7 · Results 3/5: levers 1 and 2（4:30–5:25，B）

**页面标题：** Scoring plans by σ², or a better latent, does not help  
**指：** 左上图 +5.1 → 两组 held-out；左下紫框小柱图的第 5 根；再指右图四个变体

> Can uncertainty or a better latent fix this? On the left are the same uncertainty planners on three test sets. On our panel, gated truncation gained 5 points. On Junping’s held-out panel, at 25 and at 50 steps, the gain disappeared: every uncertainty planner scored below LeWM. The small chart shows why: sigma squared tracks the model’s real error one block ahead, a rank correlation of 0.69, but only 0.08 at block five, where CEM scores plans.
>
> On the right, better latents: a multi-step loss cuts prediction error four-fold, *[and a pose head makes the cube’s rotation readable.]* Yet planning does not improve. A better model is not a better planner here.

**中文提示：** 左上：同样的 σ² 规划器放在三个测试面板上。我们的面板 gated +5.1；换到 Junping 的 held-out 面板（25 步、50 步），所有 σ² 规划器都低于 LeWM，+5 消失了。左下小柱图解释原因：σ² 和真实误差的相关性第 1 块 0.69，到 CEM 打分的第 5 块只剩 0.08。右图：多步损失让 5 块预测误差降到约 1/4，cube-pose 头让方块旋转能读出来（0.36 → 0.95），但规划成功率没提高，加本体感觉的两个变体反而更差。gated 的意思：第一步不确定性高的候选只按更短的步数打分。

### 8 · Results 4/5: lever 3（5:25–6:15，B）

**页面标题：** An action prior fixes the grasp; more search does not  
**指：** 左上示意图（橙点落进绿区）→ 左下两条线 → 右图“拿起”那组橙色柱子

> What does fix it is a better starting point. Liwei trained a goal-conditioned behaviour-cloning network and used it to seed 75 of CEM’s 300 candidates, so the search starts near plans that work. Success rose from 72 to 96 percent and stayed near 95 at every budget, while original CEM stayed near 72 even with six times more iterations. The search is not short of compute; it is short of good proposals.
>
> On the right, the held-out panel shows the same by task type: uncertainty gating leaves the pick-up at 17 percent, while starting CEM from a prior lifts it to 98.

**中文提示：** 真正有效的是更好的起点。左上示意图：先验给的候选（橙点）直接落在能到达目标的区域里。左下（Liwei）：用目标条件的行为克隆网络给 CEM 第一轮 300 个候选里的 75 个提供初值，成功率 72% → 96%，迭代 5 到 30 次都在 95% 左右；原始 CEM 迭代多 6 倍还是 72% 左右。搜索缺的不是算力，是好的候选。右图（Junping 的 held-out 面板，同一批 500 个起点）：“拿起”这一类，σ² 门控 17%，从先验开始的 CEM 98%。三人分讲：B 到这里交给 C。

### 9 · Results 5/5: lever 3（6:15–7:05，C）

**页面标题：** Uncertainty helps by setting how wide to search  
**指：** 先指右上深蓝卡里的宽度示意，再指 73.2 → 77.4 的橙色括号，最后指 86.0 的深蓝柱子

> So where does uncertainty actually help? In setting the search width. Junping’s sigma-p CEM samples around the prior’s action, as widely as the prior is unsure. On 50-step goals that beats the same search with a fixed width by 4.2 points, and the prior alone by 5.2; both are significant in paired tests.
>
> The strongest planner reaches 86 percent: LeWM imagines several proposals from the prior, and a learned time-to-goal picks one. That is 14 points above the prior alone. *[On 25-step goals the prior alone already reaches 99 percent,]* so the world model earns its keep on longer goals.

**中文提示：** 不确定性真正有用的地方：决定搜多宽。右上示意：固定宽度每一步一样宽；σ_p-CEM 在先验有把握的地方搜得窄、没把握的地方搜得宽（宽度是 0.1 × σ_p）。50 步目标下比固定宽度高 4.2（p = 3×10⁻⁴），比只用先验高 5.2（p = 0.005），都是配对检验显著。最强的规划器 86%：先验给出几个方案，LeWM 想象每个方案的结果，用学到的“到达目标还要几步”挑一个，比只用先验高 13.8，但不用 σ²。25 步目标下只用先验就有 99%，所以世界模型在更远的目标上才显出价值。

### 10 · Demonstration（7:05–7:50，C）

**页面标题：** Demo: real rollouts, failure and recovery  
**指：** 先上面一行，再下面一行；GIF 自动循环，不用点

> Here are two real rollouts; each one first shows the outcome, then plays from the start. Top: LeWM loses the cube early. In this episode the gated planner recovers and places it at step 24, but on the held-out panel such wins cancel out.
>
> Bottom: the same task and seed. Original CEM never lifts the cube; with the prior, the gripper picks it up and carries it to the goal. We picked these to illustrate; the evidence is the paired statistics.

**中文提示：** 放映时 GIF 自动循环（先停在结局，再从头播），不用点。上面一行：专家 / LeWM / gated。LeWM 早早把方块弄丢，这一回合 gated 第 24 步放到目标，但在 held-out 面板上这种赢和输抵消了。下面一行：同一任务同一种子，原始 CEM 没拿起方块，先验引导的 CEM 拿起并送到目标。最后一定说清：这是挑出来的示例，证据是前面的配对统计。

### 11 · Discussion（7:50–8:50，C）

**页面标题：** What we achieved, and what limits our claims  
**指：** 左边四个数字，从左上到右下；再指右边四条局限

> What did we achieve? Ground truth for over 41 thousand candidate plans. With it we located the bottleneck: 83 percent of failed plans had no good candidate, so the problem is the search. We found a way for uncertainty to help, by setting the search width: plus 36 points over LeWM on 50-step goals. And our best planner reaches 86 percent.
>
> Our claims have limits. All tasks are 25- or 50-step goal-reaching segments, not full pick-and-place episodes. The latent variants and Liwei’s planner have not run on the held-out panel. *[The pretrained LeWM may have seen our test trajectories.]* And on short goals, the prior alone is already enough.

**中文提示：** 左边四个数字：4 万多个候选方案的真值；83% 的失败是搜索失败；σ_p 定搜索宽度比 LeWM 高 35.6；最好的规划器 50 步目标 86%（LeWM 41.8%）。右边局限四条：都是 25/50 步的目标片段，不是完整的抓取放置；潜变量变体和 Liwei 的方法还没在 held-out 面板上跑；预训练 LeWM 可能见过测试轨迹；短目标下只用先验就够了。

### 12 · Conclusion（8:50–9:30，C）

**页面标题：** The search is the bottleneck  
**指：** 深蓝框里的一句话，再从左到右三张卡

> So, to answer our question: for a frozen LeWM on 3D pick-and-place, planning fails in the search. Uncertainty helps by shaping the search, not by scoring plans.
>
> Three takeaways. Sigma squared is a short-horizon signal, so do not score whole plans with it. A better latent is not a better planner. And fix the search: start from a prior, and let its uncertainty set how wide to look. Thank you, and we’re happy to take questions.

**中文提示：** 一句话回答研究问题；三条要点（颜色对应三个方向）；谢谢、请提问。分工和后续计划在附录备用页（第 13 页），被问到再翻过去。

## 可能被问到的问题

**1. Who did what?**  
Zhaobin: the LeWM reproduction, exact-replay ground truth, the σ² study, the search-failure diagnosis and the latent variants. Liwei: action-prior-guided CEM on Cube and PushT and the iteration sweep. Junping: the planning code and PushT pilot, the pre-registered σ² test, the held-out panel, Gaussian priors, σ_p-CEM and time-to-goal selection. The backup slide after the conclusion lists it.  
中文：翻到附录第 13 页指着说。

**2. What would you do next?**  
Test full pick-and-place episodes with goals taken from other episodes; put every method, latent variants included, on one shared held-out panel; and fine-tune the predictor on the planner’s own 41k replays, as in DAgger.  
中文：后续三条也在附录第 13 页。

**3. Why is LeWM at 65.6, 56.4, 41.8 and 72.4%?**  
Different test sets. 36% of our panel starts at the goal; Junping’s held-out panel keeps only non-trivial starts (25- and 50-step goals); Liwei runs 5 CEM iterations on 50 held-out trajectories. That is why we compare gains.  
中文：测试集不同：我们的面板 36% 起点就在目标；held-out 只留非平凡起点；Liwei 用 5 次迭代、50 条轨迹。所以只比提升。

**4. What exactly is gated truncation?**  
Candidates whose first-step σ² is above a calibrated threshold (the median planning-time σ²) are scored at a shorter horizon instead of the full five blocks; the others keep the full horizon.  
中文：第一步 σ² 高于阈值（规划时 σ² 的中位数）的候选只按更短的步数打分，其余照常按 5 块打分。

**5. Why did gated truncation not replicate?**  
Junping’s runs used 10 CEM iterations instead of 30, a recalibrated gate and only non-trivial held-out starts; we did not isolate which factor matters.  
中文：Junping 用 10 次迭代（我们 30 次）、重新校准门限、只用非平凡起点；没有拆开验证是哪个因素。

**6. Why keep LeWM frozen?**  
To isolate the planner: if we retrained the world model, a gain could come from the model rather than from how we plan. The representation lever is the one place we retrain, and we report it separately.  
中文：为了把“规划器”的作用单独拿出来；表征方向是唯一重训的地方，单独报告。

**7. Isn’t this just behaviour cloning?**  
On 25-step goals the prior alone is enough (99.0%). On 50-step goals planning adds to it: σ_p-CEM +5.2 and time-to-goal selection +13.8 over the prior alone, both paired and significant.  
中文：25 步目标下只用先验就够；50 步目标下规划在先验之上还有 +5.2 和 +13.8，配对显著。

**8. Is time-to-goal selection behaviour cloning?**  
No: the prior proposes one action chunk per goal distance, LeWM rolls each one out, and a learned time-to-goal picks one.  
中文：不是：先验按不同目标距离各提一个方案，LeWM 想象每个结果，再用学到的“到达步数”挑。

**9. Does the prior cost extra compute?**  
It is one forward pass of a small network. Prior-guided CEM keeps 300 samples per iteration; the prior only seeds 75 of the first round.  
中文：先验只是一个小网络的一次前向；CEM 每轮还是 300 个样本，先验只提供第一轮的 75 个。

**10. Are the dot sketches data?**  
No. They illustrate what the numbers mean: CEM’s candidates miss the plans that reach the goal (83% of failures), and candidates started from a prior land among them.  
中文：不是数据，是示意图，帮助理解 83% 那个结论和先验的作用。

**11. Did the ensemble see the test data?**  
Ours was trained on 95% of the dataset episodes, which include 429 of our 450 test episodes. Junping’s held-out panel avoids this for the parts we trained, not for the pretrained LeWM.  
中文：我们的 ensemble 用了 95% 的数据，覆盖 450 个测试回合里的 429 个；held-out 面板避开了我们训练的部分，但避不开预训练 LeWM。

**12. Is 95.6% full pick-and-place success?**  
No: goal reaching 25 steps ahead (cube within 4 cm), as in the LeWM protocol; the same holds for all our numbers.  
中文：不是：和 LeWM 论文协议一样，是 25 步后的目标到达（方块在 4 cm 内）。

**13. What about PushT?**  
Junping’s earlier PushT pilot: σ² planners +1.3 to +2.2 points, not significant. Liwei’s prior-guided CEM on PushT: +19.6 points, and +30.0 over the prior alone.  
中文：PushT 上 σ² 规划器 +1.3 到 +2.2，不显著；Liwei 的先验引导 CEM +19.6，比只用先验高 30.0。

**14. How are pick / carry / put down defined?**  
By where the cube is at the start and at the goal: pick = table → gripper, carry = gripper → gripper, put down = gripper → table (Junping’s panel, 25-step goals).  
中文：看方块开始和目标时在桌上还是夹爪里。

**15. Why not one panel for everything?**  
The held-out panel already runs σ² planners and priors on the same 500 start states; the latent variants and Liwei’s CEM are not on it yet. One shared panel for every method is our first next step.  
中文：held-out 面板已经把 σ² 和先验放在同一批起点上；潜变量和 Liwei 的方法还没上，这是下一步第一件事。

## 演练提示

- 至少完整计时排练一次，三人分讲时要练换人那一句。
- 数字只说讲稿里的整数或一位小数，图上有完整数字。
- Demo 页的动图放映时自动播放，先停在结局，再从头播，不用点。
- 课程用老师的电脑放映：12 号前把 pptx 交到 Canvas 的 “Course Project Final Presentation”。字体只用了 Cambria 和 Calibri，Office 都自带。
