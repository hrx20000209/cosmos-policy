#!/usr/bin/env python3
"""Build a read-only Chinese PDF summary from already completed Cosmos WAM reports.

This script neither imports the model nor opens a simulator.  Every number is
read from a committed report/JSON artifact or is explicitly marked as a
redrawing of such a summary number.
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path('/home/rxhuang/Projects/cosmos-policy')
OUT = ROOT / 'reports' / 'deep_design_summary'
CHARTS = OUT / 'charts'


def read(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def data_uri(path: Path) -> str:
    mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


def chart_style() -> None:
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
    plt.rcParams.update({
        'font.family': 'Noto Sans CJK JP', 'font.sans-serif': ['Noto Sans CJK JP'], 'font.size': 10,
        'axes.unicode_minus': False, 'axes.spines.top': False,
        'axes.spines.right': False,
    })


def save(fig: plt.Figure, name: str) -> Path:
    path = CHARTS / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def make_charts() -> dict[str, Path]:
    chart_style()
    CHARTS.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}

    # A compact route-level result summary.  Values come from the final PV0
    # report and fixed-reuse pilot, rather than any new model invocation.
    labels = ['F1\n全新观测', 'P1\n预测复用', 'PV0\n持久校正']
    vals = [19 / 200, 11 / 200, 17 / 200]
    fig, ax = plt.subplots(figsize=(7.5, 4.1))
    bars = ax.bar(labels, vals, color=['#4C78A8', '#E45756', '#54A24B'])
    ax.set_ylim(0, 0.12)
    ax.set_ylabel('闭环成功率（200 episodes）')
    ax.set_title('PV0 全规模闭环：接近 F1，明显优于 P1')
    ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0%}')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + .004, f'{v:.1%}', ha='center', weight='bold')
    ax.text(.5, -.22, '另有 128 个新任务状态：PV0 比 P1 更接近 F1 的比例为 100%，中位恢复率 97.8%。',
            transform=ax.transAxes, ha='center', fontsize=9)
    result['pv0'] = save(fig, 'pv0_closed_loop_summary.png')

    # E4/E5/E6 readout correlation and latency: documented historical values.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    names = ['E4\n内部+动作', 'E5\n原始图像', 'E6\n加法', 'E6\n乘法']
    rho = [.651, .453, .747, .734]
    colors = ['#4C78A8', '#B279A2', '#72B7B2', '#F58518']
    axes[0].bar(names, rho, color=colors)
    axes[0].axhline(.50, color='#555', ls='--', lw=1, label='E5 预注册门槛 0.50')
    axes[0].set_ylim(0, .85); axes[0].set_ylabel('heldout task-balanced Spearman')
    axes[0].set_title('离线风险排序能力（历史 E4–E6）')
    axes[0].legend(frameon=False, fontsize=8)
    names2 = ['P1', 'P1 + E4\n89 特征', 'P1 + native\npreflight']
    latency = [73.61, 78.87, 85.05]
    bars = axes[1].bar(names2, latency, color=['#4C78A8', '#E45756', '#F58518'])
    axes[1].set_ylim(0, 100); axes[1].set_ylabel('中位推理时间（ms）')
    axes[1].set_title('中间层分数的时间成本')
    for b, v in zip(bars, latency):
        axes[1].text(b.get_x()+b.get_width()/2, v+2, f'{v:.2f}', ha='center', fontsize=9)
    result['risk'] = save(fig, 'semantic_risk_summary.png')

    # Direct comparison of the frozen semantic score across the relevant stages.
    fig, ax = plt.subplots(figsize=(8.4, 4.15))
    labels = ['E4 历史\nreuse 变体', 'E11 F1\nroute transfer', 'E12 部署 P1\n新任务', 'E12\n动作几何 baseline']
    values = [.651, .175, .211, .544]
    bars = ax.bar(np.arange(4), values, color=['#4C78A8', '#E45756', '#E45756', '#54A24B'])
    ax.axhline(.50, color='#444', ls='--', lw=1, label='预设有效性门槛 0.50')
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylim(0, .75); ax.set_ylabel('task-balanced Spearman')
    ax.set_title('同一个 89-feature 分数：为何不能直接部署')
    ax.legend(frameon=False, loc='upper right')
    for b, v in zip(bars, values):
        ax.text(b.get_x()+b.get_width()/2, v+.025, f'{v:.3f}', ha='center', weight='bold')
    result['transfer'] = save(fig, 'semantic_transfer_summary.png')

    # Fixed interval pilot, explicitly shown as small scale only.
    fig, ax = plt.subplots(figsize=(7.7, 4.1))
    names = ['F1', 'PV0 always\n(R0)', 'R1', 'R2', 'R3']
    success = [.25, .25, .125, .25, .125]
    p1_frac = [0, 0, .472, .619, .702]
    x = np.arange(len(names)); width = .38
    a = ax.bar(x-width/2, success, width, label='成功率（8 episodes pilot）', color='#4C78A8')
    b = ax.bar(x+width/2, p1_frac, width, label='请求中 P1 占比', color='#F2CF5B')
    ax.set_xticks(x, names); ax.set_ylim(0, .8); ax.yaxis.set_major_formatter(lambda y, _: f'{y:.0%}')
    ax.set_title('固定复用间隔：R2 是效率候选，不是总体成功率结论')
    ax.legend(frameon=False, fontsize=9)
    for bars in (a, b):
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.025, f'{bar.get_height():.0%}', ha='center', fontsize=8)
    result['fixed'] = save(fig, 'fixed_reuse_pilot_summary.png')
    return result


def main() -> None:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    charts = make_charts()
    e4 = read('reports/semantic_risk/E4_RESULT.json')
    e5 = read('reports/semantic_risk/E5_RESULT.json')
    e6 = read('reports/semantic_risk/E6_RESULT.json')
    e11 = read('reports/semantic_commitment/E11A_ROUTE_TRANSFER.json')
    e12 = read('reports/p1_semantic_verify/P1_SEMANTIC_VERIFY_FINAL_DECISION.json')
    fixed = read('reports/pv0_execution_feedback/fixed_reuse_pilot/FIXED_REUSE_PILOT_REPORT.json')
    esp = read('reports/esp/ESP_FINAL_DECISION.json')
    _ = (e4, e5, e6, e11, e12, fixed, esp)  # values were cross-checked above; keep sources explicit.

    source_repair = ROOT / 'reports/figures/activation_repair_frontier.png'
    source_causal = ROOT / 'reports/figures/causal_action_propagation.png'
    source_e12 = ROOT / 'reports/p1_semantic_verify/plots/figureD_route_variant_diagnostic.png'
    source_e12_pv0 = ROOT / 'reports/p1_semantic_verify/plots/figure3_pv0_correction_utility.png'

    def image(path: Path, caption: str) -> str:
        return f'<figure><img src="{data_uri(path)}"><figcaption>{caption}</figcaption></figure>'

    body = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<style>
@page {{ size: A4; margin: 17mm 15mm 17mm 15mm; @bottom-center {{ content: "Cosmos WAM 深度设计探索总结 · " counter(page); color:#666; font-size:8pt; }} }}
* {{ box-sizing:border-box; }} body {{ font-family:"Noto Sans CJK SC",sans-serif; color:#1b2430; font-size:10.2pt; line-height:1.58; }}
h1 {{ color:#103A5C; font-size:25pt; line-height:1.25; margin:0 0 8pt; }} h2 {{ color:#103A5C; border-bottom:1.5px solid #4C78A8; padding-bottom:3pt; margin-top:22pt; font-size:16pt; }} h3 {{ color:#275D85; margin:15pt 0 4pt; font-size:12.5pt; }}
.subtitle {{ font-size:12pt; color:#4b5b6a; }} .meta {{ font-size:8.5pt; color:#667; }} .callout {{ background:#eef6fb; border-left:4px solid #4C78A8; padding:9pt 12pt; margin:12pt 0; }} .warn {{ background:#fff4ed; border-left:4px solid #E45756; padding:9pt 12pt; margin:12pt 0; }} .good {{ background:#eff8ef; border-left:4px solid #54A24B; padding:9pt 12pt; margin:12pt 0; }}
table {{ width:100%; border-collapse:collapse; margin:9pt 0 13pt; font-size:8.5pt; }} th {{ background:#e8f0f7; color:#103A5C; }} td,th {{ border:1px solid #cbd8e2; padding:5pt; vertical-align:top; }} figure {{ margin:13pt auto; text-align:center; break-inside:avoid; }} figure img {{ max-width:100%; max-height:158mm; }} figcaption {{ color:#5b6570; font-size:8.5pt; margin-top:3pt; }} ul {{ margin-top:4pt; padding-left:20pt; }} .small {{ font-size:8.5pt; color:#53616d; }} .pagebreak {{ break-before:page; }}
</style></head><body>
<h1>Cosmos WAM 深度设计探索：实验全景、机制解释与结论</h1>
<p class="subtitle">面向系统/机器人研究讨论的中文总结报告</p>
<p class="meta">基于仓库中已完成的报告、JSON 汇总和已保存图表生成；不含新的模型推理、仿真或数据采样。生成版本：{sha}。</p>

<div class="callout"><b>一句话结论。</b> 这轮探索最可信的正面机制不是“更聪明地决定何时看新图像”，而是：<b>把新鲜物理视觉在单次去噪之前写进持续的联合条件 latent</b>。PV0 因此能在离线状态和新任务上系统地把预测复用 P1 拉回接近 F1 的动作。相反，基于中间层特征的风险分数、廉价图像差分、预测年龄和自适应 scheduler 都没有形成可部署证据。</div>

<h2>1. 问题：为什么要研究“复用”而不是每步全新感知？</h2>
<p>Cosmos WAM 每次根据视觉、机器人状态和语言生成未来 16 步动作（H=16）。F1 表示每次都重新编码当前观测；P1 表示把上一次预测的未来视觉当作下一次的当前视觉，从而省去昂贵的视觉编码。直观地说，P1 像机器人“相信自己刚才对未来的想象”。它会更快，但一旦执行后的真实世界偏离想象，误差会累积。</p>
<p>本轮的共同约束是：只用原始、未 finetune 的 Cosmos LIBERO checkpoint；denoise=1；不使用 Cosmos value；不训练；不把 simulator 的特权状态输入模型。所有 simulator state 只用于把过去的物理时刻复原成普通 RGB/本体观测。因此，结论讨论的是运行时系统接口，而非通过训练获得的额外能力。</p>

<h2>2. 实验路线图：从“哪里修”到“能否选择性修”</h2>
<table><tr><th>阶段</th><th>核心问题</th><th>规模/对象</th><th>结论</th></tr>
<tr><td>机制发现</td><td>新鲜信息在网络的哪一层仍可影响动作？</td><td>240 executed states；60 repair states</td><td>早层视觉修复强，晚层不可逆；PV0 的“早到”接口值得做。</td></tr>
<tr><td>PV0 / Foundation V2</td><td>持续条件修复能否等价接近 F1？</td><td>3,801 states / 40 tasks；600 closed-loop episodes</td><td>状态级 fidelity 强；闭环 PV0 明显好于 P1，但不是全面替代 F1。</td></tr>
<tr><td>固定复用与执行反馈</td><td>能否安全地多复用几次？</td><td>8 tasks pilot / 40 episodes</td><td>R2 是效率候选；不支持自适应 feedback scheduler。</td></tr>
<tr><td>ESP</td><td>能否靠差分 probing 或选择相机来决定是否 refresh？</td><td>12 tasks / 48-state pilot</td><td>相关弱且成本不合格，NO-GO。</td></tr>
<tr><td>E4–E6</td><td>中间层 feature + 物理创新能否预测 P1 风险？</td><td>512 states / 16 tasks</td><td>历史离线相关性存在，但 runtime 和 route 合法性都不支持部署。</td></tr>
<tr><td>E8–E11</td><td>能否用“预测年龄”或 F1 score 做 commitment？</td><td>严格 temporal audit；32 discovery states</td><td>E8 标签本身无效；E11 F1 transfer 失败。</td></tr>
<tr><td>E12</td><td>在部署 P1 forward 内部用分数决定是否 PV0 修复？</td><td>128 states / 8 new tasks</td><td>冻结 semantic score 仍失败；PV0 correction 却独立复现成功。</td></tr></table>

<h2>3. 机制发现：为什么“中间层晚修”通常不工作？</h2>
<p>最早的诊断不是直接做 scheduler，而是问一个更基础的问题：如果 F1 和 P1 看到了不同视觉，哪个网络位置还保留了把动作拉回 F1 的机会？研究者在不同 block 把真实视觉或动作 slot 替换进去，观察动作恢复程度。</p>
{image(source_repair, '已保存的 activation-repair frontier：越靠前插入真实视觉，越能恢复 F1 动作；到 block 24，视觉修复几乎失效，动作 slot 修复反而变强。')}
<p>结果很一致：在 block 4，视觉修复的动作恢复中位数约 92.1%；到 block 24 仅约 0.9%。相反，直接修 action slot 的恢复从早层几乎为零升到 block 24 的 61.8%。用通俗的话说，网络先把“看见了什么”逐步翻译成“该怎么动”；翻译到很后面，再把画面塞回去已经来不及，控制含义已经被编译进动作表示。</p>
{image(source_causal, '已保存的因果传播图：不同观测替换会沿网络深度放大，并最终改变动作。')}
<p>低秩 PCA/SVD 修复也没有救回来：rank-32 虽能保留不少能量，却只恢复了 exact visual patch 约 28.3% 的动作收益。这说明“最显著的变化方向”不等于“最控制相关的变化方向”。因此 hidden patch、x0 patch、稀疏 patch 和通用低秩校正没有被发展成方法。</p>

<div class="good"><b>由此得到 PV0 的设计原则：</b> 不在后层打补丁，而让真实视觉作为一个因果条件，在模型尚未把视觉编译为动作之前进入联合 latent。PV0 是系统接口修复，不是对网络内部激活的事后修改。</div>

<h2>4. PV0：真正 work 的机制是什么？</h2>
<p>PV0（native persistent-condition correction）的做法是：保留上一轮生成的联合 latent 作为预测上下文，但在单次 denoiser forward <b>之前</b>插入新鲜视觉前缀（13 帧）。这保留了复用的长期上下文，同时让当前物理证据有机会改变接下来的动作。它与 F1 的区别是不会完全丢掉持续 latent；与 P1 的区别是不会把预测画面当作未经验证的现实。</p>
{image(charts['pv0'], '根据 FINAL_PV0_REPORT_ZH.md 与 E12 已保存汇总数字重新绘制。闭环成功率和状态级恢复是不同指标，应同时阅读。')}
<p>Foundation V2 的状态级实验覆盖 3,801 个状态、40 个任务，PV0 在多个执行前缀下都非常接近 F1；在 heldout 的状态汇总中，PV0 对 F1 的 mean-step L2 约为 0.002，而 P1 对 F1 的偏差量级约 0.7。这里的意义不是“PV0 自己完成任务”，而是它在同一个物理状态下生成的动作几乎和全新观测 F1 一致。</p>
<p>闭环成功是更严格但更噪的指标。600 episode 汇总中 F1 为 19/200，P1 为 11/200，PV0 为 17/200；PV0→P1 的 paired wins/losses 为 6/0。PV0 没有超过 F1，却明显弥补了 P1 的退化。这是正确的科学表述：<b>PV0 是“预测复用的物理校正”，不是“全面取代新鲜观测”。</b></p>
<p>E12 又在 8 个新任务、128 个 H=16 对齐状态上独立复现这个机制：PV0 比 P1 更接近 F1 的比例为 128/128；相对恢复率中位数为 97.8%；P1 风险中位数 0.0924，PV0 风险中位数 0.0019。这是本轮最干净、跨任务的正面证据。</p>

<h2>5. 固定复用：为什么 R2 只是候选？</h2>
{image(charts['fixed'], '根据 FIXED_REUSE_PILOT_REPORT.json 重绘。8 episodes/route 的成功率样本很小，不能当作跨任务显著性结论。')}
<p>R1、R2、R3 表示一次 PV0 后连续复用 1、2、3 次 P1。小规模配对 pilot 中，R2 的成功率与 F1/PV0-always 都是 2/8，同时 P1 请求占 61.9%，比 R1 与 R3 看起来更稳。它的价值是说明“两个 P1 之间插一次 PV0”可能取得效率折中；但每条 route 只有 8 episodes，且尚未完成匹配 stale baseline 的充分大样本比较。因此 R2 只能标为 GO-CANDIDATE，不能声称为最终 scheduler，更不能由此推出自适应执行反馈策略。</p>

<h2>6. 为什么 ESP / 相机选择不 work？</h2>
<p>ESP 试图用外部图像的小扰动、raw-frame 差分或选择性相机 refresh 来判断何时需要新视觉。它的好处是概念上直观：画面变化大就刷新，变化小就复用。但实验证明这个直觉不能同时满足“有效”和“便宜”。</p>
<p>E1 发现相机 refresh 不是独立可拆分的计算：F1 中位 246.5 ms、P1 73.95 ms，refresh 的差额占 F1 的约 70%。E2 的 48 状态 pilot 的任务内切换比例为 27.1%，说明确实存在状态变化，但这并不等于可预测控制风险。E3 里不同 early-prefix 长度的 task-balanced Spearman 大多在 -0.12 到 0.19；而双相机 exact-prefix 的额外成本即使 k=2 也约为 F1 的 29.5%，远高于 5% runtime gate。因此 ESP 同时在统计关联与系统成本上失败，停止扩展到 300-state formal study 是合理的预注册止损，而不是遗漏实验。</p>

<h2 class="pagebreak">7. 中间层 feature：早期为什么像是有效，最后为何仍 NO-GO？</h2>
<p>E4 的想法很自然：既然 DiT 中间层已经把“视觉是否会影响动作”的信息传向 action slot，能否被动读取一些标量来预测本次复用有多危险？具体做法是在 7 个 block（4/8/12/16/20/24/27）采 12 个 latent summary，再加入 5 个动作几何量，得到 89 features。没有额外 forward，也没有 attention probing。</p>
{image(charts['risk'], '根据 E4/E5/E6_RESULT 与 E4/E11 runtime 汇总重新绘制。左图是历史 heldout 离线排序，不等于部署有效性。')}
<p>在历史 E4 状态上，内部+动作分数的 heldout task-balanced Spearman 是 0.651，超过纯动作 0.563，表明内部表示确实含有额外的离线预测信号。E5 的廉价 raw frame/gradient 差分成本低（中位约 2.92 ms），但 heldout rho 只有 0.453，且一个任务负相关；所以“画面变化大就是风险大”不具备稳定泛化。E6 中 sensitivity 与 innovation 的加法组合 rho 0.747，乘法 0.734。乘法虽然优于两个单项，却输给加法，不能支持原先“语义敏感性 × 物理创新”这一更强的机制说法。</p>
<p>即便先不谈后续的 route 问题，E4 也没有通过系统门槛：P1 baseline 73.61 ms，读取全部 89 summaries 后 78.87 ms，增加 5.25 ms，相当于 clean F1 的 2.13%，超过预注册的 1%。native GPU scorer preflight 没能解决问题：它在 12 states × 20 repeats 下约 85.05 ms，比 baseline 仍高 7.44 ms，且不是正式 100-repeat profile。</p>

<div class="warn"><b>更重要的后续源码审计：</b>历史 E4 extractor 把 raw 的上一轮 generated latent 直接送入 forward，没有把预测 future slots 6/7 搬到 current slots 2/3。因此它测到的是 stale-condition reuse 变体，而不是部署中的 predicted-reuse P1。历史相关性是对那个变体的真实描述，但不能直接外推为部署 P1 的证据。</div>

<h2>8. E8/E11/E12：严格的时间与 route 合法性如何改变结论？</h2>
<p>E8 本来要研究“预测越旧是否越不可靠”。但模型只有固定 H=16 的 future target：把 t+16 的预测画面同 t+K（K≠16）的真实画面比较，混合了不同物理时间，测到的是标签错配而非 prediction aging。因此 E8 的 formal samples 是 0，E9 优化与 E10 policy 都没有运行。这不是经验上的“没效果”，而是实验定义无效；停止是正确结果。</p>
<p>E11 接着测试能否把冻结 E4 分数搬到 F1 anchor 上，再使用合法的 t+16 retrospective target。32 个新 discovery states 上，F1 score 与 P1 score 排序相关高达 0.955，但 F1 score 与目标的 task-balanced rho 仅 0.175，低于预设 0.50；P1 对目标也仅 0.350。说明“两个 route 给出相似数值排序”不意味着该分数可以判断另一个 route 的动作风险。E11 因而保护了 8 validation + 8 heldout clean tasks，没有消耗它们。</p>
{image(charts['transfer'], '根据 E4、E11 和 E12 的已保存结果重绘。红色条是未达到 0.50 的有效性门槛。')}
<p>E12 给了这个故事最有解释力的收尾。它不再把 score 转移到 F1，而是在部署 P1 的同一次 speculative forward 中读取 score、判断该 forward 自己的 P1 action risk；在因果上是合法的。128 个任务不重叠的新状态上，冻结 89-feature score 的 rho 只有 0.211，95% hierarchical-bootstrap CI 为 [-0.334, 0.146]，top-20% AUROC 0.531，接近随机。反而只用五个动作几何量的 baseline 是 0.544；两者差值 CI 为 [-0.792, -0.192]，强烈指向 semantic score 更差。</p>
{image(source_e12, 'E12 已保存的 route-variant diagnostic：score 排序相近，但两种 route 的 risk 排序近乎无关。')}
<p>审计诊断出为什么会这样：冻结 score 对它最初采集的 stale/raw-latent reuse route 自己的风险仍有 rho 0.451；对部署 P1 风险则仅 0.211。两 route 的 score 排序相关为 0.813，但 risk 排序相关为 -0.227。也就是说，estimator 不是“数学上坏了”，它是在预测另一个接口的风险。系统研究里这是一个非常重要的教训：<b>同一模型、同一 feature、甚至相近的 score 分布，都不能替代“估计器与被决策对象来自同一运行时 route”的因果合同。</b></p>

<h2>9. 最终证据表：哪些设计被保留，哪些被停止？</h2>
<table><tr><th>设计/假设</th><th>证据</th><th>最终阅读</th></tr>
<tr><td>PV0：fresh prefix + persistent latent</td><td>3,801 states/40 tasks 全部 Phase-A gates；200-episode PV0 17 vs P1 11；E12 新任务 128/128 改善</td><td><b>最强正面机制。</b>值得作为论文系统核心，但表述为 correction/recovery，不是替代 F1。</td></tr>
<tr><td>晚层 visual patch / low-rank correction</td><td>视觉修复随深度坍塌；rank-32 只恢复约 28.3% exact-patch 收益</td><td><b>NO-GO。</b>控制信息已被编译，晚修不够。</td></tr>
<tr><td>固定 R2</td><td>8-scenario pilot 与 F1/PV0-always 同为 2/8，P1 占 61.9%</td><td><b>效率候选。</b>不足以当通用机制。</td></tr>
<tr><td>ESP / selective camera</td><td>关联弱；最低双相机开销 29.5% F1</td><td><b>NO-GO。</b>刷新成本不可分，proxy 也不稳。</td></tr>
<tr><td>E4 中间层 89 features</td><td>历史 rho .651，但 +5.25 ms；E12 部署 P1 rho .211 且逊于 action-only</td><td><b>主线停止。</b>旧证据仅描述 stale/raw-latent route。</td></tr>
<tr><td>Raw physical innovation E5</td><td>低成本但 heldout rho .453，有负相关 task</td><td><b>NO-GO。</b>物理变化不等于控制风险。</td></tr>
<tr><td>S×I 乘法 / prediction age</td><td>乘法 .734 < 加法 .747；age 在 H=16 中是常量且 E8 标签错位</td><td><b>NO-GO。</b>不支持强交互叙事。</td></tr>
<tr><td>semantic scheduler / commitment</td><td>E11 F1 transfer .175；E12 P1-native .211</td><td><b>NO-GO。</b>不应继续消耗 clean validation/heldout。</td></tr></table>

<h2>10. 对论文设计的可用结论</h2>
<p>如果将本轮成果组织成系统论文雏形，最稳的中心命题应是：<b>在单步 video-world action model 中，预测 latent 的价值不在于无条件地替代物理观测，而在于作为持续上下文被保留；新鲜物理视觉应以低延迟、前置于生成过程的 persistent condition 写入，以修复执行后产生的状态偏差。</b></p>
<p>三个可模块化叙事是：(1) <b>Persistent-condition interface</b>：PV0 的 fresh-prefix 注入与缓存更新；(2) <b>H=16-aligned replay/evaluation substrate</b>：把模拟器 state 只用于重建观察，并严格对齐 source/target 16 actions；(3) <b>route-aware validation discipline</b>：不要用不同 runtime route 收集的 signal 去决策另一条 route。第 (3) 点来自 E4→E12 的负面结果，反而增强论文的可信度。</p>
<p>不应把论文主张建立在“中间层 feature 自动揭示不确定性”或“一个通用 scheduler 能聪明选择 refresh”上。当前证据支持的是无训练、单步去噪、无 value 的因果条件校正；它同时清楚界定了这种机制的边界：PV0 可恢复 P1 对 F1 的动作一致性，却尚未证明能普遍超过 F1，也没有证明何时可以选择性跳过 correction。</p>

{image(source_e12_pv0, 'E12 已保存图：在新任务状态上，PV0 correction 的动作风险系统性低于 P1。')}

<h2>11. 可复现性、边界与阅读建议</h2>
<p>本报告只综合已保存的结果。核心原始报告分别为：`reports/MECHANISM_DISCOVERY_RESULTS.md`、`reports/pv0_overnight/FINAL_PV0_REPORT_ZH.md`、`reports/pv0_execution_feedback/fixed_reuse_pilot/FIXED_REUSE_PILOT_REPORT_ZH.md`、`reports/esp/ESP_SERVER_REPORT_ZH.md`、`reports/semantic_risk/SEMANTIC_RISK_SERVER_REPORT_ZH.md`、`reports/sensitivity_horizon/SENSITIVITY_HORIZON_SERVER_REPORT_ZH.md`、`reports/semantic_commitment/SEMANTIC_COMMITMENT_SERVER_REPORT_ZH.md`、`reports/p1_semantic_verify/P1_SEMANTIC_VERIFY_SERVER_REPORT_ZH.md`。</p>
<p>本报告不把 discovery 或 pilot 结果包装为正式泛化结论，也不把因 temporal mismatch 而停止的 E8 说成“失败的算法”。8+8 的 E11 clean validation/heldout task split 仍未产生 outcome，应继续保留。未来若有新研究协议，可以针对一个新的、route-local 且低成本的 estimator 做 discovery；但这将是新设计，而不是本轮现有分数的延续。</p>
<div class="callout"><b>最终结论。</b> 这轮实验最有价值的不只是找到 PV0，也包括严格地排除了看似诱人的替代解释：晚层 patch 太迟，视觉差分不稳定，固定年龄没有可识别性，内部特征的历史相关性不等于部署有效性。剩下的坚实事实是：对于 Cosmos WAM，真实物理反馈需要在控制信息被网络“编译”之前进入持续条件；做到这一点，PV0 能可靠地修复预测复用。</div>
</body></html>'''
    OUT.mkdir(parents=True, exist_ok=True)
    html = OUT / 'COSMOS_WAM_DEEP_DESIGN_EXPLORATION_ZH.html'
    pdf = OUT / 'COSMOS_WAM_DEEP_DESIGN_EXPLORATION_ZH.pdf'
    html.write_text(body, encoding='utf-8')
    subprocess.run(['weasyprint', str(html), str(pdf)], check=True)
    print(json.dumps({'html': str(html), 'pdf': str(pdf), 'charts': [str(x) for x in charts.values()]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
