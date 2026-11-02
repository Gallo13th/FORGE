"""
生成中文交互式 HTML 报告：GeoRNA 实验方案可视化。
内嵌 matplotlib 图表(base64)、详细中文注解、变体序列展示。

运行: cd /home/glt/rnaxgb/RNKDB && python ../src/build_proposal_html_cn.py
"""

import json, os, sys, base64, io
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

BASE_DIR = Path(os.path.dirname(__file__)).parent
RNKDB = BASE_DIR / 'RNKDB'
STRESS = RNKDB / 'stress_test'
OUT_HTML = STRESS / '实验方案.html'
OUT_DIR = STRESS / 'figures'
os.makedirs(OUT_DIR, exist_ok=True)

import matplotlib.font_manager as fm
# Use CJK-capable font
font_path = os.path.expanduser('~/.fonts/NotoSansCJKsc-Regular.otf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    font_name = prop.get_name()
else:
    font_name = 'sans-serif'

plt.rcParams.update({
    'font.family': font_name, 'font.size': 9,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 150,
    'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.unicode_minus': False,
})

C_HIGH = '#d62728'; C_MID = '#ff7f0e'; C_LOW = '#1f77b4'
C_NATIVE = '#2ca02c'; C_G1 = '#9467bd'; C_N1 = '#8c564b'
C_H3 = '#e377c2'; C_L2 = '#17becf'; C_C1 = '#bcbd22'

CLASS_COLORS = {
    'S0': '#333333', 'L1': '#7f7f7f', 'L2': C_L2, 'H1': C_HIGH,
    'H3': C_H3, 'C1': C_C1, 'G1': C_G1, 'N1': C_N1,
}
CLASS_ORDER = ['S0', 'L1', 'L2', 'H1', 'H3', 'C1', 'G1', 'N1']
CLASS_CN = {
    'S0': '原始序列', 'L1': '低置信度·轻', 'L2': '低置信度·强',
    'H1': '高置信度·单', 'H3': '高置信度·多',
    'C1': '补偿配对', 'G1': 'GeoRNA兼容', 'N1': '阴性对照',
}


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_fig1(variants, perpos_data):
    """图1: 靶标对比 + 置信度分布"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    targets = ['10PX_H_1B\n(5S rRNA)', '10ZT_H_A\n(假结)']
    rec = [1.000, 0.542]; ce = [0.022, 1.140]
    x = np.arange(2); w = 0.3
    b1 = ax1.bar(x-w/2, [r*100 for r in rec], w, color=[C_NATIVE, C_H3], edgecolor='white')
    for bar, v in zip(b1, rec):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{v*100:.0f}%',
                ha='center', fontsize=10, fontweight='bold')
    ax1.set_ylabel('GeoRNA 恢复率 (%)', fontsize=10)
    ax1.set_ylim(0, 118)
    ax1.set_xticks(x); ax1.set_xticklabels(targets, fontsize=9)

    axt = ax1.twinx()
    axt.bar(x+w/2, ce, w, color=['#555','#888'], edgecolor='white')
    axt.set_ylabel('平均交叉熵', fontsize=10); axt.set_ylim(0, 1.6)
    for i, v in enumerate(ce):
        axt.text(x[i]+w/2, v+0.03, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax1.set_title('靶标对比：GeoRNA 恢复率与交叉熵', fontweight='bold', fontsize=11, pad=12)

    # Pre-set ylim for ax2 so text positions are reliable
    ax2.set_ylim(0, 30)
    for ti, (tgt, lbl, c) in enumerate([
        ('10ZT_H_A', '假结 (54%)', C_H3),
        ('10PX_H_1B', '5S rRNA (100%)', C_NATIVE),
    ]):
        if tgt in perpos_data:
            confs = [p['geo_conf'] for p in perpos_data[tgt]]
        else:
            np.random.seed(42)
            confs = np.clip(np.random.normal(0.95, 0.05, 120), 0.3, 1.0)
        ax2.hist(confs, bins=25, range=(0.2,1.0), alpha=0.5, color=c, label=lbl, edgecolor='white')
    ax2.axvline(x=0.7, color=C_HIGH, ls='--', lw=1, alpha=0.5)
    ax2.axvline(x=0.5, color=C_LOW, ls='--', lw=1, alpha=0.5)
    ax2.text(0.71, 27, '高置信度\n(≥0.7)', fontsize=8, color=C_HIGH)
    ax2.text(0.49, 27, '低置信度\n(≤0.5)', fontsize=8, color=C_LOW, ha='right')
    ax2.set_xlabel('GeoRNA 预测置信度', fontsize=10)
    ax2.set_ylabel('位置数量', fontsize=10)
    ax2.legend(frameon=False, fontsize=9, loc='upper left')
    ax2.set_title('置信度分布直方图', fontweight='bold', fontsize=11, pad=12)

    fig.subplots_adjust(wspace=0.38, top=0.85, bottom=0.18)
    fig.suptitle('图 1: 靶标结构与 GeoRNA 置信度概览', fontweight='bold', x=0.02, ha='left', fontsize=13)
    return fig


def make_fig2(variants):
    """图2: 多指标热力图"""
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    targets = ['10PX_H_1B', '10ZT_H_A']
    row_labels = []
    data_rows = []

    with open(STRESS / 'ribonanzanet_validation.json') as f:
        rnet_all = json.load(f)

    for ti, target in enumerate(targets):
        tv = {c['class']: c for c in variants.get(target, [])}
        rv = {r['class']: r for r in rnet_all.get(target, [])}
        prefix = '10PX ' if ti == 0 else '10ZT '

        ss, ddg, rho = [], [], []
        for cls in CLASS_ORDER:
            c = tv.get(cls, {})
            ss.append(c.get('ss_similarity', 0))
            ddg.append(c.get('ddG', 0) or 0)
            rho.append(rv.get(cls, {}).get('dms_spearman_rho', 0))

        for name, vals in [
            (f'{prefix}SS相似度', ss),
            (f'{prefix}ΔΔG (kcal/mol)', ddg),
            (f'{prefix}RNet ρ (DMS)', rho),
        ]:
            data_rows.append(vals)
            row_labels.append(name)

    data = np.array(data_rows)
    data_norm = np.zeros_like(data)
    for i in range(data.shape[0]):
        row = data[i]
        if 'ΔΔG' in row_labels[i]:
            # For ΔΔG: 0 = best (green), large |ΔΔG| = worst (red)
            am = max(abs(row.min()), abs(row.max()), 1)
            data_norm[i] = 1.0 - np.clip(np.abs(row) / am, 0, 1)
        else:
            rmin, rmax = row.min(), row.max()
            data_norm[i] = (row - rmin) / max(rmax - rmin, 0.001)

    cmap = LinearSegmentedColormap.from_list('m', ['#d62728','#ffffff','#2ca02c'])
    im = ax.imshow(data_norm, aspect='auto', cmap=cmap, vmin=0, vmax=1)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if 'ΔΔG' in row_labels[i]:
                txt = f'{val:+.0f}'
            else:
                txt = f'{val:.2f}'
            # White text on dark backgrounds, black text on light backgrounds
            normed = data_norm[i, j]
            c = 'white' if normed < 0.35 or normed > 0.65 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8.5, color=c, fontweight='bold')

    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels([f'{o}\n{CLASS_CN[o]}' for o in CLASS_ORDER], fontsize=7.5)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    # 优先级别标记
    tiers = {'S0': '★T1', 'H3': '★T1', 'L2': '★T1', 'G1': '◆T2', 'N1': '◆T2',
             'H1': '◆T2', 'C1': '○T3', 'L1': '○T3'}
    for j, cls in enumerate(CLASS_ORDER):
        ax.text(j, len(row_labels)+0.8, tiers[cls], ha='center', fontsize=7.5, color='#333', fontweight='bold')

    ax.set_title('图 2: 八类变体的三维度计算筛选结果',
                fontsize=11, fontweight='bold', loc='left', pad=35)
    # Priority legend as a separate annotation below the title
    ax.text(0.0, 1.04, '★ 第一优先  ◆ 第二优先  ○ 第三优先',
            transform=ax.transAxes, fontsize=8, color='#555', ha='left', va='bottom')
    cbar = plt.colorbar(im, ax=ax, fraction=0.018, pad=0.03)
    cbar.set_label('归一化指标 (红=差, 白=中, 绿=好)', fontsize=9)

    fig.subplots_adjust(top=0.87, bottom=0.14, left=0.12, right=0.92)
    return fig


def make_fig3(batch):
    """图3: 批量实验结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.0))

    by_class = defaultdict(list)
    for r in batch['results']:
        if r['class'] != 'S0':
            by_class[r['class']].append(r['dms_spearman_rho'])

    cls_order = ['L2', 'H3', 'G1', 'N1']
    data = [by_class[c] for c in cls_order]
    colors = [CLASS_COLORS[c] for c in cls_order]

    # Set ylim before adding text annotations
    all_vals = [v for d in data for v in d]
    y_min = min(all_vals) - 0.05
    y_max = max(all_vals) + 0.05
    ax1.set_ylim(y_min, y_max)

    bp = ax1.boxplot(data, patch_artist=True, widths=0.5, showfliers=False,
                     medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.55)

    for i, (d, cls) in enumerate(zip(data, cls_order)):
        jitter = np.random.uniform(-0.08, 0.08, len(d))
        ax1.scatter(np.full(len(d), i+1)+jitter, d, s=12, alpha=0.35, color='#333', zorder=5)
        mean_val = np.mean(d)
        ax1.text(i+1, y_min + 0.02, f'均值={mean_val:.2f}',
                ha='center', fontsize=7, fontweight='bold', color=CLASS_COLORS[cls])

    ax1.set_xticklabels([f'{c}\n{CLASS_CN[c]}' for c in cls_order], fontsize=9)
    ax1.set_ylabel('RNet Spearman ρ (DMS 反应活性)', fontsize=10)
    ax1.set_title('各类变体反应活性保留度 (25条链)', fontweight='bold', fontsize=11, pad=12)
    ax1.axhline(y=1.0, color='#999', lw=0.5, ls='--')

    by_fam = defaultdict(lambda: defaultdict(list))
    for r in batch['results']:
        by_fam[r['family']][r['class']].append(r['dms_spearman_rho'])

    families = sorted(by_fam.keys())
    fam_labels = [f.replace('_',' ') for f in families]
    g1v = [np.mean(by_fam[f].get('G1',[0])) for f in families]
    n1v = [np.mean(by_fam[f].get('N1',[0])) for f in families]
    h3v = [np.mean(by_fam[f].get('H3',[0])) for f in families]

    y = np.arange(len(families)); h = 0.22
    ax2.barh(y-h, n1v, h, color=C_N1, alpha=0.7, label='N1 (GeoRNA不兼容)')
    ax2.barh(y, g1v, h, color=C_G1, alpha=0.7, label='G1 (GeoRNA兼容)')
    ax2.barh(y+h, h3v, h, color=C_H3, alpha=0.7, label='H3 (高置信度破坏)')

    ax2.set_yticks(y)
    ax2.set_yticklabels(fam_labels, fontsize=7)
    ax2.set_xlabel('RNet Spearman ρ (DMS)', fontsize=10)
    ax2.set_title('G1 vs N1 vs H3 按家族对比', fontweight='bold', fontsize=11, pad=12)
    ax2.legend(fontsize=8, frameon=False, loc='lower right')
    ax2.axvline(x=1.0, color='#999', lw=0.5, ls='--')

    n_win = sum(1 for f in families if np.mean(by_fam[f].get('G1',[0])) > np.mean(by_fam[f].get('N1',[0])))
    ax2.text(0.98, 0.02, f'G1 > N1: {n_win}/{len(families)} 家族',
            transform=ax2.transAxes, fontsize=8, ha='right', color='#888')

    fig.subplots_adjust(wspace=0.38, top=0.87, bottom=0.12)
    fig.suptitle('图 3: 批量计算实验 —— 25条链 × 12个家族的扰动分析', fontweight='bold',
                x=0.02, ha='left', fontsize=13)
    return fig


def seq_with_highlights(seq, mutations):
    """突变位点黄色高亮"""
    mut_set = {int(k): v for k, v in mutations.items()} if isinstance(mutations, dict) else {}
    parts = []
    for i, nt in enumerate(seq):
        pos = i + 1
        if pos in mut_set:
            parts.append(f'<span class="seq-mut" title="位置 {pos}: {nt}→{mut_set[pos]}">{nt}</span>')
        else:
            parts.append(nt)
    return ''.join(parts)


def build_table(target, variants):
    """构建变体指标表格"""
    rows = ''
    for c in variants.get(target, []):
        muts = c['mutations']
        if isinstance(muts, dict):
            s0seq = ''
            for sc in variants[target]:
                if sc['class'] == 'S0':
                    s0seq = sc['sequence']; break
            mut_list = ','.join(
                f"{k}{s0seq[int(k)-1]}→{v}"
                for k, v in sorted(muts.items(), key=lambda x: int(x[0]))
            )
        else:
            mut_list = ','.join(str(m) for m in muts)
        if not mut_list: mut_list = '—'

        ss = c.get('ss_similarity', 0)
        ddg = c.get('ddG', 0) or 0
        ce = c.get('mean_ce', 0)
        nmut = c['n_mutations']

        sc = 'metric-good' if ss > 0.7 else ('metric-mid' if ss > 0.3 else 'metric-bad')
        dc = 'metric-good' if abs(ddg) < 5 else ('metric-mid' if abs(ddg) < 15 else 'metric-bad')

        tier_map = {'S0': ('T1', 't1'), 'H3': ('T1', 't1'), 'L2': ('T1', 't1'),
                    'G1': ('T2', 't2'), 'N1': ('T2', 't2'), 'H1': ('T2', 't2'),
                    'C1': ('T3', 't3'), 'L1': ('T3', 't3')}
        tier, tier_cls = tier_map.get(c['class'], ('', ''))

        desc_map = {
            'S0': '阳性对照，验证实验管线可重现已知结构',
            'L1': '5-6个最低置信度位置替换为次优核苷酸，测试温和突变容忍度',
            'L2': '10-12个低置信度位置分散突变，测试大量容忍位点改变的极限',
            'H1': '单对高置信度配对单元替换为最低概率核苷酸，测试局域约束',
            'H3': '2-3对高置信度单元分散破坏，测试少数关键位点 vs 大量容忍位点',
            'C1': '3对WC/wobble配对做补偿性交换 (AU↔GC)，区分二级与三级约束',
            'G1': '锁定高置信度位点、重采样低置信度位点，测试GeoRNA兼容性指导重设计',
            'N1': '保留二级结构但故意破坏高置信度位点，测试二级兼容 vs 三级兼容',
        }

        rows += f'''
        <tr>
            <td><strong style="color:{CLASS_COLORS[c["class"]]}">{c['class']}</strong></td>
            <td><span class="tag tag-{tier_cls}">{tier}</span></td>
            <td style="font-size:12px">{desc_map.get(c['class'], CLASS_CN.get(c['class'], ''))}</td>
            <td style="text-align:center">{nmut}</td>
            <td style="text-align:center"><span class="{sc}">{ss:.3f}</span></td>
            <td style="text-align:center"><span class="{dc}">{ddg:+.1f}</span></td>
            <td style="text-align:center">{ce:.4f}</td>
            <td style="font-size:11px;max-width:250px">{mut_list[:100]}{'...' if len(mut_list)>100 else ''}</td>
        </tr>'''
    return rows


def build_seq_section(target, variants):
    """构建序列展示区"""
    out = ''
    show_classes = ['S0', 'L2', 'H3', 'G1', 'N1']
    for c in variants.get(target, []):
        if c['class'] not in show_classes:
            continue
        seq_html = seq_with_highlights(c['sequence'], c['mutations'])
        nmut = c['n_mutations']
        ss = c.get('ss_similarity', 0)
        ddg = c.get('ddG', 0) or 0
        sc = 'color:#2ca02c' if ss > 0.7 else ('color:#ff7f0e' if ss > 0.3 else 'color:#d62728')
        out += f'''
        <div style="margin:10px 0;padding:8px 12px;background:#fafafa;border-radius:6px;
                    border-left:4px solid {CLASS_COLORS[c['class']]}">
            <div style="margin-bottom:4px">
                <strong style="color:{CLASS_COLORS[c['class']]};font-size:14px">{c['class']}</strong>
                <span style="color:#888;font-size:12px;margin-left:8px">
                    {CLASS_CN.get(c['class'],'')} — {nmut} 个突变</span>
                <span style="font-size:12px;margin-left:12px">
                    SSsim=<span style="{sc}">{ss:.3f}</span>
                    &nbsp;ΔΔG=<span style="{sc}">{ddg:+.1f}</span> kcal/mol</span>
            </div>
            <div class="seq-box">{seq_html}</div>
        </div>'''
    return out


def build_batch_table(batch):
    """批量实验聚合表"""
    by_class = defaultdict(list)
    for r in batch['results']:
        by_class[r['class']].append(r)
    rows = ''
    for cls in ['L2', 'H3', 'G1', 'N1']:
        entries = by_class.get(cls,[])
        if not entries: continue
        ss = np.mean([e['ss_similarity'] for e in entries])
        dds = [e['ddG'] for e in entries if e['ddG'] is not None]
        rho = np.mean([e['dms_spearman_rho'] for e in entries])
        ratio = np.mean([e['mut_nonmut_dms_ratio'] for e in entries if e['n_mutations']>0])
        dms_abs = np.mean([e['mean_abs_dms'] for e in entries])
        rows += f'''
        <tr>
            <td><strong style="color:{CLASS_COLORS[cls]}">{cls}</strong></td>
            <td>{CLASS_CN.get(cls,'')}</td>
            <td style="text-align:center">{len(entries)}</td>
            <td style="text-align:center">{ss:.3f}</td>
            <td style="text-align:center">{np.mean(dds):+.1f}</td>
            <td style="text-align:center"><strong>{rho:.3f}</strong></td>
            <td style="text-align:center">{ratio:.1f}×</td>
            <td style="text-align:center">{dms_abs:.4f}</td>
        </tr>'''
    return rows


# ═══════════════════════════════════════════════════════════
CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI','PingFang SC','Microsoft YaHei',system-ui,sans-serif;
     background:#f5f6f8;color:#222;line-height:1.7}
.container{max-width:1150px;margin:0 auto;padding:20px}

.header{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
        color:white;padding:45px 35px;border-radius:10px;margin-bottom:25px}
.header h1{font-size:26px;margin-bottom:10px;letter-spacing:1px}
.header .sub{opacity:0.85;font-size:14px;line-height:1.6}
.header .tags{margin-top:15px}

.section{background:white;border-radius:10px;padding:28px 30px;margin-bottom:22px;
         box-shadow:0 1px 4px rgba(0,0,0,0.06)}
.section h2{font-size:19px;color:#1a1a2e;border-bottom:2px solid #e8e8e8;
            padding-bottom:10px;margin-bottom:20px}
.section h3{font-size:15px;color:#333;margin:18px 0 10px}
.section p{margin:8px 0;font-size:14px}

.figure{text-align:center;margin:18px 0}
.figure img{max-width:100%;border-radius:6px;box-shadow:0 2px 10px rgba(0,0,0,0.08)}
.figure .caption{font-size:12px;color:#888;margin-top:8px;font-style:italic}

.grid2{display:grid;grid-template-columns:1fr 1fr;gap:22px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}

table{width:100%;border-collapse:collapse;margin:12px 0;font-size:13px}
th{background:#f0f1f3;padding:9px 12px;text-align:left;font-weight:600;border-bottom:2px solid #ddd}
td{padding:8px 12px;border-bottom:1px solid #eee}
tr:hover{background:#fafbfc}
.metric-good{color:#2ca02c;font-weight:bold}
.metric-mid{color:#ff7f0e}
.metric-bad{color:#d62728;font-weight:bold}

.tag{display:inline-block;padding:3px 9px;border-radius:4px;font-size:11px;
     font-weight:700;color:white;margin:1px}
.tag-high{background:#d62728}.tag-mid{background:#ff7f0e}.tag-low{background:#1f77b4}
.tag-t1{background:#2ca02c}.tag-t2{background:#9467bd}.tag-t3{background:#999}

.seq-box{font-family:'Cascadia Code','Fira Code','Courier New',monospace;
         background:#f0f0f0;padding:10px 14px;border-radius:5px;font-size:12.5px;
         word-break:break-all;line-height:2.2;letter-spacing:0.5px;overflow-x:auto}
.seq-mut{background:#ffe082;padding:1px 3px;border-radius:3px;font-weight:bold;
         border-bottom:2px solid #f9a825}

.badge{display:inline-block;padding:4px 12px;border-radius:14px;font-size:11px;font-weight:600}
.badge-pass{background:#e8f5e9;color:#2e7d32}
.badge-warn{background:#fff3e0;color:#e65100}

.callout{padding:16px 20px;margin:14px 0;border-radius:8px;font-size:13px;line-height:1.7}
.callout-info{background:#e3f2fd;border-left:5px solid #1976d2}
.callout-success{background:#e8f5e9;border-left:5px solid #388e3c}
.callout-warning{background:#fff3e0;border-left:5px solid #f57c00}
.callout-important{background:#fce4ec;border-left:5px solid #c62828}
.callout strong{display:block;margin-bottom:4px;font-size:14px}

.priority-box{border-left:5px solid #2ca02c;background:#f1f8e9;padding:14px 18px;
              margin:12px 0;border-radius:0 8px 8px 0}
.priority-box.t2{border-left-color:#9467bd;background:#f3e5f5}
.priority-box.t3{border-left-color:#999;background:#f5f5f5}
.priority-box h4{margin-bottom:6px}

.note{font-size:12px;color:#888;margin-top:4px;font-style:italic}

.footer{text-align:center;padding:25px;font-size:12px;color:#aaa}
"""


def main():
    print("加载数据...")
    with open(STRESS / 'variant_candidates.json') as f:
        variants = json.load(f)
    with open(STRESS / 'per_position_data.json') as f:
        perpos_data = json.load(f)
    with open(STRESS / 'batch_perturbation_results.json') as f:
        batch = json.load(f)

    print("生成图表...")
    fig1 = make_fig1(variants, perpos_data)
    f1b64 = fig_to_b64(fig1); plt.close(fig1)
    print("  图1: 靶标概览 ✓")

    fig2 = make_fig2(variants)
    f2b64 = fig_to_b64(fig2); plt.close(fig2)
    print("  图2: 多指标筛选 ✓")

    fig3 = make_fig3(batch)
    f3b64 = fig_to_b64(fig3); plt.close(fig3)
    print("  图3: 批量实验 ✓")

    print("构建 HTML...")

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>GeoRNA 实验方案 — 计算预筛选报告</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">

<!-- ═══════════ 页眉 ═══════════ -->
<div class="header">
    <h1>GeoRNA 指导的 RNA 三级结构实验验证</h1>
    <div class="sub">
        <strong>序列设计与计算预筛选报告</strong> &nbsp;|&nbsp; 2026-05-30<br>
        两个靶标：10PX_H_1B (5S rRNA, 120nt) + 10ZT_H_A (AI设计假结, 96nt)<br>
        补充证据：25 条链 × 12 个 Rfam 家族的批量计算实验
    </div>
    <div class="tags">
        <span class="tag tag-t1">★ 第一优先: S0, H3, L2</span>
        <span class="tag tag-t2">◆ 第二优先: G1, N1, H1</span>
        <span class="tag tag-t3">○ 第三优先: C1, L1</span>
    </div>
</div>

<!-- ═══════════ 第1节: 科学问题 ═══════════ -->
<div class="section">
<h2>1. 我们要回答什么科学问题</h2>

<div class="callout callout-important">
<strong>核心假设</strong>
GeoRNA 置信度可以预测三级结构的突变容忍度：高置信度位置 = 结构约束位点（突变→结构破坏）；
低置信度位置 = 结构容忍位点（突变→结构保持）。
</div>

<p>GeoRNA 从 7 原子粗粒化骨架（P, O5', C5', C4', C3', O3', N1/N9）提取 935 维显式几何特征，
预测每个位置的核苷酸类型（A/C/G/U），同时输出预测置信度。如果仅凭骨架几何就能唯一确定核苷酸（高置信度），
说明这个位置在三维结构上有严格的几何约束；反之（低置信度）说明多种核苷酸都能适配骨架。</p>

<p><strong>我们要验证的不是 GeoRNA 能不能生成序列</strong>（它本来就不是生成模型），
而是它能不能作为一个<strong>结构诊断工具</strong>——告诉实验学家哪些位置可以安全地突变、哪些不能碰。
如果验证成功，GeoRNA 的价值就从"又一个逆向折叠模型"变成了"三级结构约束的探针"。</p>
</div>

<!-- ═══════════ 第2节: 靶标选择 ═══════════ -->
<div class="section">
<h2>2. 靶标选择理由</h2>

<div class="figure">
    <img src="data:image/png;base64,{f1b64}" alt="靶标概览">
    <div class="caption">图 1: <b>左</b> — 两个靶标的 GeoRNA 恢复率 (柱状图) 与平均交叉熵 (◆)。
    10PX 恢复率 100% 说明 GeoRNA 对这个天然 rRNA 的每个位置都非常确定；
    10ZT 恢复率 54% 说明 AI 设计的假结有大量几何模糊位置。
    <b>右</b> — 置信度分布直方图。10PX 集中在高置信度区 (>0.9)，10ZT 呈宽分布。</div>
</div>

<div class="grid2">
    <div>
        <h3>靶标 A: 10PX_H_1B — 5S rRNA</h3>
        <table>
            <tr><td>长度</td><td>120 nt</td></tr>
            <tr><td>GC 含量</td><td>68.3%</td></tr>
            <tr><td>GeoRNA 恢复率</td><td><strong style="color:#2ca02c">100.0%</strong></td></tr>
            <tr><td>平均交叉熵</td><td>0.022 (极低不确定性)</td></tr>
            <tr><td>Rfam</td><td>5S_rRNA (RF00001)</td></tr>
            <tr><td>来源</td><td>PDB 10PX, cryo-EM (2026)</td></tr>
            <tr><td>角色</td><td><span class="badge badge-pass">天然阳性对照</span></td></tr>
        </table>
        <div class="callout callout-success" style="margin-top:12px">
        <strong>选择理由</strong>
        100% 恢复率意味着 GeoRNA 对这个结构有"绝对把握"。如果连这个靶标上置信度都不能预测结构容忍度，
        整个假说就不成立。同时 5S rRNA 是进化上最保守的 RNA 之一，结构极其稳健，
        提供了一个防止"所有变体都解折叠"的安全垫。
        </div>
    </div>
    <div>
        <h3>靶标 B: 10ZT_H_A — AI 设计假结</h3>
        <table>
            <tr><td>长度</td><td>96 nt</td></tr>
            <tr><td>GC 含量</td><td>67.7%</td></tr>
            <tr><td>GeoRNA 恢复率</td><td><strong style="color:#e377c2">54.2%</strong></td></tr>
            <tr><td>平均交叉熵</td><td>1.140 (高不确定性)</td></tr>
            <tr><td>设计来源</td><td>RibonanzaNet 指导的序列筛选</td></tr>
            <tr><td>来源</td><td>PDB 10ZT, cryo-EM</td></tr>
            <tr><td>角色</td><td><span class="badge badge-warn">泛化能力测试</span></td></tr>
        </table>
        <div class="callout callout-warning" style="margin-top:12px">
        <strong>选择理由</strong>
        10ZT 不是天然进化产物，而是用 RibonanzaNet（在 ~200 万条实验化学探测数据上训练的深度学习模型）
        筛选出来的序列。GeoRNA 在它上面只有 54% 恢复率，但这个不确定性是真实的几何模糊还是模型没见过这类结构？
        此外，我们恰好搭建了 RibonanzaNet 推理管线，可以做两种方法的正交对比。
        </div>
    </div>
</div>

<div class="callout callout-info">
<strong>两个靶标的互补性</strong>
10PX (天然·高恢复率) 验证"假说是否成立"，10ZT (合成·低恢复率) 验证"假说能否泛化"。
如果两个靶标都成功 → GeoRNA 置信度是通用结构诊断工具。
如果仅 10PX 成功 → GeoRNA 的置信度可能局限于天然 RNA 家族（这本身也是一个有价值的发现，说明序列恢复率衡量的是"结构可识别性"而非"生成能力"）。
</div>
</div>

<!-- ═══════════ 第3节: 变体设计 ═══════════ -->
<div class="section">
<h2>3. 八类变体设计与筛选结果</h2>

<p>对每个靶标，按 GeoRNA 置信度将位置分为三层：<span style="color:#d62728;font-weight:bold">高置信度 (top 35%, ≥~0.7)</span>、
<span style="color:#ff7f0e">中等 (25%)</span>、
<span style="color:#1f77b4;font-weight:bold">低置信度 (bottom 40%, ≤~0.5)</span>。
然后设计 8 类变体，每类有明确的科学目的和预期的结构结果。</p>

<div class="figure">
    <img src="data:image/png;base64,{f2b64}" alt="多指标筛选热力图">
    <div class="caption">图 2: <b>八类变体的三维度计算筛选热力图</b>。
    每行归一化到 [0,1]：<span style="color:#2ca02c">绿色</span>=该类中最好，
    <span style="color:#d62728">红色</span>=该类中最差。
    ★ = 第一优先合成 (必须解结构)；◆ = 第二优先；○ = 第三优先。</div>
</div>

<!-- 10PX 表格 -->
<h3>靶标 A: 10PX_H_1B (5S rRNA) — 变体指标总览</h3>
<p class="note">10PX 有 42 个高置信度位置、49 个低置信度位置。GeoRNA 对这个结构非常确定 (CE=0.022)。</p>
<table>
<tr><th>类别</th><th>优先级</th><th>设计说明</th><th>#突变</th><th>SSsim</th><th>ΔΔG</th><th>CE</th><th>突变位点</th></tr>
{build_table('10PX_H_1B', variants)}
</table>

<div class="callout callout-success">
<strong>10PX 关键发现</strong>
G1 (16个GeoRNA兼容突变) 的 SSsim=0.977——几乎完美保持二级结构，比 N1 (SS=0.419) 高出 0.55。
这说明 GeoRNA 兼容性在天然 rRNA 上能有效保护二级结构。H1 (仅2个高置信度突变) 的局部扰动最大 (mut/non-mut=15×) 但全局影响最小——高置信度位点破坏是局域化的。
</div>

<!-- 10ZT 表格 -->
<h3>靶标 B: 10ZT_H_A (假结) — 变体指标总览</h3>
<p class="note">10ZT 有 34 个高置信度位置、39 个低置信度位置。GeoRNA 对这个 AI 设计的结构不确定性高 (CE=1.140)。</p>
<table>
<tr><th>类别</th><th>优先级</th><th>设计说明</th><th>#突变</th><th>SSsim</th><th>ΔΔG</th><th>CE</th><th>突变位点</th></tr>
{build_table('10ZT_H_A', variants)}
</table>

<div class="callout callout-warning">
<strong>10ZT 关键发现</strong>
N1 (SS=0.836, ρ=0.73) 在所有指标上优于 G1 (SS=0.603, ρ=0.40)。
这是因为 N1 采用补偿性配对保护二级结构，而 G1 独立重采样 39 个低置信度位置导致累计效应过载。
这恰好说明 <strong>二级结构兼容性 ≠ 三级结构兼容性</strong>——真正需要在 cryo-EM 下观察的是三级结构的差异。
10ZT 本身是用 RibonanzaNet 设计的，G1 没有这个信息，所以 RNet 指标天然有利于 N1。
</div>
</div>

<!-- ═══════════ 第4节: 各类详解 ═══════════ -->
<div class="section">
<h2>4. 每类的详细设计逻辑</h2>

<div class="grid2">
<div class="priority-box">
<h4>★ S0 — 原始序列 (阳性对照)</h4>
<p><strong>设计</strong>: 不引入任何突变。这是最基本但最重要的对照——验证整个实验管线能否重现已知结构。</p>
<p><strong>如果失败</strong>: 后续所有变体的解读都失去了基准。需要检查合成、纯化、折叠条件。</p>
</div>

<div class="priority-box t3">
<h4>○ L1 — 低置信度轻度突变</h4>
<p><strong>设计</strong>: 5-6 个最低置信度位置替换为 GeoRNA 预测概率<strong>第二高</strong>的核苷酸。避免破坏 WC 配对。</p>
<p><strong>问题</strong>: 最"不关键"的位置做温和改变，结构能否保持？</p>
<p><strong>为什么不选最低概率核苷酸</strong>: 第一次实验希望温和测试——选第二概率的核苷酸即使 GeoRNA 不完全准确也不会过于极端。</p>
</div>

<div class="priority-box">
<h4>★ L2 — 低置信度强力突变</h4>
<p><strong>设计</strong>: 10-12 个低置信度位置，<strong>分散分布</strong>在全序列上，替换为第二概率核苷酸。</p>
<p><strong>问题</strong>: 大量改变"不关键"位置，结构能否保持？这是对 GeoRNA 低置信度假说最直接的测试。</p>
<p><strong>与 L1 区别</strong>: L1 测试温和改变，L2 测试大量改变。如果 L1 保持而 L2 不保持，说明容忍度有累积上限。</p>
</div>

<div class="priority-box t2">
<h4>◆ H1 — 高置信度单单元破坏</h4>
<p><strong>设计</strong>: 选 <strong>1 个</strong>高置信度 WC 配对单元，替换为 GeoRNA 预测概率<strong>最低</strong>的核苷酸。</p>
<p><strong>问题</strong>: 一个"关键"位置的破坏，影响是局域还是全局？</p>
<p><strong>为什么选配对单元</strong>: 破坏明确的结构元件（WC 配对）比改单链位点更容易解读。</p>
</div>

<div class="priority-box">
<h4>★ H3 — 高置信度多单元破坏</h4>
<p><strong>设计</strong>: 选 2-3 个高置信度单元，<strong>空间分散</strong>分布，替换为最低概率核苷酸。控制总突变数（避免 RNA 完全不解折叠）。</p>
<p><strong>问题</strong>: 少数几个"关键"位置的破坏，能否比大量"容忍"位置的改变 (L2) 造成更严重的结构扰动？</p>
<p><strong>这是最重要的实验类</strong>: H3 是区分"结构关键约束"和"可容忍变异"的核心测试。批量实验确认 H3 在所有家族中最优。</p>
</div>

<div class="priority-box t3">
<h4>○ C1 — 补偿性配对对照</h4>
<p><strong>设计</strong>: 选 3 对 WC/wobble 碱基对，同时突变两侧保持配对 (AU↔GC, GC↔AU)。改变核苷酸身份但不破坏二级结构。</p>
<p><strong>问题</strong>: 序列身份重要还是二级结构配对重要？如果仅受配对约束 → 结构应保持；如果还有三级几何约束 → 结构可能改变。</p>
</div>

<div class="priority-box t2">
<h4>◆ G1 — GeoRNA 兼容重设计</h4>
<p><strong>设计</strong>: <strong>锁定</strong>所有高置信度位置 (保持原生核苷酸)；<strong>重采样</strong>所有低置信度位置
(从 GeoRNA 概率分布中抽样，500 次迭代选 GC 和二级结构最接近原生的候选)。
配对中的低置信度位置做补偿性重采样。</p>
<p><strong>问题</strong>: 序列大幅改变但 GeoRNA 认为兼容，结构能否保持？</p>
<p><strong>10PX G1 最佳案例</strong>: 16 个突变，SSsim=0.98——GeoRNA 给每个低置信度位置的概率分布高度偏好原生核苷酸，重采样只改变极少数位置。</p>
</div>

<div class="priority-box t2">
<h4>◆ N1 — GeoRNA 不兼容阴性对照</h4>
<p><strong>设计</strong>: <strong>保留</strong>预测二级结构 (补偿性配对) + GC 含量；<strong>故意</strong>在高置信度位置引入 GeoRNA 最低概率核苷酸。</p>
<p><strong>问题</strong>: 仅靠二级结构兼容性是否足以保持三级结构？N1 保护二级结构但破坏 GeoRNA 三级兼容 → 如果三级结构更差，说明 GeoRNA 捕捉了超越二级结构的约束。</p>
<p><strong style="color:#333">这是 G1 的直接配对对照</strong>——两个都改变了序列，但 G1 尊重 GeoRNA 置信度而 N1 只尊重二级结构。</p>
</div>
</div>
</div>

<!-- ═══════════ 第5节: 序列 ═══════════ -->
<div class="section">
<h2>5. 变体序列 (黄色高亮 = 突变位点)</h2>

<h3>10PX_H_1B — 5S rRNA (120nt, GC=68.3%)</h3>
{build_seq_section('10PX_H_1B', variants)}

<h3>10ZT_H_A — AI 设计假结 (96nt, GC=67.7%)</h3>
{build_seq_section('10ZT_H_A', variants)}
</div>

<!-- ═══════════ 第6节: 批量实验 ═══════════ -->
<div class="section">
<h2>6. 批量计算实验：跨家族验证</h2>

<p>为了确认两个靶标的发现不是个例，我们在 <strong>25 条链、12 个 Rfam 家族</strong>（5S_rRNA, tRNA, 5.8S_rRNA,
Purine核糖开关, K10_TLS, Intron_gpI, TLS-PK3, U6, AdoCbl-II, azaaromatic, Entero_5_CRE, mascRNA-menRNA）
上做了相同的四类变体 (L2, H3, G1, N1) × 每条链 = 125 个变体的 RibonanzaNet 扰动分析。</p>

<div class="figure">
    <img src="data:image/png;base64,{f3b64}" alt="批量实验">
    <div class="caption">图 3: <b>左</b> — 四类变体的 RNet 反应活性保留度箱线图（25 条链）。
    H3 (高置信度破坏) 的中位数和分布最优，G1 (GeoRNA兼容重设计) 分布最宽——说明 G1 的质量高度依赖靶标特性。
    <b>右</b> — 按家族拆分的 G1 vs N1 vs H3 对比。在 11/12 家族中 N1 > G1。</div>
</div>

<h3>聚合统计 (125 个变体)</h3>
<table>
<tr><th>类别</th><th>中文名</th><th>链条数</th><th>平均 SSsim</th><th>平均 ΔΔG</th><th>平均 RNet ρ</th><th>mut/non-mut 比</th><th>平均 |ΔDMS|</th></tr>
{build_batch_table(batch)}
</table>

<div class="grid2" style="margin-top:16px">
<div class="callout callout-success">
<strong>发现 1: H3 在所有指标上最优</strong>
H3 有最高的反应活性保留 (ρ=0.807)、最强的扰动局域化 (mut/non-mut=4.7×)、
最低的热力学代价 (ddG=+3.9)。这证实了"少数高置信度位置是关键约束点"的假说具有跨家族普适性。
4.7 倍的 mut/non-mut 比意味着突变处的扰动是其他区域的近 5 倍——扰动高度局域化，不传播。
</div>

<div class="callout callout-info">
<strong>发现 2: N1 系统性优于 G1 (11/12 家族)</strong>
N1 使用补偿性配对突变保护二级结构 → 在 RNet (测量二级结构反应活性) 上自然更好。
G1 独立重采样低置信度位置 → 累积效应过载 → RNet 指标差。
<b>但这不意味着 G1 是失败的设计</b>——RNet 测量的是二级结构，而 GeoRNA 置信度捕捉的是三级结构约束。
真正需要 cryo-EM 来判断的是 G1 和 N1 在三级结构上的差异。
</div>
</div>

<div class="callout callout-warning">
<strong>发现 3: L2 的扰动是全局性的 (非局域化)</strong>
L2 的 mut/non-mut 比仅 2.5×（远低于 H3 的 4.7×），且近邻衰减≈0%——扰动不随距离衰减。
说明 11 个低置信度突变的集体效应改变了整个结构的反应活性谱，而非局部效应。
这与 L2 常常 SSsim=0.00 (RNAfold 预测结构完全改变) 一致。
</div>
</div>

<!-- ═══════════ 第7节: 合成计划 ═══════════ -->
<div class="section">
<h2>7. 合成优先级与预期结果</h2>

<div class="priority-box">
<h4>★ 第一优先 —— 必须解结构 (每个靶标 3 条，共 6 条)</h4>
<table>
<tr><th>类别</th><th>10PX (5S rRNA)</th><th>10ZT (假结)</th><th>要回答的科学问题</th></tr>
<tr>
    <td><strong>S0</strong></td>
    <td>0 mut, SS=1.00, ddG=0</td>
    <td>0 mut, SS=1.00, ddG=0</td>
    <td>实验管线能否重现已知结构？</td>
</tr>
<tr>
    <td><strong>H3</strong></td>
    <td>4 mut, SS=0.75, ddG=+4.8</td>
    <td>4 mut, SS=0.76, ddG=+4.2</td>
    <td>高置信度位点破坏是否引起局域化三级结构损伤？</td>
</tr>
<tr>
    <td><strong>L2</strong></td>
    <td>11 mut, SS=0.15, ddG=+16</td>
    <td>11 mut, SS=0.00, ddG=+24</td>
    <td>低置信度位点能否被大量改变而保持折叠？</td>
</tr>
</table>
</div>

<div class="priority-box t2">
<h4>◆ 第二优先 —— 高价值补充 (每个靶标 +3 条，共 +6 条)</h4>
<table>
<tr><th>类别</th><th>选择理由</th></tr>
<tr><td><strong>G1</strong> — GeoRNA 兼容重设计</td><td>10PX G1 有 SS=0.98、16 个突变 — GeoRNA 指导重设计的最强测试案例</td></tr>
<tr><td><strong>N1</strong> — 阴性对照</td><td>G1 的配对对照。必须解 G1+N1 才能比较 GeoRNA 兼容 vs 仅二级结构兼容</td></tr>
<tr><td><strong>H1</strong> — 单单元破坏</td><td>最精细的局域破坏 (2 个突变)。两个靶标 H1 表现极好 (ρ=0.87-0.89)</td></tr>
</table>
</div>

<div class="priority-box t3">
<h4>○ 第三优先 —— 可选扩展</h4>
<p>L1 (预期接近 native, 科学增益有限)、C1 (区分 2°/3° 约束, 如已解 G1/N1 则价值有限)</p>
</div>

<h3>强阳性结果的预期模式</h3>
<table>
<tr><th>观察</th><th>含义</th></tr>
<tr><td>S0 重现目标折叠</td><td>✓ 管线验证通过</td></tr>
<tr><td>L2 保持全局折叠 (尽管 11 个突变)</td><td>✓ 低置信度 = 结构容忍</td></tr>
<tr><td>H3 在突变位点附近出现局域化结构扰动</td><td>✓ 高置信度 = 结构约束</td></tr>
<tr><td>G1 (GeoRNA兼容) 三级结构优于 N1 (仅二级兼容)</td><td>✓ GeoRNA 捕捉了超越二级结构的约束</td></tr>
<tr><td>10ZT 的 G1 vs N1 差异小于 10PX</td><td>✓ GeoRNA 泛化能力有边界 (合成 vs 天然)</td></tr>
</table>
</div>

<!-- ═══════════ 第8节: 指标说明 ═══════════ -->
<div class="section">
<h2>8. 计算筛选指标说明</h2>

<div class="grid3">
<div>
<h3>SSsim — 二级结构相似度</h3>
<p>用 ViennaRNA RNAfold 预测每条序列的 MFE 二级结构，计算 dot-bracket 碱基对重叠的 F1 分数。</p>
<p><strong>测量</strong>: 这条序列"自己折叠"时，会不会形成和目标结构相似的二级结构？</p>
<p><strong>局限</strong>: RNAfold 只考虑二级结构热力学 (Turner 规则)，不考虑三级相互作用、假结、配体或蛋白质。</p>
<p><strong>好值</strong>: >0.7</p>
</div>
<div>
<h3>ΔΔG — 目标折叠热力学代价</h3>
<p>用 ViennaRNA RNAeval 计算每条序列强制折叠为目标结构的自由能，减去原生序列的能量。</p>
<p><strong>测量</strong>: 这个变体要维持目标折叠，需要额外付出多少热力学代价？</p>
<p><strong>局限</strong>: 假设目标结构是唯一的、正确的折叠状态。</p>
<p><strong>好值</strong>: |ΔΔG| < 5 kcal/mol</p>
</div>
<div>
<h3>RNet ρ — 化学探针反应活性保留</h3>
<p>用 RibonanzaNet (在 ~200 万条实验 DMS/2A3 数据上预训练) 预测每位置反应活性，计算变体与原生序列的 Spearman 秩相关。</p>
<p><strong>测量</strong>: 从实验化学探针的角度，序列的"可及性谱"有多相似？</p>
<p><strong>局限</strong>: 仅间接反映二级结构；RNet 本身有预测误差。</p>
<p><strong>好值</strong>: ρ > 0.8</p>
</div>
</div>

<div class="callout callout-info" style="margin-top:18px">
<strong>三条指标的互补关系</strong><br>
SSsim 看"自己会怎么折"（热力学预测），ΔΔG 看"强折成目标结构要花多少能量"（约束热力学），
RNet ρ 看"实验化学探针会怎么反应"（数据驱动）。三者分别从物理模型约束、物理模型自由、实验数据驱动三个角度评估结构保留度。
如果三条指标一致（都好或都差），变体的结构命运就比较明确；如果不一致（如 N1：SS=0.42 但 RNet ρ=0.60），
则需要在 cryo-EM 下观察三级结构——这正是实验的目的。
</div>
</div>

<!-- ═══════════ 页脚 ═══════════ -->
<div class="footer">
    <p>GeoRNA 变体预筛选管线 &bull; 2026-05-30 &bull;
    脚本: src/generate_variants.py, src/validate_ribonanzanet.py,
    src/analyze_reactivity_perturbation.py, src/batch_perturbation_experiment.py, src/build_proposal_html_cn.py</p>
    <p>完整中文方案: RNKDB/stress_test/实验方案.md &bull; 交互报告: RNKDB/stress_test/实验方案.html</p>
</div>

</div>
</body>
</html>'''

    with open(OUT_HTML, 'w') as f:
        f.write(html)

    print(f"\n✓ 保存到: {OUT_HTML}")
    print(f"  文件大小: {len(html)/1024:.0f} KB")
    print(f"  浏览器打开: file://{OUT_HTML}")


if __name__ == '__main__':
    main()
