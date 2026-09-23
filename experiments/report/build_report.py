"""Generates docs/EXPERIMENT_SUMMARY.html (and the published Team-account artifact).
Inputs next to this file: s4d_runs.json / s4d_stats.json / s4d_compact.json
(refreshed from output/scaffold4d/* by the collection snippets recorded in
the 2026-09-20..22 session) and an img/ folder of thumbnails, full-size
images and MP4 animations. Non-Scaffold4D rows are hard-coded from their
RESULTS.md files. Run: python experiments/report/build_report.py -> index.html."""
import json, os, html, glob, datetime
D = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(D, "img")
runs = {r["run"]: r for r in json.load(open(os.path.join(D, "s4d_runs.json")))}
stats = json.load(open(os.path.join(D, "s4d_stats.json")))
compact = json.load(open(os.path.join(D, "s4d_compact.json")))

def count_cell(key):
    s = stats.get(key)
    if not s:
        return "—"
    nf = s.get("frames", 90)
    mean = (f'<br><span class="sub">有用/查询 {nf} 帧均值 <b>{s["useful_query_mean"]:,}</b> ({s["useful_query_min"]:,}–{s["useful_query_max"]:,})</span>'
            if "useful_query_mean" in s else "")
    return (f'<b>{s["anchors"]:,}</b> 锚点 × {s["children"]}<br>= {s["total"]:,} 高斯'
            f'<br><span class="sub">活跃/查询 {s["active"]:,}</span><br><span class="sub">有用/查询 t=0.5 {s["useful_query"]:,}</span>{mean}'
            f'<br><span class="sub">有用总数 {s["useful_total"]:,} ({s["useful_total"] / s["total"]:.0%})</span>'
            + (f'<br><span class="sub">{nf} 帧从未选入 {s["never_selected"]:,} ({s["never_selected"] / s["total"]:.0%}),其中“有用” {s["useful_never_selected"]:,}</span>'
               f'<br><span class="sub"><b>任一时刻起作用 {s["useful_total"] - s["useful_never_selected"]:,} ({(s["useful_total"] - s["useful_never_selected"]) / s["total"]:.0%})</b>;全程无用 {s["total"] - s["useful_total"] + s["useful_never_selected"]:,} ({(s["total"] - s["useful_total"] + s["useful_never_selected"]) / s["total"]:.0%})</span>' if "never_selected" in s else ""))
thumbs = {os.path.basename(p) for p in glob.glob(os.path.join(IMG, "*.jpg"))}

def viz(key):
    items = []
    for tag, label in (("curve", "曲线"), ("slice", "切片")):
        f = f"{key}__{tag}.jpg"
        full = f"{key}__{tag}_full.jpg"
        if f in thumbs:
            big = full if os.path.exists(os.path.join(IMG, full)) else f
            items.append(f'<button class="viz" type="button" data-full="img/{big}" data-cap="{html.escape(key)} · {label}"><img src="img/{f}" alt="{label}"><span>{label}</span></button>')
    v = f"{key}__dvr.mp4"
    if os.path.exists(os.path.join(IMG, v)):
        poster = f"{key}__slice.jpg" if f"{key}__slice.jpg" in thumbs else f"{key}__curve.jpg"
        items.append(f'<button class="viz" type="button" data-video="img/{v}" data-cap="{html.escape(key)} · 90 帧体渲染:GT | 预测 | 体素误差 | 渲染图误差"><img src="img/{poster}" alt="动图"><span>▶ 动图</span></button>')
    return '<div class="vizrow">' + "".join(items) + "</div>" if items else '<span class="none">未生成</span>'

PILL = {"ref": ("ref", "参照"), "final": ("final", "当前结论"), "old": ("old", "已取代"), "bad": ("bad", "未修复核 · 作废"),
        "run": ("run", "进行中"), "neg": ("old", "负面记录")}

def pill(kind, text=None):
    cls, default = PILL[kind]
    return f'<span class="pill {cls}">{html.escape(text or default)}</span>'

KERNEL = {"fixed": ("k-ok", "修复核 · 自检"), "fixed-fp": ("k-ok", "修复核 · 指纹判定"), "fixed-script": ("k-ok", "修复核 · 脚本用 .venv/bin/python"),
          "fixed-likely": ("k-warn", "很可能修复核 · 未记录"), "unfixed": ("k-bad", "未修复核"), "unfixed-by-design": ("k-neutral", "未修复核(有意,原始参照)"), "na": ("k-neutral", "—")}

def kernel_tag(kind):
    cls, text = KERNEL[kind]
    return f'<div class="ktag {cls}">{text}</div>'

def row(name, what, path, scale, key, status, psnr, minmax="", best=False, counts=None, kernel=None):
    extra = f'<td class="counts">{counts}</td>' if counts is not None else ""
    ktag = kernel_tag(kernel) if kernel else ""
    return f'''<tr{' class="best"' if best else ''}>
<td class="name">{name}</td>
<td class="what">{what}<div class="path">{html.escape(path)}</div></td>
<td class="scale">{scale}</td>{extra}
<td class="vizcell">{viz(key)}</td>
<td>{status}{ktag}</td>
<td class="num">{psnr}<div class="sub">{minmax}</div></td></tr>'''

def s4d(name, run, what, status, best=False, note=""):
    r = runs[run]
    key = ("scale__" + run.split("/")[1]) if run.startswith("scale/") else "scaffold4d__" + run.replace("/", "__")
    if run.startswith(("decay/", "long/")):
        key = "scaffold4d__" + run.replace("/", "__")
    scale = f'<b>{r["seconds"]} s</b> · {r["steps"]:,} 步<br>{r["mb"]:.2f} MB'
    if r.get("frames", 90) != 90:
        scale += f'<br><span class="sub"><b>只训 {r["frames"]} 帧</b>(PSNR 为这 {r["frames"]} 帧均值)</span>'
    c = compact.get(run)
    if c:
        scale += f'<br><span class="sub">压缩后 <b>{c["mb_after"]:.1f} MB</b> · {c["psnr_after"]:.2f} dB ({c["psnr_delta"]:+.3f})</span>'
    kernel = ("unfixed" if run.startswith("ablation/") else "fixed-fp" if run == "vorts90_native_recipe"
              else "fixed-likely" if run == "vorts90_matched_comparison" else "fixed")
    return row(name, what + (f'<div class="note">{note}</div>' if note else ""), "output/scaffold4d/" + run + "/", scale, key, status,
               f'{r["psnr"]:.2f} dB', f'{r["pmin"]:.2f} – {r["pmax"]:.2f}', best, counts=count_cell(run), kernel=kernel)

# ---------- chart data ----------
points = []
def pt(series, mb, psnr, label, faded=False):
    points.append(dict(s=series, x=mb, y=psnr, l=label, f=faded))
pt("static", 93.77, 41.22, "逐帧 ESVR,未修复核")
pt("static", 90.0, 39.38, "逐帧 ESVR“修复核”对照(实为未修复核 + 位置冻结,作废)", faded=True)
pt("static", 301.1, 38.86, "逐帧简化 ESVR")
for nm, mb1, ps in (("S1b 21k/22k步", 0.93, 41.72), ("S2b 30k/22k步", 1.31, 42.29), ("S3b 16k/67k步", 0.71, 44.73), ("S4b 63k/67k步", 2.78, 47.05), ("S5b 139k/67k步", 6.13, 49.67)):
    pt("static", mb1 * 90, ps, f"静态逐帧外推 {nm}(单帧 × 90)")
pt("deform", 2.83, 22.35, "Deformable 历史基线")
pt("deform", 3.19, 25.96, "Deformable 配置对齐")
pt("deform", 3.19, 24.21, "Deformable 单变量 LR")
pt("deform", 3.19, 28.90, "Deformable 原生 LR 结构")
pt("esvr4d", 2.29, 28.77, "ESVR4D 合并 LR 组")
pt("esvr4d", 2.29, 25.09, "ESVR4D 原生 LR,30k")
pt("esvr4d", 2.29, 25.68, "ESVR4D 原生 LR,40k")
pt("s4d_old", 1.60, 25.03, "Scaffold4D 配置对齐版")
pt("s4d_old", 13.79, 24.35, "Scaffold4D 原生配方版")
for run, r in runs.items():
    if run.startswith("ablation/"):
        pt("s4d_bad", r["mb"], r["psnr"], "ablation/" + run.split("/")[1] + "(未修复核)", faded=True)
    elif run.startswith("fixed_kernel/"):
        pt("s4d_fix", r["mb"], r["psnr"], "fixed_kernel/" + run.split("/")[1])
    elif run.startswith(("scale/", "decay/", "long/", "fifty/", "next/")) and not run.startswith("next/E_group"):
        partial = r.get("frames", 90) != 90
        label = run + (f"(只训 {r['frames']} 帧,不与 90 帧数字可比)" if partial else "")
        pt("s4d_fix", r["mb"], r["psnr"], label, faded=partial)
        if run in compact:
            pt("s4d_compact", compact[run]["mb_after"], compact[run]["psnr_after"], label + " · 压缩后", faded=partial)

E_GROUPS = sorted(k for k in runs if k.startswith("next/E_group"))
if len(E_GROUPS) == 9:
    e_mb = sum(runs[k]["mb"] for k in E_GROUPS)
    e_psnr = sum(runs[k]["psnr"] for k in E_GROUPS) / 9   # equal-sized groups: mean of means = 90-frame mean
    pt("s4d_fix", round(e_mb, 1), round(e_psnr, 3), "E1–E9 分组合计(9 × 10 帧,同等总计算量)")
    if all(k in compact for k in E_GROUPS):
        pt("s4d_compact", round(sum(compact[k]["mb_after"] for k in E_GROUPS), 1), round(sum(compact[k]["psnr_after"] for k in E_GROUPS) / 9, 3), "E1–E9 分组合计 · 压缩后")

# ---------- page ----------
today = datetime.date.today().isoformat()
n_s4d = len(runs)
best_key, best_fix = max(((k, r) for k, r in runs.items() if k.startswith(("fixed_kernel/", "scale/", "decay/", "long/", "fifty/", "next/")) and r.get("frames", 90) == 90), key=lambda kv: kv[1]["psnr"])

parts = []
parts.append(f'''<title>ESVR 时变扩展实验记录</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@500;600;700&family=Public+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&family=Noto+Serif+SC:wght@600&display=swap">
<style>
:root{{--bg:#f3f6f8;--surface:#ffffff;--ink:#17222c;--ink2:#4b5967;--ink3:#7a8794;--line:#d8e0e6;--line2:#bcc7d1;
--accent:#1d6b86;--accent-soft:#e0eef3;--good:#2b7a4b;--good-soft:#e1f1e6;--warn:#9a6414;--warn-soft:#f6ecd8;--bad:#a13a3a;--bad-soft:#f5e1e1;--info:#3f5f9b;--info-soft:#e3e9f5;--mono-bg:#eaf0f4;
--c-static:#7a8794;--c-deform:#b5651d;--c-esvr4d:#7b4fa3;--c-s4d:#1d6b86;}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#12181d;--surface:#192129;--ink:#e6ecf1;--ink2:#aeb9c4;--ink3:#7a8794;--line:#28323c;--line2:#3a4652;
--accent:#5fb0cc;--accent-soft:#1a3440;--good:#5fbd86;--good-soft:#173324;--warn:#d9a24a;--warn-soft:#3a2d14;--bad:#e07a7a;--bad-soft:#3d1f1f;--info:#8fb0e6;--info-soft:#1e2a40;--mono-bg:#1f2a33;
--c-static:#93a0ad;--c-deform:#d98d4a;--c-esvr4d:#b08ad6;--c-s4d:#5fb0cc;}}}}
:root[data-theme="dark"]{{--bg:#12181d;--surface:#192129;--ink:#e6ecf1;--ink2:#aeb9c4;--ink3:#7a8794;--line:#28323c;--line2:#3a4652;
--accent:#5fb0cc;--accent-soft:#1a3440;--good:#5fbd86;--good-soft:#173324;--warn:#d9a24a;--warn-soft:#3a2d14;--bad:#e07a7a;--bad-soft:#3d1f1f;--info:#8fb0e6;--info-soft:#1e2a40;--mono-bg:#1f2a33;
--c-static:#93a0ad;--c-deform:#d98d4a;--c-esvr4d:#b08ad6;--c-s4d:#5fb0cc;}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--ink);font-family:"Public Sans","PingFang SC","Microsoft YaHei",system-ui,sans-serif;font-size:14.5px;line-height:1.6;margin:0;padding:48px 20px 96px}}
.page{{max-width:980px;margin:0 auto}}
h1,h2,h3{{font-family:"Spectral","Noto Serif SC",Georgia,serif;font-weight:600;text-wrap:balance;margin:0}}
code,.mono,.path,.num,.stat b{{font-family:"JetBrains Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}}
header{{border-bottom:1px solid var(--line2);padding-bottom:22px;margin-bottom:28px}}
.eyebrow{{font-family:"JetBrains Mono",monospace;font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:8px}}
h1{{font-size:30px;letter-spacing:-.01em;margin-bottom:8px}}
.sub{{color:var(--ink2);max-width:66ch}}
.meta{{display:flex;gap:22px;flex-wrap:wrap;font-size:12.5px;color:var(--ink3);margin-top:14px}} .meta b{{color:var(--ink2);font-weight:600}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:8px;overflow:hidden;margin-bottom:28px}}
.stat{{background:var(--surface);padding:16px 18px}} .stat .l{{font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--ink3)}}
.stat b{{display:block;font-size:22px;font-weight:600;margin:6px 0 2px}} .stat b.hero{{color:var(--accent)}} .stat .s{{font-size:12px;color:var(--ink3)}}
.panel{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:20px 22px;margin-bottom:36px}}
.panel h2{{font-size:17px;margin-bottom:4px}} .panel .lead{{color:var(--ink2);font-size:13px;margin:0 0 14px}}
.chartwrap{{overflow-x:auto}} svg text{{font-family:"JetBrains Mono",monospace;font-size:11px;fill:var(--ink2)}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;color:var(--ink2);margin-top:8px}} .legend i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}}
.tip{{position:absolute;pointer-events:none;background:var(--ink);color:var(--bg);padding:4px 8px;border-radius:4px;font-size:12px;display:none;white-space:nowrap}}
section.method{{margin-bottom:44px}}
.mh{{display:flex;align-items:baseline;gap:12px;margin-bottom:4px}} .mh .letter{{font-family:"Spectral",serif;color:var(--accent);font-weight:600}} .mh h2{{font-size:20px}}
.lineage{{font-size:13px;color:var(--ink3);margin-bottom:14px}}
h3.grp{{font-size:14px;color:var(--ink2);margin:22px 0 8px;font-family:"Public Sans",sans-serif;font-weight:600;letter-spacing:.02em}}
.tablewrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--surface)}}
table{{width:100%;border-collapse:collapse;font-size:13px;min-width:860px}}
thead th{{text-align:left;font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--ink3);font-weight:600;padding:10px 12px 8px;border-bottom:1px solid var(--line2)}}
thead th.num{{text-align:right}}
tbody td{{padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}} tbody tr:last-child td{{border-bottom:none}}
tr.best{{background:var(--accent-soft)}}
td.name{{font-weight:600;min-width:120px;max-width:170px}} td.what{{color:var(--ink2);max-width:320px}} td.what .note{{color:var(--ink3);font-size:12px;margin-top:4px}}
.path{{color:var(--ink3);font-size:11.5px;margin-top:4px;word-break:break-all}}
td.scale{{color:var(--ink2);font-size:12.5px;white-space:nowrap}} td.scale b{{color:var(--ink)}}
td.counts{{color:var(--ink2);font-size:12.5px;white-space:nowrap;font-variant-numeric:tabular-nums}} td.counts b{{color:var(--ink)}} td.counts .sub{{color:var(--ink3);font-size:11.5px}}
td.num{{text-align:right;font-size:15px;font-weight:600;white-space:nowrap}} td.num .sub{{font-size:11px;font-weight:400;color:var(--ink3)}}
.vizrow{{display:flex;gap:6px}} .viz{{display:flex;flex-direction:column;align-items:center;gap:2px;width:58px;text-decoration:none;background:none;border:0;padding:0;cursor:zoom-in;font:inherit}}
.viz:focus-visible img{{outline:2px solid var(--accent);outline-offset:2px}}
#lb{{position:fixed;inset:0;background:rgba(10,14,18,.88);display:none;align-items:center;justify-content:center;flex-direction:column;gap:10px;z-index:50;padding:24px;cursor:zoom-out}}
#lb.open{{display:flex}} #lb img,#lb video{{max-width:96vw;max-height:86vh;border-radius:6px;background:#fff;box-shadow:0 12px 40px rgba(0,0,0,.5)}} #lb video{{cursor:default;background:#000}}
#lb .cap{{color:#e6ecf1;font-family:"JetBrains Mono",monospace;font-size:12.5px}} #lb .hint{{color:#93a0ad;font-size:12px}}
.viz img{{width:58px;height:58px;object-fit:cover;border-radius:5px;border:1px solid var(--line);background:var(--mono-bg)}} .viz:hover img{{border-color:var(--accent)}}
.viz span{{font-size:10.5px;color:var(--ink3)}} .none{{color:var(--ink3);font-size:12px;font-style:italic}}
.pill{{display:inline-block;font-size:11px;font-weight:600;padding:2px 9px;border-radius:100px;white-space:nowrap}}
.pill.ref{{background:var(--info-soft);color:var(--info)}} .pill.final{{background:var(--good-soft);color:var(--good)}} .pill.old{{background:var(--warn-soft);color:var(--warn)}}
.pill.bad{{background:var(--bad-soft);color:var(--bad)}} .pill.run{{background:var(--accent-soft);color:var(--accent)}}
.ktag{{font-size:10.5px;margin-top:6px;white-space:nowrap;font-family:"JetBrains Mono",monospace}} .ktag.k-ok{{color:var(--good)}} .ktag.k-warn{{color:var(--warn)}} .ktag.k-bad{{color:var(--bad)}} .ktag.k-neutral{{color:var(--ink3)}}
table.dev td,table.dev th{{font-size:12.5px;padding:8px 10px}} table.dev{{min-width:0}}
.callout{{border-left:3px solid var(--accent);background:var(--accent-soft);border-radius:0 6px 6px 0;padding:12px 16px;font-size:13.5px;color:var(--ink2);margin-top:14px}} .callout b{{color:var(--ink)}}
.callout.warn{{border-color:var(--bad);background:var(--bad-soft)}}
dl.cfg{{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;font-size:13px;margin:0}} dl.cfg dt{{color:var(--ink3);font-family:"JetBrains Mono",monospace;font-size:12px}} dl.cfg dd{{margin:0;color:var(--ink2)}}
footer{{margin-top:48px;padding-top:18px;border-top:1px solid var(--line);font-size:12.5px;color:var(--ink3)}}
@media (max-width:640px){{.stats{{grid-template-columns:repeat(2,1fr)}}}}
</style>
<div class="page">
<header>
<div class="eyebrow">ESVR · time-varying reconstruction · status report</div>
<h1>三种动态场景表征融入 ESVR 的实验记录</h1>
<p class="sub">把 Deformable 3D Gaussians、4D Gaussian Splatting、4D-Scaffold-GS 三种时变表征接入 ESVR 的标量体素直接监督框架(基元只有中心/尺度/旋转/一个标量强度,无颜色、无视角、无透明度合成)。本页汇总 vorts 90 帧序列上全部真实数据实验,含配置、改动、规模与结果。</p>
<div class="meta"><span><b>数据</b> vorts_timevarying,90 帧,128³,float32,值域归一化到 [0,1]</span><span><b>口径</b> 3D 体素 PSNR,逐帧计算后取均值</span><span><b>更新</b> {today}</span></div>
<div class="callout warn" style="margin-top:14px"><b>所有数字都是训练集拟合质量。</b>全部实验用全 90 帧训练,从未做过留出时间点的验证;因此本页不能回答“模型是否学到了可插值的连续运动”,动图也只是把 90 个训练时刻串起来,帧间没有插值。这是自项目交接起就存在、至今未补的最大空白。另:2026-09-22 起每次运行会写 <code>voxelizer.json</code>(加载的 CUDA 扩展路径与 SHA256),并拒绝外来扩展;各行的“核”标签是对更早运行的事后判定,依据见 A 节与 D.2 说明。</div>
</header>
<div class="stats">
<div class="stat"><div class="l">当前最佳动态模型</div><b class="hero">{best_fix["psnr"]:.2f} dB</b><div class="s">Scaffold4D · {best_fix["anchors"]//1000}k 锚点 · {best_fix["steps"]//1000}k 步 · 压缩后 {compact.get(best_key, {}).get("mb_after", best_fix["mb"]):.1f} MB</div></div>
<div class="stat"><div class="l">静态天花板(单帧,修复核)</div><b>49.67 dB</b><div class="s">139k 基元 · 67k 步 · 6.1 MB/帧;21k 基元 / 22k 步 = 41.72</div></div>
<div class="stat"><div class="l">Scaffold4D 已完成运行</div><b>{n_s4d + 11}</b><div class="s">含 15 组未修复核(作废)、11 组筛选、修复核完整训练 {n_s4d - 17} 组</div></div>
<div class="stat"><div class="l">目标</div><b>50 dB</b><div class="s">压缩前模型尽量小;静态单帧要 14 万基元 / 67k 步才到。探针:只训 10 帧、每帧优化 ×9 也只到 41.2(D.8)</div></div>
</div>
<div class="panel">
<h2>质量 – 模型大小总览</h2>
<p class="lead">横轴为推理所需权重的 float32 大小(对数),纵轴为 90 帧平均 PSNR。灰点是逐帧独立训练的静态 ESVR;淡蓝点是用未修复 CUDA 核跑的 Scaffold4D 消融(作废,仅作记录)和只训 10 帧的探针 K1(不与 90 帧可比)。悬停查看名称。</p>
<div class="chartwrap" id="chart"></div>
<div class="legend"><span><i style="background:var(--c-static)"></i>静态 ESVR(逐帧)</span><span><i style="background:var(--c-deform)"></i>Deformable-MLP</span><span><i style="background:var(--c-esvr4d)"></i>ESVR4D</span><span><i style="background:var(--c-s4d)"></i>Scaffold4D(修复核)</span><span><i style="background:var(--c-s4d);opacity:.35"></i>Scaffold4D(早期 / 未修复核)</span><span><i style="background:var(--good)"></i>Scaffold4D 训练后压缩(fp16 + 剪枝)</span></div>
</div>
''')

# ---------- A static ----------
parts.append('''<section class="method"><div class="mh"><span class="letter">A</span><h2>静态 ESVR 基线</h2></div>
<div class="lineage">不涉及时变表征,衡量"每帧独立重建能到多好";也是 40 dB 目标的参照系(逐帧 ESVR 每帧约 2.4 万个有效基元)</div>
<div class="tablewrap"><table><thead><tr><th>结果</th><th>配置 / 路径</th><th>规模 / 耗时</th><th>高斯数</th><th>可视化</th><th>状态</th><th class="num">平均 PSNR</th></tr></thead><tbody>''')
PF = lambda per, total, lo, hi, note: f'<b>{per:,}</b> / 帧(均值)<br>90 帧共 {total:,}<br><span class="sub">单帧 {lo:,} – {hi:,}</span><br><span class="sub">{note}</span>'
parts.append(row("零模型基线:复制上一帧", "不训练任何模型,用第 t−1 帧预测第 t 帧,89 对逐帧 PSNR 的均值(单对 24.3 – 30.4;帧 1→2 这一对是 30.39,2026-09-22 之前页面误把它当成了均值);用第 1 帧预测第 90 帧为 13.77 dB;90 帧均值体积对各帧只有 18.33 dB(时间不变成分仅占能量 13.5%)。任何低于此的动态模型都不如“什么都不建模”", "(数据本身)",
    "0 s<br>0 MB", "none", pill("ref", "零模型"), "28.80 dB", "24.3 – 30.4", counts="—", kernel="na"))
parts.append(row("单帧 vorts0001 基线", "原生 <code>train_chunk.py</code> 配方,22,400 迭代,增密+剪枝,未修复核", "output/experiments/vorts0001_baseline/",
    "<b>36.8 s</b><br>0.92 MB", "experiments__vorts0001_baseline", pill("ref"), "41.77 dB", counts='<b>21,401</b> 个高斯<br><span class="sub">20,000 初始 → 增密 → 剪枝后</span>', kernel="unfixed-by-design"))
parts.append(row("90 帧逐帧原生 ESVR", "与上行相同配方,90 帧各训一套,未修复核", "ESVR/output/esvr_original_per_frame/",
    "<b>3627 s</b>(40.3 s/帧)<br>93.77 MB", "esvr_original_per_frame", pill("ref"), "41.22 dB", counts=PF(23673, 2130583, 21131, 26656, "全部有效(每帧一个静态模型,无时间门)"), kernel="unfixed-by-design"))
parts.append(row("90 帧逐帧,“修复核”对照", "同上 + position 学习率 ÷4096;<b>实际经 activate 运行在未修复核上</b>(见 A 节说明),量到的是学习率 ÷4096 的代价", "output/experiments/vorts90_cuda_fixed_lr_only/",
    "<b>3466 s</b><br>≈ 90 MB", "experiments__vorts90_cuda_fixed_lr_only", pill("bad"), "39.38 dB", "38.65 – 39.94", counts=PF(24084, 2167597, 22391, 26262, "未修复核 + 位置几乎冻结"), kernel="unfixed"))
parts.append(row("单帧,“修复核”对照", "同首行 + position 学习率 ÷4096;<b>同样运行在未修复核上</b>", "output/experiments/vorts0001_cuda_fixed_lr_only/",
    "<b>36.0 s</b>", "experiments__vorts0001_cuda_fixed_lr_only", pill("bad"), "39.94 dB", counts='<b>22,424</b> 个高斯<br><span class="sub">20,000 → 峰值 83,546 → 剪枝后</span>', kernel="unfixed"))
parts.append(row("单帧,学习率+阈值都补偿", "position 学习率 ÷4096 且 densify 阈值 ×4096,未修复核:阈值相对梯度大 4096 倍,增密被抑制", "output/experiments/vorts0001_cuda_fixed_matched_lr/",
    "<b>36.7 s</b>", "experiments__vorts0001_cuda_fixed_matched_lr", pill("bad"), "38.81 dB", counts='<b>5,409</b> 个高斯<br><span class="sub">20,000 → 几乎只剪不增</span>', kernel="unfixed"))
STATIC = [("S1b_init20k_x1", "<b>修复核,正确配方</b>:原生学习率不变,densify 阈值 ×4096(2.5e-7 → 1.024e-3),其余同首行", "22,400 步 · 0.93 MB", 21247, 63332, 41.72, "final", "与未修复核的 41.77 一致:修复本身无代价"),
          ("S2b_init60k_x1", "同上,初始 60k 基元", "22,400 步 · 1.31 MB", 29817, 88809, 42.29, "ref", None),
          ("S3b_init20k_x3", "同 S1b,全部调度 ×3", "67,200 步 · 0.71 MB", 16243, 58938, 44.73, "ref", "更长训练比更多基元有效:比 S2b 高 2.4 dB,基元只有一半"),
          ("S4b_init60k_x3_noprune", "初始 60k,调度 ×3,关闭全局重要性剪枝", "67,200 步 · 2.78 MB", 63246, 87428, 47.05, "ref", None),
          ("S5b_init150k_x3_noprune", "初始 150k,调度 ×3,关闭全局剪枝", "67,200 步 · 6.13 MB", 139250, 164175, 49.67, "final", "<b>这个基元 + 核在此数据上的实测天花板:单帧 49.7 dB</b>,90 帧逐帧约 550 MB")]
for name, what, scale, final, peak, psnr, kind, note in STATIC:
    parts.append(row(name, what + (f'<div class="note">{note}</div>' if note else ""), f"output/experiments/static_ceiling/{name}_chunk_0/", f"<b>{scale.split(' · ')[0]}</b><br>{scale.split(' · ')[1]}", "none",
                     pill(kind, "静态天花板" if kind == "final" else "参照"), f"{psnr:.2f} dB", best=(kind == "final"),
                     counts=f'<b>{final:,}</b> 个高斯<br><span class="sub">峰值 {peak:,}</span>', kernel="fixed"))
parts.append(row("90 帧逐帧简化 ESVR", "<code>train_dynamic.py --allow-single-frame</code>,固定 3 万图元", "output/esvr_per_frame_baseline/",
    "<b>466 s</b><br>301 MB", "esvr_per_frame_baseline", pill("ref"), "38.86 dB", counts=PF(30000, 2700000, 30000, 30000, "固定,无增密"), kernel="fixed-likely"))
parts.append('''</tbody></table></div>
<div class="callout warn"><b>2026-09-22 更正——“修复核对照”(39.94 / 39.38)实际运行在未修复核上。</b>把 09-13 那次单帧配置在真正的修复核上原样重跑,第一次增密就失控(20,000 → 4.85M 基元,OOM),而 09-13 的日志里基元峰值只有 83,537——同一条命令在同一个核上不可能两种表现;它是经 <code>run_vorts0001.sh</code> 的 <code>source activate</code> 跑在 <code>ESVR/.venv</code> 的未修复扩展上的。机制:修复后位置梯度大 4096 倍,按未修复梯度调的 densify 阈值 2.5e-7 会选中全部基元;当时把 OOM 归因于学习率并 ÷4096,而 Adam 对梯度尺度不敏感,那一步只是把位置冻住。<b>正确配方</b>是学习率不动、阈值 ×4096:S1b = 41.72 dB,与未修复核的 41.77 一致——<b>修复本身没有代价</b>,之前“−1.84 dB”是位置冻结的代价。</div>
<div class="callout"><b>高斯数(从各帧最终迭代的 PLY 直接计数):</b>原生逐帧 90 帧共 2,130,583 个(均值 23,673)。<b>静态天花板(S 系列):</b>同一基元与核,单帧加到 139k 基元、67k 步,可达 <b>49.67 dB</b>(6.1 MB/帧,90 帧约 550 MB;原始数据 755 MB)。训练长度和容量同样重要:20k 基元训 3 倍长(S3b)44.7 dB,超过 60k 基元训 1 倍长(S2b)42.3。这给出两个参照:与动态模型同等每时刻容量(~2 万有用高斯)的静态模型充分训练可到 44–45 dB;50 dB 需要每时刻约 14 万个有效基元。<br><br><b>每帧优化预算对比(动态模型为何落后静态同容量模型 ~5 dB):</b>静态 S3b 每帧 67,200 步 × 64³ 块 ≈ 8,400 次全体积等效遍历;动态 I3 是 480k 步 ÷ 90 帧 ÷ 8 块 ≈ 670 次/帧,少 12.6 倍(120k 步的运行只有 167 次/帧)。ESVR 论文自己的 vorts 配方(22,400 步)也只是 2,800 次/帧,离收敛很远。原生配方的 <code>position_lr</code> 是在梯度被压缩 4096 倍的前提下调出来的,修复后必须 ÷4096 补偿才能稳定。</div></section>''')

# ---------- B deformable ----------
parts.append('''<section class="method"><div class="mh"><span class="letter">B</span><h2>Deformable-MLP</h2></div>
<div class="lineage">Deformable 3D Gaussians → canonical 椭球 + 独立形变 MLP(位置+时间 Fourier 编码 → 位移/旋转/尺度偏移),分支 <code>codex/deformable-esvr</code></div>
<div class="tablewrap"><table><thead><tr><th>版本</th><th>配置 / 路径</th><th>规模 / 耗时</th><th>锚点 / 高斯</th><th>可视化</th><th>状态</th><th class="num">平均 PSNR</th></tr></thead><tbody>''')
FIXED = lambda n: f'<b>{n:,}</b> 个高斯<br>固定,无增密<br><span class="sub">全程全部活跃(一套 canonical + 形变网络)</span>'
parts.append(row("历史基线", "未修复核,PLY 热启动,21,401 图元固定", "output/dynamic/vorts90_full_geometry/", "<b>432 s</b> · 40,000 步<br>2.83 MB", "dynamic__vorts90_full_geometry", pill("old", "已过时"), "22.35 dB", counts=FIXED(21401), kernel="unfixed-by-design"))
parts.append(row("配置对齐版", "修复核,随机初始化,3 万图元(对齐 ESVR4D),单一共享学习率", "output/dynamic/vorts90_matched_comparison/", "<b>1023 s</b> · 40,000 步<br>3.19 MB", "dynamic__vorts90_matched_comparison", pill("old", "仅供对比"), "25.96 dB", counts=FIXED(30000), kernel="fixed-script"))
parts.append(row("单变量学习率对照", "配置对齐 + 学习率换成原生量级,但 4 组仍合并为一个值", "output/dynamic/vorts90_matched_native_lr/", "<b>488 s</b> · 40,000 步<br>3.19 MB", "dynamic__vorts90_matched_native_lr", pill("old", "仅供对比"), "24.21 dB", counts=FIXED(30000), kernel="fixed-script"))
parts.append(row("<b>原生学习率结构修复版</b>", "position/scale/rotation/opacity 拆成 4 个独立学习率(取 ESVR 自身静态值)+ 形变网络自己的组", "output/dynamic/vorts90_native_canonical_lr/", "<b>500 s</b> · 40,000 步<br>3.19 MB", "dynamic__vorts90_native_canonical_lr", pill("final"), "28.90 dB", best=True, counts=FIXED(30000), kernel="fixed-script"))
parts.append('''</tbody></table></div>
<div class="callout">未实现增密(donor 的判据基于相机屏幕梯度)。canonical 学习率取 ESVR 自身值而非 donor 原生值,与另两个方法的"用 donor 原生值"标准不一致,尚无定论。</div></section>''')

# ---------- C esvr4d ----------
parts.append('''<section class="method"><div class="mh"><span class="letter">C</span><h2>ESVR4D</h2></div>
<div class="lineage">4D Gaussian Splatting(复旦)→ 原生 4D 时空协方差,对时间做 Schur 补条件化得到每时刻的 3D 椭球再交给 ESVR 核,无网络;分支 <code>codex/4dgs-esvr</code></div>
<div class="tablewrap"><table><thead><tr><th>版本</th><th>配置 / 路径</th><th>规模 / 耗时</th><th>锚点 / 高斯</th><th>可视化</th><th>状态</th><th class="num">平均 PSNR</th></tr></thead><tbody>''')
FIXED4D = lambda n: f'<b>{n:,}</b> 个 4D 高斯<br>固定,无增密<br><span class="sub">每时刻按时间边缘 &gt; 0.05 切片选入</span>'
parts.append(row("合并 LR 组(修复前)", "修复核,3 万图元固定,增密关闭,xyz/time/scale/opacity 共用 lr 5e-3", "output/esvr4d/vorts90_fixed_points/", "<b>179 s</b> · 40,000 步<br>2.29 MB", "esvr4d__vorts90_fixed_points", pill("old", "已定位为 bug"), "28.77 dB", counts=FIXED4D(30000), kernel="fixed-likely"))
parts.append(row("原生 LR 结构(步数未对齐)", "5 组原生学习率(xyz 1.6e-4→1.6e-6 等),但只跑 30,000 步", "output/esvr4d/vorts90_native_lr/", "<b>158 s</b> · 30,000 步<br>2.29 MB", "esvr4d__vorts90_native_lr", pill("old", "被步数对齐版取代"), "25.09 dB", counts=FIXED4D(30000), kernel="fixed-likely"))
parts.append(row("<b>原生 LR 结构(步数对齐)</b>", "同上,步数补齐到 40,000,与首行严格单变量;重载零误差", "output/esvr4d/vorts90_native_lr_40k/", "<b>213 s</b> · 40,000 步<br>2.29 MB", "esvr4d__vorts90_native_lr_40k", pill("final"), "25.68 dB", best=True, counts=FIXED4D(30000), kernel="fixed-likely"))
parts.append('''</tbody></table></div>
<div class="callout">原生位置学习率比共享值小 30–1000 倍,而 ESVR4D 的运动完全依赖 xyz 本身移动,所以修正学习率结构后 PSNR 反而下降 −3.09 dB(单变量验证)。增密实现了但关闭:开启后质量回退(加法渲染下 clone/split 原地翻倍),次要因素未查明。</div></section>''')

# ---------- D scaffold4d ----------
th = '<div class="tablewrap"><table><thead><tr><th>运行</th><th>相对参照的改动 / 路径</th><th>规模 / 耗时</th><th>锚点 / 高斯</th><th>可视化</th><th>状态</th><th class="num">平均 PSNR</th></tr></thead><tbody>'
parts.append('''<section class="method"><div class="mh"><span class="letter">D</span><h2>Scaffold4D</h2></div>
<div class="lineage">4D-Scaffold-GS → 稀疏 4D 锚点(冻结)+ 32 维特征 + 三个共享 MLP 生成 K=10 个子椭球,恒定速度运动,广义高斯时间门(β=4);分支 <code>codex/scaffold4d-esvr</code></div>
<div class="panel" style="margin-bottom:18px"><h2>共同配方(除非表中注明)</h2>
<dl class="cfg">
<dt>训练</dt><dd>120,000 步;每步随机 1 帧 + 随机 64³ 块(tile 采样);损失 0.8·L1 + 0.2·(1−SSIM) + 0.01·mean(∏scale)</dd>
<dt>优化器</dt><dd>4D-Scaffold-GS 原生 7 组:anchor 0(冻结)、offset 0.01→1e-4、log_offset_scale 0.007、feature 0.0075、opacity-MLP 0.002→2e-5、shape-MLP 0.004、flow-MLP 0.001;Adam eps 1e-15,不裁剪</dd>
<dt>锚点生长</dt><dd>上游算法移植:统计 ‖∂L/∂xyz‖·opacity_t·σ^γ(γ=1,ESVR 的 3D 中心梯度替代 2D 屏幕梯度),窗口 1,500–60,000 每 100 步,三级 4D 格子 16/4/1 体素 × 1 帧,只在空格子中心新建,特征 max-pool;阈值按实测梯度 p90 校准(修复核 1.2e-3;未修复核时 4e-7)</dd>
<dt>ESVR 适配项</dt><dd>子椭球尺度下限 0.5 体素;opacity 头 sigmoid(上游 tanh+掩码会在零特征初始化下全部死亡);时间截断 1e-3;σ/offset 数值钳位;锚点数上限(上游无)</dd>
<dt>初始化</dt><dd>6,000 锚点均匀随机(空间 + 时间),尺度取最近邻距离;子椭球初始幅值 sigmoid(0)=0.5;速度上限 tanh 原生 1</dd>
<dt>锚点 / 高斯列</dt><dd><b>锚点</b> = 训练结束时的锚点数;<b>高斯</b> = 锚点 × K,即模型能生成的子椭球总数(每个锚点固定生成 K 个,多余的靠压小/压暗关闭);<b>活跃/查询</b> = 训练末段每步渲染时通过时间门(opacity &gt; 1e-3)被送入 CUDA 的子椭球数;<b>有用/查询 t=0.5</b> = 其中强度 &gt; 1/255 且尺度未贴在 0.5 体素下限的,在序列中点 t = 0.5 这一个时刻切片统计;<b>有用/查询 90 帧均值</b> = 同一条件在全部 90 个训练时刻各切一次后的均值(括号内为最小–最大);<b>有用总数</b> = 全体子椭球中满足同一条件的数量及占比(与时间无关);<b>90 帧从未选入</b> = 在全部 90 个训练时刻都没有通过时间门(home 时刻漂出 [0,1] 或时间窗落在两帧之间)的子椭球,它们对这 90 帧的重建零贡献、可精确删除;“其中有用”是这些里按强度/尺度本来算“有用”的数量——修复核的 90 帧运行里只有 ~1%,说明“有用总数”几乎没有被时间上死掉的高斯抬高(例外是只训 10 帧的 K1:随机 t 初始化让 61% 的孩子落在没训过的时刻,其中 6.3 万“有用”);<b>任一时刻起作用</b> = 有用 且 至少在一个训练时刻被选入,即真正参与过重建的子椭球数及占比;<b>全程无用</b> = 其余全部(太暗 / 贴尺度下限 / 从未选入),这两个数相加等于“高斯”总数。“有用/查询”与“任一时刻起作用”之差反映跨时刻复用的程度。对照:静态单帧 ESVR(修复核、正确配方)21k 基元 / 22k 步 = 41.7 dB,16k 基元 / 67k 步 = 44.7 dB,139k 基元 / 67k 步 = 49.7 dB(A 节 S 系列)</dd>
</dl></div>
<h3 class="grp">D.1 早期版本(核状态未记录)</h3>''' + th)
parts.append(s4d("配置对齐版", "vorts90_matched_comparison", "对齐 ESVR4D:40,000 步、6,000 锚点 × 5 子元、单一 lr 5e-3、feat_lr 5e-3、MLP lr 2e-3、增密关;clone/split 式增密代码", pill("old"), note="children/feat_lr 均非论文原生值"))
parts.append(s4d("原生配方版", "vorts90_native_recipe", "论文原生 7 组学习率、120k 步、增密开(ESVR4D 借来的 clone/split,阈值 2e-4)、scale 下限 1e-4、初始幅值 0.5、速度上限 1", pill("old"), note="训练中途退化:活跃子椭球 33 万 → 2.1 万,PSNR 30k 步时 ~26.5 后跌到 24.3;半数子椭球贴在 1e-4 下限永久失活;14k/40k 锚点全透明未剪"))
parts.append('</tbody></table></div><h3 class="grp">D.2 修复适配后的留一法消融 —— 全部经由未修复 CUDA 核训练(作废,仅作记录)</h3>')
parts.append('''<div class="callout warn" style="margin:0 0 10px"><b>为什么作废:</b><code>ESVR-4dgs/.venv/bin/activate</code> 是 9 月 4 日随目录从 <code>ESVR/</code> 复制来的,硬编码 <code>VIRTUAL_ENV=…/ESVR/.venv</code>;<code>source activate && python</code> 实际运行原文件夹的解释器和未修复扩展(位置梯度 ×1/4096)。有限差分:经 activate 比值 36 = 1/dVoxel²,经 <code>.venv/bin/python</code> 比值 1.00。下列 15 组全部如此。前向(PSNR 的度量)正确,错的是训练信号;结论方向在修复核上全部复现(见 D.3)。</div>''' + th)
B = "B_full"
parts.append(s4d("B_full(参照)", "ablation/B_full", "全部适配修复 + 上游式生长;速度上限 8、初始幅值 0.1、锚点按 GT 强度采样(data init)", pill("bad")))
parts.append(s4d("E_floor_old", "ablation/E_floor_old", "scale 下限改回 1e-4(0.0064 体素)", pill("bad"), note="−3.32 dB:小于 0.45 体素的基元碰不到体素中心,零梯度、永久死亡"))
parts.append(s4d("A_fixed6000", "ablation/A_fixed6000", "关闭生长,6,000 锚点固定", pill("bad"), note="−1.43 dB"))
parts.append(s4d("H_old_optimizer", "ablation/H_old_optimizer", "Adam eps 1e-8 + 全局梯度裁剪 1.0(旧设置)", pill("bad"), note="−0.59;未修复核下梯度与 eps 同量级"))
parts.append(s4d("G_init_random", "ablation/G_init_random", "锚点均匀随机初始化", pill("bad"), note="0.00"))
parts.append(s4d("F_opacity_init0.5", "ablation/F_opacity_init0.5", "初始幅值 0.5", pill("bad"), note="+0.13"))
parts.append(s4d("I_no_sigma_weight", "ablation/I_no_sigma_weight", "γ=0,即普通 Scaffold-GS 统计", pill("bad"), note="+0.17"))
parts.append(s4d("D_flow1", "ablation/D_flow1", "速度上限改回原生 1", pill("bad"), note="+0.52,在种子噪声内"))
parts.append(s4d("C_cap100k", "ablation/C_cap100k", "锚点上限 100k", pill("bad"), note="+0.78"))
parts.append(s4d("B_full_s1 / s2", "ablation/B_full_s1", "B 换种子 1(s2 = 31.59)", pill("bad"), note="B 三种子 31.62 ± 0.24"))
parts.append(s4d("J_default_s42 / s1 / s2", "ablation/J_default_s42", "当前默认(速度 1、幅值 0.5、随机初始化)三种子:31.31 / 32.13 / 31.41", pill("bad"), note="31.61 ± 0.45;与 B 无差别;种子噪声最大 0.8 dB"))
parts.append(s4d("K_fixed6000_flow1", "ablation/K_fixed6000_flow1", "6,000 固定 + 当前默认(与 A 差三项,非单变量)", pill("bad")))
parts.append('</tbody></table></div><h3 class="grp">D.3 修复核重跑(当前有效结果)</h3>' + th)
parts.append(s4d("<b>F_default_s42</b>", "fixed_kernel/F_default_s42", "当前默认配置(见上方共同配方),40k 上限", pill("final"), note="与未修复核同配置(31.61 ± 0.45)持平"))
parts.append(s4d("F_default_s1 / s2", "fixed_kernel/F_default_s1", "同上,种子 1(s2 = 31.52)", pill("final"), note="三种子 31.84 ± 0.29 dB —— 修复核下的噪声尺子;≤ 0.6 dB 的单次差别不作结论"))
parts.append(s4d("<b>F_cap100k</b>", "fixed_kernel/F_cap100k", "锚点上限 100k", pill("final"), best=True, note="+1.03 dB;预算未饱和"))
parts.append(s4d("F_fixed6000", "fixed_kernel/F_fixed6000", "关闭生长,6,000 锚点固定", pill("final"), note="−2.53 dB:上游式生长在加法渲染下有效"))
parts.append(s4d("F_floor_old", "fixed_kernel/F_floor_old", "scale 下限改回 1e-4", pill("final"), note="−2.95 dB"))
parts.append('''</tbody></table></div>
<div class="callout"><b>修复核 checkpoint 诊断(F_default_s42):</b>40 万子椭球中只有 23% 有用(强度 > 1/255 且不贴下限),39% 的锚点没有一个有用孩子,每次渲染选入的孩子里 60% 一个体素都碰不到(ESVR <code>prune_zeros</code> 规则),每时刻有用孩子 1–1.7 万,而逐帧 ESVR 每帧约 2.4 万。上游的 <code>min_opacity</code> 剪枝在 sigmoid 下失效(整个训练只删 1–70 个锚点)。这是 40 dB 之前必须解决的容量问题。</div>
</section>
<section class="method"><h3 class="grp">D.4 筛选:回收死容量 / 每锚点孩子数(修复核,40k 步,生长窗口 1.5k–20k,每组只改一项)</h3>''' + th)
SCREEN = [("T0_control", "基线(当前默认)", "13.3", 30.823, "27.91 – 32.49", "", False),
 ("T1_volreg_visible", "体积正则只作用可见孩子(<code>--vol-reg-visible</code>)", "13.3", 30.779, "27.47 – 32.27", "−0.04", False),
 ("T2_prune_dead", "ESVR 式死锚点回收(<code>--prune-dead</code>)", "13.3", 30.972, "27.26 – 32.82", "+0.15;删了 12,385 死锚点又长回 12.7k,有用率仍 20% —— 新生的同样死掉", False),
 ("T3_spread", "新生 10 个孩子在格子内散开(<code>--offset-spread 0.5</code>)", "13.3", 30.075, "27.40 – 31.49", "−0.75", False),
 ("T4_childscale4", "新生子尺度 4 体素", "13.3", 29.626, "26.20 – 31.00", "−1.20:大孩子叠在一点过冲更重", False),
 ("T5_sigma1000", "σ 钳位 100 → 1000", "13.3", 30.493, "27.49 – 31.71", "−0.33", False),
 ("T6_novol", "λ_vol = 0(ESVR 本无体积正则)", "13.3", 30.253, "26.41 – 31.77", "−0.57:正则对抑制大孩子有用", False),
 ("T7_combo", "T1+T2+T3+T4", "13.3", 30.305, "27.35 – 31.81", "−0.52", False),
 ("T8_K5", "K = 5", "10.1", 30.475, "26.90 – 31.91", "−0.35,大小 −24%", False),
 ("T9_K5_cap80k", "K = 5,锚点上限 80k", "20.2", 31.581, "27.79 – 33.11", "+0.76,大小 ×1.5", True),
 ("T10_K3", "K = 3", "8.8", 30.332, "27.01 – 31.58", "−0.49,大小 −34%", False)]
for name, what, mb, psnr, mm, note, best in SCREEN:
    parts.append(row(name, what + (f'<div class="note">{note}</div>' if note else ""), f"output/scaffold4d/screen_fixed/{name}/", f"40,000 步<br>{mb} MB", "screen__" + name, pill("final", "筛选"), f"{psnr:.2f} dB", mm, best, counts=count_cell("screen_fixed/" + name), kernel="fixed"))
parts.append('''</tbody></table></div>
<div class="callout"><b>筛选结论:</b>七种"回收/保护死容量"的修法没有一种有效,三种有害。死容量是存储上的浪费,不是 PSNR 的瓶颈——40k 锚点上模型已把可用容量用尽,多余孩子被压死是优化器在"关闭"它们(上游用 tanh 免费掩掉,这里只能压小/压暗)。K=3/5/10 每字节效率相近,K=5 略优。</div>
</section>
<section class="method"><h3 class="grp">D.5 阶段 2:预算、步数、学习率衰减(修复核,120k 步)</h3>''' + th)
parts.append(s4d("G1_K10_150k_120k", "scale/G1_K10_150k_120k", "锚点上限 150k(长满)", pill("final"), note="对比 F_cap100k(33 MB,32.95):33→50 MB 只 +0.04 dB,预算在此配方下饱和;监控曲线 60k 步后走平且相邻检查点摆动 ±0.7 dB"))
parts.append(s4d("G2_K5_240k_120k", "scale/G2_K5_240k_120k", "K=5,锚点上限 240k(实际长到 195k,后期梯度低于阈值)", pill("final"), note="与 G1 大小匹配(49 MB):K=5 与 K=10 无差别"))
parts.append(s4d("<b>G5_K10_150k_120k_decay</b>", "scale/G5_K10_150k_120k_decay", "G1 + <code>--final-lr-scale 0.01</code>:feature / scale / shape-MLP / flow-MLP 四组上游全程恒定的学习率也按对数线性衰减到 1/100(offset、opacity-MLP 上游本就衰减)", pill("final"), best=True, note="单变量 +5.05 dB;曲线单调(20k 29.7 → 60k 35.8 → 100k 37.2 → 120k 37.2);重载逐字节一致;每次查询约 2.2 万有用孩子,接近逐帧 ESVR 的 2.4 万"))
parts.append('''</tbody></table></div>
<div class="callout"><b>学习率衰减是目前最大的单项杠杆。</b>上游 4D-Scaffold-GS 在 alpha 合成 + 多相机整帧监督下用恒定学习率;ESVR 的加法渲染 + 单个 64³ 块监督噪声更大,后期恒定学习率让模型在最优点附近来回跳(G1)。ESVR 自身的 position / opacity 学习率都是衰减的——这是"上游没有、参考 ESVR"的一项。G3/G4(480k 步、无衰减)在 G5 出结果后停止,GPU 让给下面的对照。</div>
</section>
<section class="method"><h3 class="grp">D.6 围绕学习率衰减的对照(修复核;每组相对 G5 只改一项)</h3>''' + th)
DECAY = [("H1_40k_decay0.01", "锚点上限 40k(与 13 MB 的 F_default 大小匹配)", "对比无衰减的 F_default 31.84 ± 0.29:+3.6 dB,小模型同样受益", False),
         ("H2_150k_decay0.1", "衰减到 1/10", "+0.37,在种子噪声(0.42)内", False),
         ("H3_150k_decay0.001", "衰减到 1/1000", "−0.19,噪声内;1/10–1/1000 都可", False),
         ("H4_150k_decay0.01_480k", "480k 步(生长窗口与衰减同比拉长)", "+1.40,最高帧 40.30;重载逐字节一致。(原注“与逐帧 ESVR 39.38 持平”已作废:那个基线是未修复核;正确的静态参照是 41.7–44.7 dB,见 A 节)", True),
         ("H5_300k_decay0.01", "锚点上限 300k(~65 MB)", "+0.11:预算不是瓶颈", False),
         ("H6_150k_decay0.01_s1", "种子 1", "与 G5 差 0.42 dB = 修复核 + 衰减下的种子噪声尺子", False)]
for name, what, note, best in DECAY:
    run = "decay/" + name
    if run in runs:
        parts.append(s4d(name, run, what, pill("final"), best=best, note=note or ""))
    else:
        parts.append(row(name, what, f"output/scaffold4d/decay/{name}/", "120,000 步" if "480k" not in name else "480,000 步", "none", pill("run"), "—", counts="—"))
parts.append('''</tbody></table></div>
<div class="callout"><b>结论:</b>步数是当前瓶颈(×4 → +1.4 dB),锚点预算不是(150k → 300k 无收益),衰减系数 1/10–1/1000 均可。<b>训练后压缩</b>(<code>experiments/compact_scaffold4d.py</code>,只读 checkpoint、产物写到 <code>compact/</code>):删掉核永远画不出的孩子(强度 &lt; 1/255,精确无损)、贴在尺度下限的孩子(−0.01 dB)和空锚点,只存保留孩子的 offset + 每孩子 1 位掩码,fp16——G5 47.3 → 7.2 MB、H4 49.8 → <b>7.2 MB / 39.42 dB</b>。训练时约 90% 的字节属于被优化器关掉的孩子。"规模"列里的"压缩后"即此。</div>
</section>
<section class="method"><h3 class="grp">D.7 冲 40 dB:步数与预算的交叉(修复核,衰减 1/100,均以 H4 / H1 为基准单变量)</h3>''' + th)
LONG = [("I3_300k_decay0.01_480k", "H4 + 锚点上限 300k", "+0.30(噪声内,但 120k 步时 300k 无收益,步数够长后预算开始起作用);重载逐字节一致", True),
        ("I2_150k_decay0.01_960k", "H4 + 960k 步", "−0.10:步数在 480k 附近饱和", False),
        ("J1_40k_decay0.01_480k", "H1 + 480k 步(40k 锚点不变,只加步数)", "+1.48 vs H1:与 150k 锚点上 H4 vs G5 的 +1.40 一致——步数 ×4 的收益不依赖预算;压缩后 2.6 MB", False)]
for name, what, note, best in LONG:
    parts.append(s4d(name, "long/" + name, what, pill("final"), best=best, note=note))
parts.append(row("I1_150k_decay0.1_480k", "H4 + 衰减 1/10", "output/scaffold4d/long/I1_150k_decay0.1_480k/", "已停止", "none", pill("old", "未完成"), "—", counts="—"))
parts.append('''</tbody></table></div>
<div class="callout"><b>两个维度近似可加:</b>预算 40k → 150k 约 +2.5 dB,步数 ×4 约 +1.5 dB,再往上都饱和(150k → 300k +0.3,480k → 960k −0.1)。I1(衰减 1/10)因四个运行并发显存吃紧而停止(H2 已表明 1/10 与 1/100 在噪声内)。最佳 I3 = 39.73 dB,距 40 dB 均值 0.27,已在种子噪声(0.42)之内;最高帧 40.69。</div>
</section>''')

parts.append('''<section class="method"><h3 class="grp">D.8 冲 50 dB 的四个探针(修复核,衰减 1/100,均以 H4 = 39.43 dB 为基准单变量;2026-09-22)</h3>''' + th)
FIFTY = [("K1_10frames_480k", "只训第 41–50 帧,步数不变 → 每帧优化量 ×9;<code>--grow-t-size 1/89</code> 保持时间格与 90 帧一致(默认会变成 1/9)", "探针",
          "10 帧均值 <b>41.20</b>(H4 在同 10 帧 39.43,I3 39.68):每帧优化 ×9 只 +1.8 dB。锚点只长到 49k(上限 150k 没长满,生长在 2,384 次检查里逐渐停止),每次查询有用孩子 3.9 万 ≈ H4 的 2.3 倍。容量翻倍 + 优化 ×9 仍比每帧预算相近的静态单帧(S3b:1.6 万基元 / 67k 步 = 44.7)低 3.5 dB → 缺口不只是训练量。副作用:随机 t 初始化让 49 万孩子里 30 万在这 10 个时刻从未被选中,其中 6.3 万“有用”孩子活在 41–50 帧之外的时间上,白占存储(压缩后 3.5 MB 已含它们)。<b>不与 90 帧数字直接比较。</b>", False),
         ("K2_block128_60k", "整体积(128³)监督、60k 步:与 480k 步 × 64³ 计算量相当,但优化器更新次数只有 1/8", "负面记录",
          "<b>33.32</b>,比 H4 低 6.1 dB,和 120k 步 × 64³ 的 F_default(31.8)同一档。同等 GPU 计算量下,瓶颈是<b>更新次数</b>,不是每步监督的范围;整体积监督也没有让生长更好(只长到 12k 锚点)。", False),
         ("K3_grow75_480k", "生长窗口 50% → 75% 的训练(<code>--densify-until 360000</code>)", "负面记录",
          "<b>38.94</b>(−0.49,略超种子噪声 0.42)。150k 上限照样长满,但更晚加入的锚点在学习率已衰减后没有时间训好:每次查询有用孩子从 H4 的 1.67 万降到 1.33 万,有用总数 8.8 万 vs 12.9 万。生长窗口 50%(与上游 60k/120k 的比例一致)已经足够。", False),
         ("K4_cap60k_prune_480k", "锚点上限 60k + <code>--prune-dead</code>(ESVR 式死锚点回收;累计回收 26,190 个锚点后由生长补回)", "当前结论",
          "<b>38.20</b>,压缩前 <b>20 MB</b> / 压缩后 4.0 MB。比 H4 低 1.2 dB(大小 2/5),比 J1(40k 上限,13 MB)高 1.3 dB:40k → 60k → 150k 锚点 = 36.9 → 38.2 → 39.4,K4 落在同一条预算曲线上,回收带来的“新鲜”锚点没有额外收益(与 D.4 的 T2 一致)。这是目前“压缩前就小”路线的最好点。", True)]
for name, what, tag, note, best in FIFTY:
    kind = {"探针": "final", "负面记录": "neg", "当前结论": "final"}[tag]
    parts.append(s4d(name, "fifty/" + name, what, pill(kind, tag), best=best, note=note))
parts.append('''</tbody></table></div>
<div class="callout"><b>结论:四条路都没有把 90 帧模型推过 40 dB,K1 给出了最重要的信息。</b>即使每帧优化量 ×9、每时刻有用高斯翻倍,10 帧模型也只有 41.2 dB;而同一基元、同一核的静态单帧在更少基元、相近每帧优化量下有 44.7(S3b),加到 14 万基元有 49.7(S5b)。所以 50 dB 的缺口有三层:(1) <b>时间维共享的表征代价</b>——所有时刻的孩子都由同一个锚点特征经共享 MLP 生成,只能做线性运动 + 一个广义高斯时间门,一个孩子跨帧复用时对每一帧都是近似;(2) <b>每时刻有效容量不足</b>——150k 锚点 × 10 孩子里每时刻只有约 1.7–2 万个有用高斯,而静态 50 dB 需要每时刻约 14 万个,现有生长规则(只往空格子里长、3D 梯度过阈值)长不出、也养不活这么多;(3) 步数在 480k 附近饱和(I2)。K2 说明瓶颈是更新次数而非批量;K3 说明把生长拖后无益;K4 说明小上限 + 回收在同一条预算曲线上,是控制“压缩前大小”的正当手段但不涨 PSNR。<br><br><b>下一步候选(均未做):</b>a) 提高每时刻有效容量——放宽“只长空格子”的限制、降低生长阈值或让每个锚点生成更多能存活的孩子,并用 K4 式上限 + 回收控制大小;b) 提高时间分辨力——每锚点分段运动、更窄的时间门或“时间局部”锚点,减少跨帧复用带来的近似;c) 补做<b>留出帧验证</b>,在加容量之前确认模型不是在逐帧记忆。</div>
</section>
<section class="method"><h3 class="grp">D.9 第二轮探针:σ 上限、分组、按误差采样、基础学习率(修复核;L1 / M1 / E 以 H4 = 39.43 为基准,N 系列以 120k 步的 G5 = 38.04 为基准;2026-09-22)</h3>''' + th)
NEXT = [("L1_sigma1000_480k", "H4 + 时间门 σ 上限 100 → 1000(<code>--sigma-max</code>)", "neg", "负面记录",
         "<b>38.96</b>(−0.47)。动机:诊断显示 H4 有 37% 的孩子贴在 σ = 100 上限(最窄半宽 0.8 帧)。放开后只有 0.1% 贴新上限,但<b>有用</b>孩子的时间半宽分布不变(中位 3.56 vs 3.48 帧,p5 1.0 vs 1.15)——原来贴上限的几乎都是暗 / 贴尺度下限的死孩子。σ 上限不是瓶颈。", False),
        ("M1_errsample_480k", "H4 + 按误差采样训练块(<code>--error-sampling 1.0</code>:(帧, 64³ 块) 按上次 MSE 成比例抽样,25% 均匀份额;<code>scaffold4d/sampling.py</code>)", "neg", "负面记录",
         "<b>39.15</b>(−0.28,噪声内),但 90 帧里 87 帧低于 H4;帧间标准差 0.42 → 0.34,最低帧 37.05 → 37.26。最终误差表里最难 / 最易块的 MSE 只差 2 倍——误差分布近乎平坦,没有“难块”可挖,挪走的均匀覆盖反而略有代价(帧采样次数 4,697–7,151,均匀为 5,333)。第一次启动因四个并发运行显存不足在 5k 步 OOM,重跑。", False)]
for name, what, kind, tag, note, best in NEXT:
    run = "next/" + name
    if run in runs:
        parts.append(s4d(name, run, what, pill(kind, tag), best=best, note=note))
    else:
        parts.append(row(name, what, f"output/scaffold4d/next/{name}/", "480,000 步", "none", pill("run"), "—", counts="—"))
if len(E_GROUPS) == 9:
    e_sec = sum(runs[k]["seconds"] for k in E_GROUPS); e_lo = min(runs[k]["pmin"] for k in E_GROUPS); e_hi = max(runs[k]["pmax"] for k in E_GROUPS)
    e_c = sum(compact[k]["mb_after"] for k in E_GROUPS) if all(k in compact for k in E_GROUPS) else None
    e_dn = sum(compact[k]["dropnever"]["mb_after"] for k in E_GROUPS) if all("dropnever" in compact.get(k, {}) for k in E_GROUPS) else None
    st = [stats[k] for k in E_GROUPS]
    e_total = sum(x["total"] for x in st); e_useful = sum(x["useful_total"] for x in st); e_never_useful = sum(x["useful_never_selected"] for x in st)
    per = " · ".join(f'E{i + 1} {runs[k]["psnr"]:.1f}' for i, k in enumerate(E_GROUPS))
    scale = f'<b>{e_sec:,} s</b> 合计 · 9 × 53,333 步<br>{e_mb:.1f} MB 合计' + (f'<br><span class="sub">压缩后 <b>{e_c:.1f} MB</b>' + (f';再删“从未选入” {e_dn:.1f} MB' if e_dn else "") + '</span>' if e_c else "")
    counts = (f'<b>9 × 16,667</b> 锚点 × 10<br>= {e_total:,} 高斯<br><span class="sub">有用总数 {e_useful:,} ({e_useful / e_total:.0%})</span>'
              f'<br><span class="sub">各组 10 帧从未选入、其中“有用” 合计 {e_never_useful:,}(随机 t 初始化把容量放在了本组没训的时刻)</span>'
              f'<br><span class="sub">有用/查询 各组均值 {min(x["useful_query_mean"] for x in st):,}–{max(x["useful_query_mean"] for x in st):,}</span>'
              f'<br><span class="sub">时间半宽中位 {min(x["halfwidth_frames_p50"] for x in st):.0f}–{max(x["halfwidth_frames_p50"] for x in st):.0f} 帧(53k 步时时间门尚未收窄;480k 步的 K1 是 0.9、H4 是 3.5)</span>')
    parts.append(row("E1–E9 分组重建(合计)", "90 帧分 9 组、每组 10 帧独立训练:每组 53,333 步、锚点上限 16,667,合计恰为 H4 的 480k 步 / 150k 锚点(<code>--grow-t-size 1/89</code>);可视化为 E5(第 41–50 帧,与 K1 同一段)"
        + f'<div class="note"><b>{e_psnr:.2f}</b>(−{39.431 - e_psnr:.1f} vs H4;各组 vs H4 同段 −3.6 … −11.3)。{per}。同等总计算量下,把 90 帧放进一个模型远好于 9 个各训 1/9 的小模型:共享的锚点特征 + MLP 让一帧上的更新惠及其它帧。K1 的 +1.8 dB(D.8)只是 9 倍每帧计算量买来的。E9(第 81–90 帧)最差,其时间门中位半宽 117 帧,几乎没有时间分辨。</div>',
        "output/scaffold4d/next/E_group{1..9}_frames_*/", scale, "scaffold4d__next__E_group5_frames_41_50", pill("neg", "负面记录"), f'{e_psnr:.2f} dB', f'{e_lo:.2f} – {e_hi:.2f}', counts=counts, kernel="fixed"))
LRS = [("N1_feat_x3_120k", "特征学习率 ×3(0.0075 → 0.0225)", "0.0:无变化", None),
       ("N2_feat_x0.3_120k", "特征学习率 ×0.3(→ 0.00225)", "<b>+0.47</b>,刚好超过种子噪声 0.42;压缩后 9.3 MB(其余 120k 步运行 7–8 MB):更低的特征学习率让更多孩子存活", None),
       ("N3_offset_x3_120k", "offset 学习率 ×3(0.01 / 1e-4 → 0.03 / 3e-4)", "+0.30,噪声内但方向为正", None),
       ("N4_offset_x0.3_120k", "offset 学习率 ×0.3(→ 0.003 / 3e-5)", "<b>−1.11</b>,明确有害:offset 学习率的方向是“再高”", None),
       ("N5_mlp_x3_120k", "三个 MLP 学习率 ×3(shape 0.012、flow 0.003、opacity 0.006 / 6e-5)", "+0.28,噪声内", None),
       ("N6_mlp_x0.3_120k", "三个 MLP 学习率 ×0.3", "", None)]
for name, what, note, _ in LRS:
    run = "next/" + name
    if run in runs:
        d = runs[run]["psnr"] - runs["scale/G5_K10_150k_120k_decay"]["psnr"]
        parts.append(s4d(name, run, "G5 + " + what, pill("final" if d > 0.42 else "neg" if d < -0.42 else "final", "当前结论" if d > 0.42 else "负面记录" if d < -0.42 else "噪声内"), note=note or f"{d:+.2f}"))
    else:
        parts.append(row(name, "G5 + " + what, f"output/scaffold4d/next/{name}/", "120,000 步", "none", pill("run"), "—", counts="—"))
parts.append('''</tbody></table></div>
<div class="callout"><b>结论:</b>四个方向里三个关掉——σ 上限(L1)、分组重建(E)、按误差采样(M1)都不涨 PSNR,其中分组在同等计算量下差 6 dB,是对“4D 共享有没有用”的最直接回答:有用,而且是 6 dB 级别的。上游七组学习率在这个数据上并未调优:offset 学习率明显偏低(×0.3 → −1.1,×3 → +0.3),特征学习率略偏高(×0.3 → +0.5),MLP 学习率也偏低(×3 → +0.3,×0.3 → −0.5)。单项效应都在 0.3–0.5 dB,与噪声同量级,但三组的方向都一致且两侧对称。<b>正在跑</b>:O1 = H4 + 特征 ×0.3 + offset ×3 + MLP ×3(480k 步)和 O2 = G5 + offset ×10(120k 步);如果 O1 ≥ +1 dB 则把这组学习率写成 ESVR 配方。另一个观察:时间门在 53k 步时中位半宽还有 12–30 帧,480k 步才收到 3.5 帧——σ 学得慢,MLP 学习率 ×3 略正可能与此有关。</div>
</section>
<section class="method"><div class="mh"><span class="letter">E</span><h2>实现与上游的对应与偏离(审计 2026-09-22)</h2></div>
<div class="lineage">基元、核、监督来自 ESVR;锚点/特征/子椭球结构、线性运动、广义高斯时间门、7 组优化器、dynamic-aware 生长来自 4D-Scaffold-GS 官方代码。不是逐行等价移植;每一处偏离及其实测影响如下(详见 <code>docs/ESVR_SCAFFOLD4D.md</code>)</div>
<div class="tablewrap"><table class="dev"><thead><tr><th>偏离上游之处</th><th>原因</th><th>实测</th></tr></thead><tbody>
<tr><td>opacity 头用 sigmoid,而非 tanh + “&gt;0 掩码”</td><td>tanh + 零特征初始化在 ~50 步内让全部孩子转负、永久死亡</td><td>没有它无法训练</td></tr>
<tr><td>上游 <code>min_opacity</code> 锚点剪枝:机制与调度同上游(每 100 步、1,500 步起、到 <code>--densify-until</code> = 总步数/2 为止,之后不再剪也不再长),但触发情形不同</td><td>K 个 sigmoid 强度之和要低于每次 0.005 才剪;上游 tanh 给出精确的 0。<b>2026-09-22 修正</b>:此前写“几乎不触发、每次运行只删 1–70 个”是错的——那只对 40k / 60k 上限的运行成立</td><td>H4(150k 上限)整个生长窗口内按此规则删了 50,879 个锚点(占新生 194,879 的 26%,集中在上限填满后的 20k–120k 步:新长出的锚点孩子全黑 → 被删 → 空位再长,形成周转);G5 41,111;I3 40,458。40k / 60k 上限:K4 仅 425(另有 <code>--prune-dead</code> 死锚点规则 26,190),T2 587。两种规则都不改变 PSNR(D.4、D.8)</td></tr>
<tr><td>生长统计用 3D 中心梯度代替 2D 屏幕梯度</td><td>没有相机;这是 ESVR 自己 <code>add_densification_stats</code> 的量</td><td>阈值按 3D 梯度 p90 校准(修复核 1.2e-3)</td></tr>
<tr><td>“观测足够”判据按块占体积比例缩放</td><td>孩子只在其 64³ 块被采到时才被观测</td><td>—</td></tr>
<tr><td>锚点数上限 + 按投票强度排序准入</td><td>上游无上限</td><td>40k / 100k / 150k / 300k 均已测</td></tr>
<tr><td>所有格子层级每轮都执行(<code>--grow-upstream-level-skip</code> 恢复上游规则)</td><td>上游只在粗层级本轮长出锚点时才执行细层级;09-22 审计才发现</td><td>未测</td></tr>
<tr><td>锚点时间均匀随机初始化(上游全在 t₀)</td><td>体数据没有 SfM 帧</td><td><code>--init data</code>(按 GT 采样)vs 随机:0.0 dB</td></tr>
<tr><td>子尺度 0.5 体素下限;σ ∈ [1e-2, 1e2];offset ∈ [−10, 10]</td><td>float32 / CUDA 失效模式(NaN、亚体素孩子零梯度)</td><td>下限:去掉 −3.3 dB</td></tr>
<tr><td>时间截断 1e-3</td><td>效率;低于核的 1/255</td><td>—</td></tr>
<tr><td>体积正则作用于全部时间激活的孩子(上游作用于被渲染的)</td><td>mini-batch 数据项 + 全局正则</td><td>只作用可见孩子 0.0 dB;去掉 −0.6 dB</td></tr>
<tr><td><code>--final-lr-scale 0.01</code>(ESVR 配方,<b>非默认</b>)</td><td>上游恒定的 4 组学习率在块监督下后期震荡 ±0.7 dB;ESVR 自身学习率也衰减</td><td>150k 锚点 +5 dB;默认保持 1.0 使 argparse 默认 = 论文值</td></tr>
<tr><td>每帧优化预算约为静态 ESVR 的 1/12</td><td>90 帧共享一次训练</td><td>静态天花板研究(A 节)</td></tr>
</tbody></table></div>
<div class="callout">逐段核对结论(代码 vs <code>generate_neural_gaussians</code> / <code>training_statis</code> / <code>anchor_growing</code> / <code>adjust_anchor</code> / <code>train.py</code>,以及 ESVR 的 <code>train_chunk.py</code> / <code>gaussian_model.py</code> / <code>forward.cu</code>):4D home 位置、线性运动、时间门(β=4)、时不变尺度旋转、7 组学习率与 <code>expon_lr</code>、增密统计权重、三层级格子生长、特征 max-pool、Adam eps、损失三项、SSIM(= 各 x 切片 2D SSIM 均值,与 ESVR 相同)、块坐标约定、评估口径(预测本在 [0,1] 内,裁剪 Δ=0)均一致;CUDA 位置梯度经有限差分验证(比值 1.00);G5/H4/I3 重载逐字节一致;压缩文件可被 <code>inference_scaffold4d.py</code> 读取并复现(H4 第 46 帧 39.34 vs 39.34 dB)。<br><br><b>独立审阅(2026-09-22,另一个不带本会话先验的代理逐行对照上游代码):</b>未发现影响任何已报数字的错误;它用单体素定位验证了块坐标约定与分块评估逐位一致,核对了扁平索引记账、优化器状态切片、生长/剪枝与上游的逐项对应、<code>expon_lr</code>、压缩位掩码往返、数据轴序与时间归一化。提出的 9 项健壮性/文档问题(缺失参数被 <code>strict=False</code> 掩盖、<code>--grow-depth 4</code> 格子尺寸为 0、<code>--init data</code> 的最近邻只用 xyz 且 N²、时间门测试靠浮点下溢通过、文档数字漂移等)已全部修复并提交。<b>本页的每一行结果数字都经过重新计算,来源仅为各运行的完整 90 帧 <code>evaluation/metrics.json</code>。</b></div>
<div class="callout"><b>关于“无用高斯”与模型大小(2026-09-22):</b>这是表征层面的结构性现象,不是生长/剪枝代码的错误。每个锚点固定生成 K=10 个孩子,不需要的孩子只能被优化器“关掉”——上游用 tanh 掩码免费关闭(但其 offset 同样占存储,上游论文报的 storage 也包含全部 K 个 offset);在 ESVR 的加法渲染下只能压暗/压小,结果相同:存储里 60–90% 是死孩子。训练时删掉它们的两种途径都已测过:死锚点回收(T2)让新生锚点以同样速度死去,有用率不变;上游 <code>min_opacity</code> 在 sigmoid 下失效。<b>有效的做法是训练后压缩</b>(精确删暗孩子 + 位掩码 + fp16,6–7×,PSNR 不变),以及用更小的锚点上限换更长训练(J1:40k 锚点 / 480k 步 = 36.9 dB,压缩后 2.6 MB)。“压缩前就小”的路线 K4(60k 上限 + 回收,480k 步)= 38.20 dB,压缩前 20 MB、压缩后 4.0 MB:落在 J1(40k)– H4(150k)同一条预算曲线上,回收本身没有额外增益(D.8)。</div>
</section>
''')
parts.append(f'''<footer>评测协议:全部实验用全 90 帧训练(无留出),PSNR 为归一化 [0,1] 值域下的 3D 体素 PSNR 逐帧均值;标注"当前结论"的运行已做 checkpoint 重载逐字节校验。规模列:训练时长取自各运行的 <code>train.jsonl</code>(RTX 5090,多数运行 2–3 个并发共享 GPU);模型大小 = 推理所需权重张量的 float32 字节数(Scaffold4D:每锚点 83 float ≈ 332 B,三个 MLP 共约 28 KB),不含 Adam 状态、未做量化/熵编码。可视化列点击在页内打开各运行自己生成的 <code>quality_curves.png</code>(逐帧 PSNR/MAE 曲线)、<code>slice_comparison.png</code>(六个时刻的中央 Z 切片 GT/预测/误差)和 <code>training_frames_dvr.gif</code>(90 帧体渲染四联:GT | 预测 | 体素级绝对误差 | 渲染图像级绝对误差,误差面板 inferno 配色、上限 0.1)。动图为随页发布方便转成了 MP4(10 fps,H.264),原始 GIF 在各运行目录;只有各方法的关键运行带动图,其余运行的 GIF 同样在目录里。完整记录见各目录 <code>RESULTS.md</code>、<code>docs/ESVR_SCAFFOLD4D.md</code>、<code>output/scaffold4d/ablation/RESULTS.md</code>。</footer>
</div>
<script>
const P={json.dumps(points, ensure_ascii=False)};
const C={{static:"var(--c-static)",deform:"var(--c-deform)",esvr4d:"var(--c-esvr4d)",s4d_old:"var(--c-s4d)",s4d_bad:"var(--c-s4d)",s4d_fix:"var(--c-s4d)",s4d_compact:"var(--good)"}};
const W=920,H=340,L=52,R=16,T=14,Bm=40;
const xs=P.map(p=>p.x),ys=P.map(p=>p.y);
const xmin=1,xmax=700,ymin=20,ymax=51;
const X=v=>L+(Math.log10(v)-Math.log10(xmin))/(Math.log10(xmax)-Math.log10(xmin))*(W-L-R);
const Y=v=>T+(ymax-v)/(ymax-ymin)*(H-T-Bm);
let s=`<svg viewBox="0 0 ${{W}} ${{H}}" width="100%" style="max-width:${{W}}px;display:block;overflow:visible">`;
for(const g of [25,30,35,40,45,50]){{s+=`<line x1="${{L}}" x2="${{W-R}}" y1="${{Y(g)}}" y2="${{Y(g)}}" stroke="var(--line)"/><text x="${{L-8}}" y="${{Y(g)+4}}" text-anchor="end">${{g}} dB</text>`;}}
for(const g of [1,2,5,10,20,50,100,200,500]){{s+=`<line y1="${{T}}" y2="${{H-Bm}}" x1="${{X(g)}}" x2="${{X(g)}}" stroke="var(--line)" stroke-dasharray="2 4"/><text x="${{X(g)}}" y="${{H-Bm+16}}" text-anchor="middle">${{g}}</text>`;}}
s+=`<text x="${{(L+W-R)/2}}" y="${{H-6}}" text-anchor="middle">模型大小 (MB, log)</text>`;
s+=`<line x1="${{L}}" x2="${{W-R}}" y1="${{Y(50)}}" y2="${{Y(50)}}" stroke="var(--accent)" stroke-dasharray="6 4" opacity=".7"/><text x="${{W-R-4}}" y="${{Y(50)-5}}" text-anchor="end" style="fill:var(--accent)">目标 50 dB</text>`;
s+=`<line x1="${{L}}" x2="${{W-R}}" y1="${{Y(28.80)}}" y2="${{Y(28.80)}}" stroke="var(--ink3)" stroke-dasharray="2 4"/><text x="${{L+6}}" y="${{Y(28.80)-5}}" style="fill:var(--ink3)">零模型:复制上一帧,89 对均值 28.80 dB(2026-09-22 修正;此前标的 30.39 只是第 1→2 帧一对)</text>`;
for(const p of P){{const r=(p.s==="s4d_fix"||p.s==="s4d_compact")?7:(p.s==="static"?6:5);const op=p.f||p.s==="s4d_old"?.35:1;const st=(p.s==="s4d_fix"||p.s==="s4d_compact")?`stroke="var(--ink)" stroke-width="1.5"`:"";
s+=`<circle class="pt" cx="${{X(p.x)}}" cy="${{Y(p.y)}}" r="${{r}}" fill="${{C[p.s]}}" opacity="${{op}}" ${{st}} data-l="${{p.l}} · ${{p.x}} MB · ${{p.y.toFixed(2)}} dB"/>`;}}
s+=`</svg><div class="tip" id="tip"></div>`;
const el=document.getElementById("chart");el.style.position="relative";el.innerHTML=s;
const tip=document.getElementById("tip");
el.querySelectorAll(".pt").forEach(c=>{{c.addEventListener("mousemove",e=>{{tip.style.display="block";tip.textContent=c.dataset.l;const b=el.getBoundingClientRect();tip.style.left=(e.clientX-b.left+12)+"px";tip.style.top=(e.clientY-b.top-28)+"px";}});c.addEventListener("mouseleave",()=>tip.style.display="none");}});
const lb=document.createElement("div");lb.id="lb";lb.innerHTML='<img alt="" hidden><video loop muted autoplay playsinline controls hidden></video><div class="cap"></div><div class="hint">点击空白处或按 Esc 关闭</div>';document.body.appendChild(lb);
const lbImg=lb.querySelector("img"),lbVid=lb.querySelector("video"),lbCap=lb.querySelector(".cap");
function closeLb(){{lb.classList.remove("open");lbImg.src="";lbVid.pause();lbVid.removeAttribute("src");lbVid.load();lbImg.hidden=true;lbVid.hidden=true;}}
document.querySelectorAll(".viz").forEach(b=>b.addEventListener("click",()=>{{
  lbCap.textContent=b.dataset.cap;
  if(b.dataset.video){{lbVid.src=b.dataset.video;lbVid.hidden=false;lbImg.hidden=true;lbVid.play().catch(()=>{{}});}}
  else{{lbImg.src=b.dataset.full;lbImg.hidden=false;lbVid.hidden=true;}}
  lb.classList.add("open");}}));
lb.addEventListener("click",e=>{{if(e.target!==lbImg&&e.target!==lbVid)closeLb();}});
document.addEventListener("keydown",e=>{{if(e.key==="Escape")closeLb();}});
</script>''')
open(os.path.join(D, "index.html"), "w").write("\n".join(parts))
print("wrote index.html", sum(len(p) for p in parts), "chars;", len(thumbs), "thumbs")
