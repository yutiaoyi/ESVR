"""Experiment lineage graph for the report page: which run is a variant of which.

Each node is (id, parent, name, detail, kind[, note]); the optional note is a
third line for a cross-reference that is not a parent-child edge. The name says what the run
changes relative to its parent (one variable, unless the name lists more);
the detail is its 90-frame vorts result unless marked otherwise. Layout is a
left-to-right tidy tree (leaves in listed order, parents centred on children).
Kinds: root, void (unfixed kernel, discarded), sup (valid, superseded),
add (valid additive), norm (normalized composition), best, frame1 (single-frame
control only), run (running / queued).
"""
import html

NODES = [
    # static ESVR
    ("esvr", None, "ESVR(静态,逐帧)", "原论文 · 加法合成", "root"),
    ("pf_old", "esvr", "逐帧 90 帧 · 未修复核", "41.22 dB · 93.8 MB", "void"),
    ("S1b", "esvr", "S1b · ESVR 自带配方", "41.21 dB · 93.8 MB", "add"),
    ("S1d", "S1b", "S1d · +数据初始化", "第 1 帧 41.98(S1b 41.81)", "frame1"),
    ("S1n", "S1b", "S1n · +归一化", "第 1 帧 45.02 · 83% 体素在分母下限", "frame1"),
    ("S1nr", "S1n", "S1nr · −强度重置", "第 1 帧 53.28 · 90 帧进行中", "run"),
    ("S5b", "esvr", "S5b · 天花板配方", "49.52 dB · 555.6 MB", "add"),
    ("S5d", "S5b", "S5d · +数据初始化", "第 1 帧 48.67(S5b 49.63)", "frame1"),
    ("S5n", "S5b", "S5n · +归一化", "第 1 帧 53.25 · 87% 体素在分母下限", "frame1"),
    ("S5nr", "S5n", "S5nr · −强度重置", "90 帧排队中", "run"),
    ("sn", "esvr", "单帧归一化测试", "ESVR 核 + 简化优化器 · 第 1 帧 55.75", "frame1"),
    ("N4A", "sn", "N4A · 显式 4D + 归一化", "53.25 dB · 27.2 MB", "norm",
     "表示 = W 系列去掉锚点/MLP · 配方不同"),
    ("N4B", "N4A", "N4B · 初始时间半宽 4 帧", "53.07 dB · 27.2 MB", "norm"),
    ("PN", "sn", "PN21 / PN10 / PN6 · 逐帧 90 帧", "51.05 / 43.70 / 38.51 dB", "norm"),
    # other dynamic families
    ("deform", None, "Deformable 3DGS → ESVR", "B 节 · 最佳 28.90 dB", "sup"),
    ("esvr4d", None, "4DGS → ESVR4D", "C 节 · 最佳 28.77 dB", "sup"),
    # Scaffold4D
    ("sc", None, "4D-Scaffold-GS → ESVR", "D 节 · Scaffold4D", "root"),
    ("sc_old", "sc", "早期版本 + 消融 A–K", "未修复核 · 作废", "void"),
    ("F", "sc", "F_default · 修复核", "31.91 dB · 13.3 MB", "sup"),
    ("Fvar", "F", "cap100k / fixed6000 / floor_old", "32.95 / 29.39 / 28.97 dB", "sup"),
    ("G2", "F", "G2 · K=5 · 上限 24 万", "32.68 dB · 49.2 MB", "sup"),
    ("G1", "F", "G1 · 上限 15 万 · 12 万步", "32.99 dB · 49.8 MB", "sup"),
    ("G5", "G1", "G5 · +学习率衰减 0.01", "38.04 dB · 47.3 MB", "sup"),
    ("Nlr", "G5", "N1–N6 · 学习率单变量", "36.93 – 38.51 dB", "sup"),
    ("O", "Nlr", "O1 / O2 · 组合与验证", "38.41 / 35.78 dB", "sup"),
    ("H4", "G5", "H4 · 48 万步(H1–H6)", "39.43 dB · 49.8 MB", "sup"),
    ("I", "H4", "I2 / I3 / J1 · 步数与上限", "39.33 / 39.73 / 36.94 dB", "sup"),
    ("K", "H4", "K1 / K3 / K4", "41.20(只训 10 帧)/ 38.94 / 38.20", "sup"),
    ("K2", "H4", "K2 · 块 128³ · 6 万步", "33.32 dB · 4.1 MB", "sup"),
    ("LMQ", "H4", "L1 / M1 / Q1", "38.96 / 39.15 / 39.09 dB", "sup"),
    ("E", "H4", "E1–E9 · 分 9 组各自训练", "33.31 dB · 50.1 MB", "sup"),
    ("P1b", "H4", "P1b · 块 128³ + 阈值 3e-4", "40.17 dB · 29.0 MB", "add"),
    ("P2RS", "P1b", "P2 / R1 / S1", "40.50 / 40.26 / 39.87 dB", "add"),
    ("P3", "P1b", "P3 · 阈值 1e-4", "40.59 dB · 49.8 MB", "add"),
    ("W1", "P3", "W1 · +归一化", "54.66 dB · 49.8 MB", "norm"),
    ("W2", "W1", "W2 · +数据初始化", "54.94 dB · 39.8 MB(fp16 20.1)", "best",
     "去掉锚点/MLP 的显式版 ≈ N4A"),
    ("W3", "W2", "W3 · 步数减半(12 万)", "52.98 dB · 38.3 MB", "norm"),
    ("W4", "W2", "W4 · 锚点上限 6 万", "进行中", "run"),
    ("X1", "P3", "X1 · 显式加法 4D 蒸馏", "40.96 dB · 11.6 MB", "add"),
    ("U", "P3", "U1–U3 · 单帧切片解放", "第 1 帧 44.38", "frame1"),
    ("J1", "P1b", "jet J1 · P1b 配方 × 体积", "jet 42.81 dB · 132.8 MB", "add"),
    ("J2b", "J1", "jet J2b · +归一化 +数据初始化", "进行中", "run"),
]

BW, BH, COLW, ROWH, PAD = 206, 38, 232, 56, 12
BH_NOTE = 52
STYLE = {  # fill, stroke, width, dash, opacity
    "root": ("var(--accent-soft)", "var(--accent)", 1.5, "", 1),
    "void": ("var(--surface)", "var(--line2)", 1, "3 3", .5),
    "sup": ("var(--surface)", "var(--line2)", 1, "", 1),
    "add": ("var(--surface)", "var(--ink2)", 1.3, "", 1),
    "norm": ("var(--norm-soft)", "var(--c-norm)", 1.5, "", 1),
    "best": ("var(--norm-soft)", "var(--c-norm)", 3, "", 1),
    "frame1": ("var(--surface)", "var(--ink3)", 1, "1.5 2.5", 1),
    "run": ("var(--surface)", "var(--warn)", 1.5, "6 4", 1),
}


def _layout():
    kids = {}
    for n in NODES:
        kids.setdefault(n[1], []).append(n[0])
    pos, row = {}, [0]

    def place(i, depth):
        ch = kids.get(i, [])
        for c in ch:
            place(c, depth + 1)
        if ch:
            y = (pos[ch[0]][1] + pos[ch[-1]][1]) / 2
        else:
            y = row[0] * ROWH
            row[0] += 1
        pos[i] = (depth * COLW, y)

    for r in kids[None]:
        place(r, 0)
        row[0] += 0.4   # gap between families
    return pos


def _svg():
    pos = _layout()
    by = {n[0]: n for n in NODES}
    w = max(x for x, _ in pos.values()) + BW + 2 * PAD
    h = max(y for _, y in pos.values()) + BH_NOTE + 2 * PAD
    out = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}" role="img" aria-label="实验谱系图">']
    for i, (x, y) in pos.items():
        par = by[i][1]
        if par:
            hp = BH_NOTE if len(by[par]) > 5 else BH
            px, py = pos[par]
            hc = BH_NOTE if len(by[i]) > 5 else BH
            x0, y0, x1, y1 = px + BW + PAD, py + hp / 2 + PAD, x + PAD, y + hc / 2 + PAD
            xm = (x0 + x1) / 2
            out.append(f'<path d="M{x0:.1f},{y0:.1f} C{xm:.1f},{y0:.1f} {xm:.1f},{y1:.1f} {x1:.1f},{y1:.1f}" fill="none" stroke="var(--line2)" stroke-width="1.3"/>')
    for i, (x, y) in pos.items():
        _, _, name, detail, kind, *note = by[i]
        bh = BH_NOTE if note else BH
        fill, stroke, sw, dash, op = STYLE[kind]
        da = f' stroke-dasharray="{dash}"' if dash else ""
        star = "★ " if kind == "best" else ""
        out.append(f'<g opacity="{op}"><title>{html.escape(name + " — " + detail + ("; " + note[0] if note else ""))}</title>'
                   f'<rect x="{x + PAD:.1f}" y="{y + PAD:.1f}" width="{BW}" height="{bh}" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{da}/>'
                   f'<text class="nm" x="{x + PAD + 8:.1f}" y="{y + PAD + 15:.1f}">{html.escape(star + name)}</text>'
                   f'<text class="dt" x="{x + PAD + 8:.1f}" y="{y + PAD + 30:.1f}">{html.escape(detail)}</text>'
                   + (f'<text class="dt" x="{x + PAD + 8:.1f}" y="{y + PAD + 45:.1f}" style="fill:var(--c-norm)">{html.escape(note[0])}</text>' if note else "")
                   + '</g>')
    out.append("</svg>")
    return "".join(out)


def _key(kind, label):
    fill, stroke, sw, dash, op = STYLE[kind]
    da = f"border-style:{'dashed' if dash else 'solid'};"
    return (f'<span><i style="display:inline-block;width:18px;height:11px;border-radius:3px;vertical-align:-1px;margin-right:6px;'
            f'background:{fill};border:{max(sw, 1):.1f}px solid {stroke};{da}opacity:{op}"></i>{label}</span>')


LINEAGE = ('<div class="panel"><h2>实验谱系图</h2>'
           '<p class="lead">每条连线从“配置来源”指向“在它基础上改动得到的实验”;方框第一行写相对上一格改了什么(一次只改一个变量,除非列出多项),第二行是 vorts 90 帧平均 PSNR 与 fp32 模型大小(另有注明的除外)。'
           '同一方框里用斜杠并列的是同一父配置下的几个单变量兄弟实验。玫红色第三行是不属于父子关系的对应:N4A 的表示就是 W 系列去掉锚点、特征和共享 MLP 后的显式版(时间门 β=4、线性运动、权重 + 数值的归一化合成都相同),但它不是从 W 改出来的,训练配方沿用单帧归一化测试(只用 L1、固定 40 万基元不增密、余弦学习率、9 万步;W 为 0.8 L1 + 0.2 SSIM、锚点生长、24 万步),所以连线仍接在“单帧归一化测试”下。图较宽,可左右滚动;悬停方框看全文。</p>'
           '<div class="lin">' + _svg() + '</div>'
           '<div class="legend">' + "".join(_key(k, l) for k, l in (
               ("root", "方法来源"), ("add", "有效 · 加法合成"), ("norm", "归一化合成"), ("best", "当前最佳"),
               ("sup", "有效 · 已被后续取代"), ("frame1", "只做了第 1 帧对照"), ("run", "进行中 / 排队"), ("void", "未修复核 · 作废"))) +
           '</div></div>\n')
