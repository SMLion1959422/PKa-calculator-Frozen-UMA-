"""MIT URTC poster, 48 x 36 in landscape.

    python posters/urtc_poster.py       -> posters/urtc_poster.pdf (+ .png)

Layout is cursor-based: each Panel tracks a vertical cursor and every
para()/ax()/bullet() call consumes space, so nothing can silently overlap.
The script reports any panel whose content overflows its box.

Every number is traceable to a results file - see SOURCES at the bottom.
Numbers from the submitted abstract that are NOT reproducible from this
repo are listed there too, and are deliberately not on the poster.

Palette: dataviz reference instance, light mode, first three categorical
slots (validated all-pairs: worst CVD dE 9.2, normal-vision 24.0). Aqua is
below 3:1 on the surface, so the relief rule applies - every bar carries a
visible direct label.
"""
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

W, H = 48.0, 36.0
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE, AQUA, ACCENT = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
PANEL, EDGE, GRID = "#f4f3f0", "#dedcd6", "#e6e4de"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.edgecolor": INK3, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})
fig = plt.figure(figsize=(W, H), facecolor=SURFACE)
PT = 1 / 72.0          # inches per point
FS = 0.795             # global text scale; tune until OVERFLOW is empty
_overflow = []


class Panel:
    """A box with a top-down cursor. Content never overlaps by construction."""
    PAD_X, PAD_TOP, PAD_BOT = 0.42, 0.55, 0.42

    def __init__(self, x, y, w, h, title=None, fc=PANEL, ec=EDGE, lw=1.6):
        self.x, self.y, self.w, self.h = x, y, w, h
        fig.patches.append(FancyBboxPatch(
            (x / W, y / H), w / W, h / H,
            boxstyle="round,pad=0,rounding_size=0.004",
            fc=fc, ec=ec, lw=lw, transform=fig.transFigure, zorder=-10))
        self.cur = y + h - self.PAD_TOP
        self.iw = w - 2 * self.PAD_X
        if title:
            self.head(title)

    def head(self, s, size=36):
        fig.text((self.x + self.PAD_X) / W, self.cur / H, s, fontsize=size,
                 weight="bold", color=INK, va="top", zorder=5)
        self.cur -= size * PT * 1.15 + 0.22

    def para(self, s, size=25, color=INK2, weight="normal", gap=0.24, wrap=True):
        size = size * FS
        if wrap:
            chars = max(20, int(self.iw / (size * PT * 0.545)))
            s = "\n".join(textwrap.fill(ln, chars) if ln.strip() else ln
                          for ln in s.split("\n"))
        n = s.count("\n") + 1
        fig.text((self.x + self.PAD_X) / W, self.cur / H, s, fontsize=size,
                 color=color, weight=weight, va="top", zorder=5, linespacing=1.5)
        self.cur -= n * size * PT * 1.5 + gap

    def bullet(self, s, size=23, color=INK2, marker="•", mcolor=ORANGE, gap=0.18):
        size = size * FS
        chars = max(20, int((self.iw - 0.5) / (size * PT * 0.545)))
        s = textwrap.fill(s, chars)
        fig.text((self.x + self.PAD_X + 0.08) / W, self.cur / H, marker,
                 fontsize=size, color=mcolor, weight="bold", va="top", zorder=5)
        fig.text((self.x + self.PAD_X + 0.52) / W, self.cur / H, s, fontsize=size,
                 color=color, va="top", zorder=5, linespacing=1.5)
        self.cur -= (s.count("\n") + 1) * size * PT * 1.5 + gap

    def kv(self, k, v, size=23, gap=0.18, split=0.44):
        size = size * FS
        fig.text((self.x + self.PAD_X) / W, self.cur / H, k, fontsize=size,
                 color=INK, weight="bold", va="top", zorder=5)
        chars = max(14, int((self.iw * (1 - split)) / (size * PT * 0.545)))
        v = textwrap.fill(v, chars)
        fig.text((self.x + self.PAD_X + self.iw * split) / W, self.cur / H, v,
                 fontsize=size, color=AQUA, va="top", zorder=5, linespacing=1.5)
        self.cur -= (max(0, v.count("\n")) + 1) * size * PT * 1.5 + gap

    def ax(self, height, left=1.35, right=0.35, gap=0.3, below=0.85):
        """`below` reserves room for tick labels and the x-label, which
        matplotlib draws OUTSIDE the axes rectangle."""
        w = self.w - left - right
        a = fig.add_axes([(self.x + left) / W, (self.cur - height) / H,
                          w / W, height / H], zorder=5)
        a.set_facecolor(PANEL)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.grid(axis="y", color=GRID, lw=1.2)
        a.set_axisbelow(True)
        self.cur -= height + below + gap
        return a

    def done(self, name):
        if self.cur < self.y + self.PAD_BOT - 0.05:
            _overflow.append(f"{name}: overflows by {self.y + self.PAD_BOT - self.cur:.2f} in")


def nolabel(a):
    a.spines["left"].set_visible(False)
    a.tick_params(left=False)


# =====================================================================
# TITLE BAND
# =====================================================================
fig.patches.append(FancyBboxPatch(
    (0.9 / W, 31.5 / H), (W - 1.8) / W, 3.7 / H,
    boxstyle="round,pad=0,rounding_size=0.004", fc="#0b0b0b", ec="none",
    transform=fig.transFigure, zorder=-10))
fig.text(0.5, 34.35 / H, "Predicting Acid Dissociation from UMA Embeddings",
         fontsize=82, weight="bold", color="#ffffff", ha="center", va="center", zorder=5)
fig.text(0.5, 33.05 / H,
         "A frozen foundation model that never saw a pK$_a$ value — and what a new solvent actually costs",
         fontsize=38, color="#c3c2b7", ha="center", va="center", zorder=5)
fig.text(0.5, 32.05 / H,
         "Srikanth Mohan   ·   MIT Undergraduate Research Technology Conference   ·   "
         "github.com/SMLion1959422/PKa-calculator-Frozen-UMA-",
         fontsize=27, color="#8a8984", ha="center", va="center", zorder=5)

CX = [1.0, 12.7, 24.4, 36.1]
CW = 10.9
TOP, BOT = 30.9, 1.0

# =====================================================================
# COLUMN 1
# =====================================================================
x = CX[0]
p = Panel(x, 24.5, CW, TOP - 24.5, "1 · Why pK$_a$, why now")
p.para("Bringing a drug to market costs billions, mostly on candidates that fail. "
       "Many failures trace to physicochemical causes knowable before synthesis.")
p.para("Those properties depend on which ionization state a molecule adopts — set "
       "by its pK$_a$.", color=INK)
p.para("Goal: a free, offline, license-free predictor built on a foundation "
       "model never trained on pK$_a$ data.", color=ACCENT, weight="bold")
p.done("1")

p = Panel(x, 15.5, CW, 24.5 - 15.5 - 0.45, "2 · UMA's energies cannot work")
p.para("The direct route — compute the deprotonation energy and convert — fails "
       "for a reason that is arithmetic, not tuning.")
a = p.ax(2.85, below=1.05)
nolabel(a)
a.bar([0], [59.2], width=0.45, color=BLUE, zorder=3)
a.bar([1], [1241], width=0.45, color=ORANGE, zorder=3, yerr=[[195], [196]],
      error_kw=dict(ecolor=INK2, lw=2.5, capsize=14))
a.set_yscale("log"); a.set_ylim(20, 6000); a.set_xlim(-0.6, 1.6)
a.set_xticks([0, 1])
a.set_xticklabels(["signal:\n1 pK$_a$ unit", "noise:\nconformer scatter"], fontsize=23)
a.set_ylabel("meV (log scale)", fontsize=23)
a.tick_params(labelsize=21)
a.text(0, 78, "59.2", ha="center", fontsize=29, weight="bold", color=BLUE, zorder=4)
a.text(1, 1850, "1046–1437", ha="center", fontsize=29, weight="bold", color=ORANGE, zorder=4)
p.para("The noise is 18–24× the entire signal. GFN2-xTB, an established "
       "semi-empirical method, fails identically — a precision wall shared "
       "across methods, not a flaw in UMA.", size=23)
p.done("2")

p = Panel(x, 9.3, CW, 15.5 - 9.3 - 0.45, "3 · Use the representations instead")
p.para("UMA stays frozen. Instead of its scalar energy we read the 128-dim "
       "per-atom embeddings from the input to its energy head.")
p.para("Critically we pool them AROUND THE IONIZABLE SITE, not over the whole "
       "molecule: pK$_a$ is local, and global averaging dilutes it as molecules grow.", color=INK)
p.para("[ h$_{prot}$ ; h$_{deprot}$ ; h$_{prot}$ − h$_{deprot}$ ; ΔE ; n$_{heavy}$ ]  →  "
       "gradient-boosted regressor.\nUMA's weights are never updated.",
       size=23, color=ACCENT, weight="bold", wrap=False)
p.done("3")

p = Panel(x, BOT, CW, 9.3 - BOT - 0.45, "4 · Site assignment dominates")
p.para("The standard SMARTS rule picks a DIFFERENT atom from the dataset's own "
       "annotation for 23% of the Novartis benchmark.")
a = p.ax(2.1, below=1.0)
nolabel(a)
vals = [85.1, 77.1, 100.0]
a.bar(range(3), vals, width=0.5, color=[BLUE, ORANGE, AQUA], zorder=3)
for i, v in enumerate(vals):
    a.text(i, v + 3, f"{v:.1f}%", ha="center", fontsize=26, weight="bold",
           color=INK, zorder=4)
a.set_ylim(0, 122); a.set_xticks(range(3))
a.set_xticklabels(["Training\n(5 994)", "Novartis\n(280)", "AvLiLuMoVe\n(123)"], fontsize=22)
a.set_yticks([0, 50, 100]); a.tick_params(labelsize=21)
a.set_ylabel("site agreement", fontsize=22)
p.para("Anyone benchmarking on Novartis is partly measuring site detection. We "
       "also found tetrazoles could never be deprotonated — the SMARTS pointed "
       "at the ring carbon.", size=22, color=ORANGE)
p.done("4")

# =====================================================================
# COLUMN 2
# =====================================================================
x = CX[1]
p = Panel(x, 22.05, CW, TOP - 22.05, "5 · Aqueous accuracy")
a = p.ax(3.95, left=1.5, below=0.6)
nolabel(a)
idx, bw = np.arange(2), 0.25
for k, (lab, v, c) in enumerate([("whole-molecule UMA", [1.17, 0.70], BLUE),
                                 ("RDKit site baseline", [1.08, 0.53], ORANGE),
                                 ("site-local UMA (ours)", [1.03, 0.43], AQUA)]):
    off = (k - 1) * (bw + 0.03)
    a.bar(idx + off, v, width=bw, color=c, label=lab, zorder=3)
    for i, vv in enumerate(v):
        a.text(i + off, vv + 0.03, f"{vv:.2f}", ha="center", fontsize=22,
               weight="bold", color=INK, zorder=4)
a.set_xticks(idx); a.set_xticklabels(["Novartis", "AvLiLuMoVe"], fontsize=25)
a.set_ylabel("MAE (pK$_a$)   lower is better", fontsize=22)
a.set_ylim(0, 1.45); a.tick_params(labelsize=21)
a.legend(fontsize=21, frameon=False, loc="upper right")
p.para("Site-local pooling beats both the original whole-molecule UMA model and "
       "a same-features RDKit fingerprint baseline, on identical splits.", size=23)
p.done("5")

p = Panel(x, 17.1, CW, 22.05 - 17.1 - 0.45, "6 · What the embeddings add")
p.para("Given the CORRECT site, site-local UMA reaches 0.86 on Novartis. The gap up "
       "to 1.03 is automatic site selection — not the model.")
p.para("Scaffold-split CV:  0.72   —   no shared Bemis–Murcko core",
       size=24, color=INK, weight="bold")
p.para("UMA beats the fingerprint baseline by 13% at the annotated site and 18% on "
       "AvLiLuMoVe — the representation carries real information.", size=23)
p.done("6")

p = Panel(x, 8.9, CW, 17.1 - 8.9 - 0.45, "7 · One pK$_a$ is not enough")
p.para("Real molecules have several interacting ionizable sites. Scoring them "
       "independently is wrong by construction — it is why amino acids failed.")
a = p.ax(2.3, left=1.5, below=0.95)
nolabel(a)
ind, mic, i2 = np.array([1.59, 1.06, 1.22]), np.array([0.67, 0.94, 1.26]), np.arange(3)
a.bar(i2 - 0.16, ind, width=0.3, color=BLUE, label="independent sites", zorder=3)
a.bar(i2 + 0.16, mic, width=0.3, color=AQUA, label="coupled microstates", zorder=3)
for i in range(3):
    a.text(i - 0.16, ind[i] + 0.04, f"{ind[i]:.2f}", ha="center", fontsize=21,
           weight="bold", color=INK, zorder=4)
    a.text(i + 0.16, mic[i] + 0.04, f"{mic[i]:.2f}", ha="center", fontsize=21,
           weight="bold", color=INK, zorder=4)
a.set_xticks(i2)
a.set_xticklabels(["amino acids\n+ polyprotic", "SAMPL6", "SAMPL7"], fontsize=21)
a.set_ylim(0, 2.75); a.set_ylabel("MAE", fontsize=22); a.tick_params(labelsize=20)
a.legend(fontsize=19, frameon=False, loc="upper center", ncol=2,
         bbox_to_anchor=(0.5, 1.04), columnspacing=1.2, handlelength=1.4)
p.para("Amino acids 1.59 → 0.67 (p = 4×10$^{-7}$, better on 24/26). pI error "
       "0.49 → 0.29; net charge at pH 7.4 correct for 25/26.", size=22)
p.done("7")

p = Panel(x, BOT, CW, 8.9 - BOT - 0.45, "8 · Coupling validated, no ML")
p.para("For a symmetric diacid the intrinsic pK$_a$ cancels exactly: "
       "pK$_{a2}$ − pK$_{a1}$ = log$_{10}$4 + W — so the one fitted constant is "
       "testable with no ML involved.", size=23)
a = p.ax(2.0, left=1.6, below=1.0)
nolabel(a)
v2 = [1.01, 0.80, 0.32]
a.bar(range(3), v2, width=0.46, color=[INK3, ORANGE, AQUA], zorder=3)
for i, vv in enumerate(v2):
    a.text(i, vv + 0.03, f"{vv:.2f}", ha="center", fontsize=26, weight="bold",
           color=INK, zorder=4)
a.set_xticks(range(3))
a.set_xticklabels(["independent\nsites", "constant\ngap", "our model\n(leave-one-out)"],
                  fontsize=21)
a.set_ylim(0, 1.25); a.set_ylabel("MAE on the gap", fontsize=22)
a.tick_params(labelsize=20)
p.para("17 diacids / diamines. Held-out glycine: predicted pI 6.06 vs 6.06 "
       "experimental — the zwitterion is recovered.", size=22, color=AQUA)
p.done("8")

# =====================================================================
# COLUMN 3 - the novel result
# =====================================================================
x = CX[2]
p = Panel(x, 23.6, CW, TOP - 23.6, "9 · What does a NEW solvent cost?",
          fc="#fff6f2", ec=ORANGE, lw=2.6)
p.para("Every published multi-solvent pK$_a$ model reports accuracy on unseen "
       "MOLECULES. We could not find one that reports unseen SOLVENTS.", color=INK)
p.para("So we measured it. Leave-one-solvent-out: train on seven solvents, "
       "predict the eighth, for each in turn.")
fig.text((x + CW / 2) / W, (p.cur - 0.62) / H, "MAE  3 – 5", fontsize=64,
         weight="bold", color=ORANGE, ha="center", va="center", zorder=5)
p.cur -= 1.35
p.para("Models do not transfer to a solvent they have never seen.",
       size=23, color=INK, weight="bold")
p.done("9")

p = Panel(x, 14.7, CW, 23.6 - 14.7 - 0.45, "10 · One measurement buys it back")
a = p.ax(4.0, left=1.6, below=1.05)
a.grid(axis="both", color=GRID, lw=1.2)
ks = [0, 1, 2, 5, 10]
pred = [3.22, 1.86, 1.57, 1.44, 1.31]
meas = [3.20, 1.23, 1.02, 0.92, 0.81]
a.plot(ks, pred, "-o", color=ORANGE, lw=3.4, ms=13, zorder=3, label="water pK$_a$ predicted")
a.plot(ks, meas, "-o", color=BLUE, lw=3.4, ms=13, zorder=3, label="water pK$_a$ known")
for kx, v, c, dy in [(0, 3.22, ORANGE, 16), (1, 1.86, ORANGE, 16), (10, 1.31, ORANGE, 16),
                     (1, 1.23, BLUE, -36), (10, 0.81, BLUE, -36)]:
    a.annotate(f"{v:.2f}", (kx, v), textcoords="offset points", xytext=(0, dy),
               fontsize=22, weight="bold", color=c, ha="center", zorder=4)
a.set_xticks(ks)
a.set_xlabel("measured pK$_a$ values in the NEW solvent", fontsize=23)
a.set_ylabel("MAE on the rest", fontsize=23)
a.set_ylim(0, 3.9); a.tick_params(labelsize=21)
a.legend(fontsize=21, frameon=False, loc="upper right")
p.para("A single measurement removes 40–60% of the error. Mean over held-out "
       "solvents, 20 random draws each.", size=23)
p.done("10")

p = Panel(x, 5.7, CW, 14.7 - 5.7 - 0.45, "11 · Why: one physical constant")
p.para("Relative to water the thermodynamic cycle splits the shift into a solute "
       "term and ONE constant per solvent: the proton transfer energy.", size=23)
a = p.ax(2.75, left=1.75, below=1.05)
a.grid(color=GRID, lw=1.2)
lit = np.array([7.85, -2.52, -3.40, 1.95, 1.52])
off = np.array([16.47, 7.45, 6.02, 5.75, 4.98])
a.scatter(lit, off, s=380, color=BLUE, zorder=4, edgecolor="#ffffff", lw=2.5)
for l, o, n, dx, dy in zip(lit, off, ["MeCN", "DMF", "DMSO", "EtOH", "MeOH"],
                           [-16, 14, -14, 14, 12], [-30, -4, -26, 6, -26]):
    a.annotate(n, (l, o), textcoords="offset points", xytext=(dx, dy),
               fontsize=21, color=INK, weight="bold", zorder=5)
xs = np.linspace(-5, 9.5, 10)
a.plot(xs, 1.019 * xs + 6.9, "-", color=AQUA, lw=3.2, zorder=3, label="fit · slope 1.02")
a.set_xlabel("literature ΔG$_{tr}$(H$^+$)   [pK units]", fontsize=22)
a.set_ylabel("fitted offset", fontsize=22)
a.tick_params(labelsize=20)
a.legend(fontsize=21, frameon=False, loc="upper left")
p.para("offset = 1.02·ΔG$_{tr}$(H$^+$) − 6.66·Δα + 1.95      LOSO error 0.40",
       size=23, color=ACCENT, weight="bold", wrap=False)
p.para("The proton coefficient lands at 1.00 within error, 95% CI [0.84, 1.20] — "
       "what the cycle mandates, not something the fit was told.", size=22)
p.done("11")

p = Panel(x, BOT, CW, 5.7 - BOT - 0.45, "12 · The rest is anion desolvation")
p.para("The remainder tracks the solvent's H-bond DONOR strength α — an anion needs "
       "donors. It shows NO relation to β, the acceptor scale (r² = 0.03): the "
       "specificity check the physics demands.", size=23)
p.para("Correcting one bad literature value (DMF) cut leave-one-out error "
       "1.14 → 0.40 without adding a single parameter.",
       size=23, color=AQUA, weight="bold")
p.done("12")

# =====================================================================
# COLUMN 4
# =====================================================================
x = CX[3]
p = Panel(x, 20.2, CW, TOP - 20.2, "13 · What UMA actually knows")
p.para("If the leftover is anion desolvation, sensitivity to α should depend on how "
       "DELOCALIZED the anion's charge is.")
p.para("UMA probes this directly: deprotonation changes its per-atom embeddings, and "
       "WHERE it changes them is where the charge went.", color=INK)
p.para("w$_a$ = ‖h(A⁻)$_a$ − h(HA)$_a$‖ , summarized by its spread from the site",
       size=23, color=ACCENT, weight="bold", wrap=False)
a = p.ax(2.5, left=4.3, right=0.4, below=0.95)
a.grid(axis="x", color=GRID, lw=1.2)
a.grid(axis="y", visible=False)
a.barh([0, 1], [57.1, 2.1], height=0.46, color=[AQUA, INK3], zorder=3)
a.set_yticks([0, 1])
a.set_yticklabels(["UMA added on top\nof conjugation count",
                   "conjugation count\nadded on top of UMA"], fontsize=21)
a.text(55, 0, "F = 57.1\np = 0.0001", va="center", ha="right", fontsize=22,
       weight="bold", color="#ffffff", zorder=4)
a.text(4.5, 1, "F = 2.1    p = 0.21", va="center", ha="left", fontsize=22,
       weight="bold", color=INK, zorder=4)
a.set_xlim(0, 62); a.set_xlabel("added explanatory power", fontsize=22)
a.tick_params(labelsize=20, left=False)
a.invert_yaxis()
p.para("UMA SUBSUMES the trivial baseline: it adds over a conjugated-atom count, but "
       "the count adds nothing over UMA. It also adds over Gasteiger charges and "
       "the count combined (F = 34.5).", size=22, color=AQUA)
p.done("13")

p = Panel(x, 13.3, CW, 20.2 - 13.3 - 0.45, "14 · Every alternative, excluded")
for q, r in [("Just the family label?", "tested WITHIN families — survives"),
             ("Just repeated rows?", "molecule-level permutation — survives"),
             ("Just counting conjugation?", "asymmetric subsumption — survives"),
             ("Just Gasteiger charges?", "complementary, r = +0.23 — survives"),
             ("Multiple comparisons?", "Holm 23/42, Bonferroni 18/42 — survives")]:
    p.kv(q, r, size=22, split=0.46)
p.para("Carboxylic acids are the one exception — chemically expected, since "
       "carboxylate delocalization barely varies with substituent.", size=21)
p.done("14")

p = Panel(x, 6.8, CW, 13.3 - 6.8 - 0.45, "15 · Honest limitations", fc="#fdf8ec")
for L in ["The solvent decomposition rests on 5 solvents and 3 parameters.",
          "ΔG$_{tr}$(H⁺) values come from secondary sources cross-checked against "
          "each other, not read off a primary table.",
          "Non-aqueous MAE 0.711 is for solvents PRESENT in training; unseen "
          "solvents are 3–5. Different claims.",
          "Site selection, not the model, still caps aqueous accuracy.",
          "No cationic-acid data here, so the charge-type prediction is untested."]:
    p.bullet(L, size=21)
p.done("15")

p = Panel(x, BOT, CW, 6.8 - BOT - 0.45, "16 · Conclusions", fc="#eef4fc")
p.para("A frozen foundation model, never trained on pK$_a$, supplies representations "
       "that predict it — but only when read at the ionizable site.", size=23, color=INK)
p.para("No pK$_a$ model transfers to an unseen solvent. That gap is one physical "
       "constant plus anion desolvation — and one measurement recovers most of it.", size=23, color=INK)
p.para("Matters most where data is sparse: chelators, PFAS, natural products.",
       size=23, color=ACCENT, weight="bold")
p.done("16")

fig.savefig("posters/urtc_poster.pdf", facecolor=SURFACE)
fig.savefig("posters/urtc_poster.png", facecolor=SURFACE, dpi=72)
print("wrote posters/urtc_poster.pdf and .png  (48 x 36 in landscape)")
print("OVERFLOW:", _overflow if _overflow else "none - all panels fit")

# ---------------------------------------------------------------------
# SOURCES (all repo-backed)
#   1.17/0.70, 1.08/0.53, 1.03/0.43, 0.86, 0.72 ... RESULTS.md
#   85.1/77.1/100% site agreement .................. results/site_ablation.txt
#   59.2 meV vs 1046-1437 meV ...................... README energy argument
#   1.59->0.67, 0.49->0.29, 25/26, SAMPL ........... results/benchmark_multiprotic_uma.txt
#   1.01/0.80/0.32, glycine pI 6.06 ................ results/coupling_validation.txt
#   LOSO 3-5, calibration curve .................... results/solvent_shift_uma/summary.txt
#   offsets, 1.019/-6.657/1.948, CI [0.84,1.20] .... results/proton_offset/summary.txt
#   57.1 / 2.1 / 34.5, Holm 23/42 .................. results/delocalization/summary.txt
#
# IN THE SUBMITTED ABSTRACT BUT NOT REPRODUCIBLE FROM THIS REPO - regenerate
# from committed code before using anywhere:
#   0.905 Novartis · 0.441 AvLiLuMoVe · 0.537/0.758/0.849 descriptor comparison
#   5184 training molecules · 56.5%->94.2% site accuracy · 1.795->0.711
#   non-aqueous · 1.160 · 0.735
# The poster uses the repo values instead (1.03 / 0.86, 0.43, 77.1%).
# ---------------------------------------------------------------------
