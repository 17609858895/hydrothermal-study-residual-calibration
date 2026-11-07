"""Publication figures regenerated from unchanged numerical results; no model fitting."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from publication_figure_utils import *
from common import RES

C_BLUE, C_RED, C_GREEN, C_PBLUE, C_VPBLUE, C_PPINK, C_PROSE, C_PGREEN = PAL
TLAB = {"Y_oil": "Biocrude yield", "Y_char": "Solid/char yield", "Y_aq": "Aqueous yield", "Y_gas": "Gas yield",
        "ER_oil": "Calculated\nenergy ratio", "HHV_oil": "Biocrude HHV"}
R2 = "R$^{2}$"
Q2 = "Q$^{2}$"

# ------------------------------------------------------------------ data
q = pd.read_csv(RES / "02_quick_baseline.csv"); q = q[q.features == "chem+process"]
conf = pd.read_csv(RES / "06_study_conformal.csv")
sr = pd.read_csv(RES / "09_srgp_summary_oil_char.csv")
d3 = json.load(open(RES / "03_summary_Y_oil_Y_char.json"))
vg = {t: pd.read_csv(RES / f"09_variogram_{t}.csv") for t in ["Y_oil", "Y_char"]}
raw = {t: pd.read_csv(RES / f"09_srgp_raw_{t}.csv") for t in ["Y_oil", "Y_char"]}
dml_path = RES / "04_fe_dml_effects_v2.csv"
dml = pd.read_csv(dml_path if dml_path.exists() else RES / "04_fe_dml_effects.csv")


# ------------------------------------------------------------------ P1 panels
def p1a(ax):
    order = ["Y_oil", "Y_char", "Y_aq", "Y_gas", "ER_oil", "HHV_oil"]
    for i, t in enumerate(order[::-1]):
        r = q[(q.target == t) & (q.split == "random")].R2.values[0]
        g = q[(q.target == t) & (q.split == "group")].R2.values[0]
        ax.plot([g, r], [i, i], color=INK, lw=1.4, zorder=1)
        ax.scatter(r, i, s=95, color=ROLE["random"], edgecolor=INK, lw=1.0, zorder=3, label="Random rows" if i == 0 else None)
        ax.scatter(g, i, s=95, color=ROLE["group"], edgecolor=INK, lw=1.0, marker="s", zorder=3,
                   label="Held-out studies" if i == 0 else None)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([TLAB[t] for t in order[::-1]])
    ax.axvline(0, color=INK, lw=0.9, ls=":")
    style_axis(ax, xlabel=f"Cross-validated {R2}")
    ax.set_xlim(-0.2, 1.0); numeric_ticks(ax, y=False, n=5)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.0)




def p1c(ax, t="Y_oil"):
    v = vg[t]
    for mm, col, mk, lab in [(False, C_BLUE, "o", "Same feedstock/catalyst/solvent"), (True, C_RED, "^", "≥1 categorical mismatch")]:
        z = v[v.any_mismatch == mm].sort_values("dist")
        ax.plot(z.dist, z.gamma, color=col, lw=1.8, marker=mk, ms=7, mec=INK, mew=0.8, label=lab)
    ax.axhline(v.sill_between.iloc[0], color=INK, ls="--", lw=1.3)
    style_axis(ax, xlabel="Condition distance", ylabel="Residual semivariance")
    numeric_ticks(ax, n=5); ax.set_ylim(0, v.sill_between.iloc[0] * 1.08)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86))


def p1d(ax):
    meth = [("naive_row", 0, "Row-calibrated scores"), ("study", 0, "Study-aware scores"), ("study_fewshot", 5, "Study + 5 anchors")]
    cols = [C_PPINK, C_BLUE, C_GREEN]
    x = np.arange(2); w = 0.2
    for j, (m, k, lab) in enumerate(meth):
        vals = [conf[(conf.target == t) & (conf.method == m) & (conf.k == k)].coverage.values[0] for t in ["Y_oil", "Y_char"]]
        ax.bar(x + (j - 1.5) * w, vals, w, color=cols[j], edgecolor=edge(cols[j]) if cols[j] in EDGE else INK, lw=1.0, label=lab)
    vals = [sr[(sr.target == t) & (sr.k == 5) & (sr.model == "SR-GP")].coverage.values[0] for t in ["Y_oil", "Y_char"]]
    ax.bar(x + 1.5 * w, vals, w, color=C_RED, edgecolor=INK, lw=1.0, label="SR-GP + 5 anchors")
    ax.axhline(0.9, color=INK, ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(["Biocrude", "Solid/char"])
    style_axis(ax, ylabel="90% interval coverage")
    ax.set_ylim(0, 1.05); numeric_ticks(ax, x=False, n=6)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.0, columnspacing=0.8)


# ------------------------------------------------------------------ P2 panels
METH = [("offset", C_BLUE, "o", "-", "Constant offset"), ("BLUP(fixed)", C_GREEN, "s", "-", "Random-intercept BLUP"),
        ("kernel(fixed)", C_PBLUE, "D", "--", "Fixed kernel"), ("SR-GP(no local)", C_PPINK, "v", "--", "SR-GP without local term"),
        ("SR-GP", C_RED, "^", "-", "SR-GP")]








def p2c(ax):
    for t, mk in [("Y_oil", "o"), ("Y_char", "s")]:
        for m, col, lab in [("offset", C_BLUE, "Constant offset"), ("SR-GP", C_RED, "SR-GP")]:
            z = sr[(sr.target == t) & (sr.model == m)].sort_values("k")
            ax.plot(z.width, z.coverage, color=col, marker=mk, ms=9, lw=1.6, mec=INK, mew=0.9,
                    label=f"{lab}, {'biocrude' if t == 'Y_oil' else 'char'}")
    ax.axhline(0.9, color=INK, ls="--", lw=1.2)
    style_axis(ax, xlabel="Interval width (points)", ylabel="Coverage")
    ax.set_ylim(0.80, 0.96); ax.set_xlim(24, 47); numeric_ticks(ax, n=5)
    ax.legend(loc="lower right", fontsize=10.5, ncol=1)


def p2d(ax, t="Y_oil", k=3):
    r = raw[t]; r = r[r.k == k]
    st = r.groupby(["study", "model"]).rmse_study.mean().unstack()
    ax.scatter(st["offset"], st["SR-GP"], s=40, facecolor="none", edgecolor=C_RED, lw=1.3)
    lim = [0, np.nanpercentile(st[["offset", "SR-GP"]].values, 98) * 1.05]
    ax.plot(lim, lim, color=INK, lw=1.1, ls="--")
    win = np.mean(st["SR-GP"] < st["offset"])
    ax.text(0.04, 0.95, f"SR-GP better in {win:.0%} of studies", transform=ax.transAxes, va="top",
            fontsize=11.5, fontweight="normal", bbox=dict(fc="white", ec="none", alpha=0.85))
    ax.set_xlim(lim); ax.set_ylim(lim)
    style_axis(ax, xlabel="Study RMSE, constant offset (points)", ylabel="Study RMSE, SR-GP (points)")
    numeric_ticks(ax, n=5)


act = pd.read_csv(RES / "11_active_summary_Y_oil_Y_char.csv")
sec = pd.read_csv(RES / "09_srgp_summary_Y_aq_Y_gas_ER_oil_HHV_oil.csv")


def p2e(ax):
    for t, mk in [("Y_oil", "o"), ("Y_char", "s")]:
        for stg, col, ls, lab in [("random", C_BLUE, "--", "Random anchors"), ("IVR (active)", C_RED, "-", "SR-GP-selected anchors")]:
            z = act[(act.target == t) & (act.strategy == stg)].sort_values("k")
            ax.plot(z.k, z.R2, color=col, ls=ls, lw=2.0, marker=mk, ms=8, mec=INK, mew=0.9,
                    label=f"{'Random' if stg == 'random' else 'Selected'}, {'biocrude' if t == 'Y_oil' else 'char'}")
    style_axis(ax, xlabel="Anchor runs", ylabel=f"Query {Q2}")
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_ylim(0.45, 0.88); numeric_ticks(ax, x=False, n=5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5,1.0),ncol=2,fontsize=10.5,columnspacing=.7,handletextpad=.3)


def p2f(ax, k=3):
    allr = pd.concat([sr, sec])
    order = ["Y_oil", "Y_char", "Y_aq", "Y_gas", "ER_oil", "HHV_oil"]
    short = {"Y_oil": "Biocrude", "Y_char": "Char", "Y_aq": "Aqueous", "Y_gas": "Gas", "ER_oil": "E ratio", "HHV_oil": "HHV$_{oil}$"}
    x = np.arange(len(order)); w = 0.36
    off = [allr[(allr.target == t) & (allr.k == k) & (allr.model == "offset")].R2.values[0] for t in order]
    gp = [allr[(allr.target == t) & (allr.k == k) & (allr.model == "SR-GP")].R2.values[0] for t in order]
    ax.bar(x - w / 2, off, w, color=C_BLUE, edgecolor=INK, lw=1.0, label="Constant offset")
    ax.bar(x + w / 2, gp, w, color=C_RED, edgecolor=INK, lw=1.0, label="SR-GP")
    ax.set_xticks(x); ax.set_xticklabels([short[t] for t in order], rotation=30, ha="right")
    style_axis(ax, ylabel=f"{Q2} with 3 anchor runs")
    ax.set_ylim(0, 1.0); numeric_ticks(ax, x=False, n=6)
    ax.legend(loc="upper left", ncol=2)


# ------------------------------------------------------------------ P3 panels
def p3a(ax):
    z = dml[dml.FE_DML.notna()].copy()
    short = {"Alkali catalyst (vs none)": "Alkali catalyst", "Organic (co)solvent (vs water)": "Organic co-solvent",
             "Temperature (+10 C)": "Temperature +10 °C", "log residence time (+1 ln-unit)": "ln(1 + t) + 1",
             "Solid loading (+5 wt%)": "Solid loading +5 wt%"}
    z["lab"] = z.treatment.map(short) + np.where(z.outcome == "Y_oil", "\nBiocrude", "\nSolid/char")
    z = z.iloc[::-1].reset_index(drop=True)
    for i, r in z.iterrows():
        ax.errorbar(r.naive_DML, i + 0.17, xerr=1.96 * r.naive_DML_se, fmt="o", ms=7, color=C_PPINK, mec=C_RED, mew=1.0,
                    ecolor=C_RED, elinewidth=1.3, capsize=3, label="Pooled DML" if i == 0 else None)
        ax.errorbar(r.FE_DML, i - 0.17, xerr=1.96 * r.FE_DML_se, fmt="s", ms=7, color=C_BLUE, mec=INK, mew=1.0,
                    ecolor=INK, elinewidth=1.3, capsize=3, label="Within-study" if i == 0 else None)
    ax.axvline(0, color=INK, lw=1.0, ls=":")
    ax.set_yticks(range(len(z))); ax.set_yticklabels(z.lab, fontsize=11)
    style_axis(ax, xlabel="Adjusted yield contrast (pp)", tick_size=11)
    numeric_ticks(ax, y=False, n=6)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)


def p3b(ax):
    rng = np.random.default_rng(7)
    for j, (t, col, mk) in enumerate([("Y_oil", C_BLUE, "o"), ("Y_char", C_RED, "s")]):
        c = pd.read_csv(RES / f"04_cate_{t}_Alkali.csv")
        c["q"] = pd.qcut(c.Ash, 4, labels=False)
        m, lo, hi = [], [], []
        studies = c.study_id.unique()
        for qq in range(4):
            m.append(c[c.q == qq].tau.mean())
            bs = []
            for _ in range(300):
                sel = rng.choice(studies, len(studies), replace=True)
                zz = pd.concat([c[c.study_id == s] for s in sel]) if False else c.set_index("study_id").loc[sel].reset_index()
                bs.append(zz[zz.q == qq].tau.mean())
            lo.append(np.nanpercentile(bs, 2.5)); hi.append(np.nanpercentile(bs, 97.5))
        x = np.arange(4) + (j - 0.5) * 0.18
        ax.errorbar(x, m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt=mk + "-", ms=9, color=col, mec=INK, mew=1.0,
                    ecolor=INK, capsize=4, lw=1.8, label="Biocrude" if t == "Y_oil" else "Solid/char")
    ax.axhline(0, color=INK, lw=1.0, ls=":")
    ax.set_xticks(range(4)); ax.set_xticklabels(["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"])
    style_axis(ax, xlabel="Feedstock ash quartile", ylabel="Adjusted alkali contrast (pp)")
    numeric_ticks(ax, x=False, n=6)
    ax.legend(loc="lower right")


bench = pd.read_csv(RES / "05_benchmark20.csv")




def p4a(ax):
    p4(ax, "Y_oil")


def p4b(ax):
    p4(ax, "Y_char")


# ------------------------------------------------------------------ P5 (supporting): phase coupling + robustness
from matplotlib.colors import LinearSegmentedColormap
DIV = LinearSegmentedColormap.from_list("blue_white_red", [C_BLUE, "#FFFFFF", C_RED])


def p5a(ax):
    pc = json.load(open(RES / "13_lab_effect_phase_corr.json"))
    M = np.array(pc["corr_mean"]); Sd = np.array(pc["corr_sd"])
    lab = ["Biocrude", "Char", "Aqueous", "Gas"]
    n = len(lab)
    Mm = np.where(np.tril(np.ones((n, n)), -1) == 1, M, np.nan)   # strict lower triangle (diagonal = 1 omitted)
    im = ax.imshow(Mm, cmap=DIV, vmin=-0.6, vmax=0.6, aspect='auto')
    for i in range(n):
        for j in range(i + 1):
            if i != j:
                ax.text(j, i, f"{M[i, j]:.2f}\n±{Sd[i, j]:.2f}", ha="center", va="center", fontsize=11, fontweight="normal", color=INK)
    ax.set_xticks(range(n)); ax.set_xticklabels(lab, rotation=30, ha="right"); ax.set_yticks(range(n)); ax.set_yticklabels(lab)
    style_axis(ax)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    cax=ax.inset_axes([0.52,0.86,0.43,0.035])
    cb = plt.colorbar(im, cax=cax, orientation='horizontal',ticks=[-.6,0,.6])
    cb.set_label("Residual correlation", fontsize=9.5, fontweight="normal",labelpad=3)
    cb.ax.tick_params(labelsize=9.0,length=3,width=.8)


def p5b(ax):
    mo = pd.read_csv(RES / "13_mo_srgp_summary.csv")
    k = 5
    groups = [("Y_oil", "MO char->oil", "Biocrude"), ("Y_char", "MO oil->char", "Char")]
    x = np.arange(2); w = 0.25
    vals = {"No anchors": [], "Other-phase anchors": [], "Same-phase anchors": []}
    for t, cross, _ in groups:
        z = mo[(mo.phase == t) & (mo.k == k)].set_index("model").R2
        vals["No anchors"].append(z["base"]); vals["Other-phase anchors"].append(z[cross]); vals["Same-phase anchors"].append(z["indep SR-GP"])
    for j, (nm, col) in enumerate(zip(vals, [C_VPBLUE, C_GREEN, C_RED])):
        ax.bar(x + (j - 1) * w, vals[nm], w, color=col, edgecolor=edge(col) if col in EDGE else INK, lw=1.0, label=nm)
    ax.set_xticks(x); ax.set_xticklabels([gl for _, _, gl in groups])
    style_axis(ax, ylabel=f"{Q2} with 5 anchor runs")
    ax.set_ylim(0, 0.9); numeric_ticks(ax, x=False, n=6)
    ax.legend(loc="upper left", fontsize=10.5)


def p5c(ax):
    cat = pd.read_csv(RES / "09_srgp_summary_Y_oil_Y_char_Y_aq_Y_gas_ER_oil_HHV_oil_catboost.csv")
    for t, mk in [("Y_oil", "o"), ("Y_char", "s")]:
        for df_, col, ls, lab in [(sr, C_BLUE, "--", "LightGBM base"), (cat, C_RED, "-", "CatBoost base")]:
            z = df_[(df_.target == t) & (df_.model == "SR-GP")].sort_values("k")
            ax.plot(z.k, z.R2, color=col, ls=ls, marker=mk, ms=8, lw=1.9, mec=INK, mew=0.9,
                    label=f"{lab}, {'biocrude' if t == 'Y_oil' else 'char'}")
    style_axis(ax, xlabel="Anchor runs in the new study", ylabel=f"SR-GP {Q2}")
    ax.set_xticks([1, 3, 5]); ax.set_ylim(0.45, 0.85); numeric_ticks(ax, x=False, n=5)
    ax.legend(loc="lower right", fontsize=10.5)


# Final panels override preliminary summaries whose evaluation supports differ.
def p1b(ax):
    icc=json.loads((RES/'10_lab_effect.json').read_text(encoding='utf-8'))
    x=np.arange(2);w=.32
    for j,(key,col,lab) in enumerate([('ICC_raw',C_BLUE,'Observed yield'),('ICC_resid',C_RED,'Held-out residual')]):
        vals=[icc[t][key] for t in ('Y_oil','Y_char')]
        bars=ax.bar(x+(j-.5)*w,vals,w,color=col,edgecolor=INK,lw=.9,label=lab)
        for bar,v in zip(bars,vals):ax.text(bar.get_x()+w/2,v+.025,f'{v:.2f}',ha='center',fontsize=11,color=INK)
    ax.set_xticks(x);ax.set_xticklabels(['Biocrude','Solid/char'])
    style_axis(ax,ylabel='Study variance share (ICC)')
    ax.set_ylim(0,.95);numeric_ticks(ax,x=False,n=5)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.12),ncol=2)


gpbench=pd.read_csv(RES/'14_gpboost_summary_six_targets.csv')  # response-specific GPBoost iterations (14b, 18)

def matched_curve(ax,t):
    methods=[('offset',C_BLUE,'o','--','Constant offset'),
             ('GPB-RE',C_GREEN,'s',':','GPBoost + intercept'),
             ('GPB-RE+GP',C_VPBLUE,'D','-.','GPBoost + local GP'),
             ('SR-GP',C_RED,'^','-','SR-GP')]
    for m,col,mark,ls,lab in methods:
        z=gpbench[(gpbench.target==t)&(gpbench.model==m)].sort_values('k')
        ax.plot(z.k,z.R2,color=col,marker=mark,ls=ls,lw=2,ms=7,mec=INK,mew=.8,label=lab)
    ax.set_xticks([1,3,5]);ax.set_ylim(.32,.86)
    style_axis(ax,xlabel='Anchor runs',ylabel=f'Query {Q2}')
    numeric_ticks(ax,x=False,n=5)
    ax.legend(loc='lower right',fontsize=11.0,handlelength=1.8)

def p2a(ax):matched_curve(ax,'Y_oil')
def p2b(ax):matched_curve(ax,'Y_char')


def p4(ax,t):
    z=bench[(bench.target==t)&~bench.model.str.startswith('Stacking')]
    g=z[z.split=='group'].set_index('model');r=z[z.split=='random'].set_index('model')
    order=g.sort_values('R2').index.tolist()
    for i,m in enumerate(order):
        gv=g.loc[m,'R2'];rv=r.loc[m,'R2']
        ax.plot([gv,rv],[i,i],color=INK,lw=.9,zorder=1)
        ax.scatter(rv,i,s=48,color=ROLE['random'],edgecolor=INK,lw=.7,zorder=3,marker='o')
        ax.scatter(gv,i,s=48,color=ROLE['group'],edgecolor=INK,lw=.7,zorder=3,marker='s')
    ax.set_yticks(range(len(order)));ax.set_yticklabels(order)
    ax.axvline(0,color=INK,lw=.9,ls=':')
    style_axis(ax,xlabel=f'Cross-validated {R2}',tick_size=10.5)
    low=min(g.R2.min(),r.R2.min())-.10
    ax.set_xlim(low,1.0);numeric_ticks(ax,y=False,n=6)
    from matplotlib.lines import Line2D
    h=[Line2D([],[],ls='',marker='o',ms=7,mfc=ROLE['random'],mec=INK,label='Random rows'),
       Line2D([],[],ls='',marker='s',ms=7,mfc=ROLE['group'],mec=INK,label='Held-out studies')]
    ax.legend(handles=h,loc='lower left',bbox_to_anchor=(0,1),ncol=2,fontsize=10)


def main(only=None):
    entries=[
      ('Fig01_study_structure',[p1a,p1b,p1c,p1d],(2,2),(8.5,7.5),(6.2,4.8)),
      ('Fig02_prediction_benchmarks',[p4a,p4b],(1,2),(8.5,9.2),(6.7,8.2)),
      ('Fig03_few_shot_transfer',[p2a,p2b,p2e,p2c,p2d,p2f],(3,2),(8.5,10.2),(6.1,4.8)),
      ('Fig04_within_study_contrasts',[p3a,p3b],(1,2),(8.5,7.2),(7.4,5.9)),
      ('Fig05_phase_coupling',[p5a,p5b],(1,2),(8.5,4.9),(6.0,5.1)),
      ('FigS01_base_learner_sensitivity',[p5c],(1,1),(6.0,5.1),(6.0,5.1))]
    for stem,panels,grid,size,single in entries:
        if only and stem not in only:continue
        save_combined_and_standalone(stem,panels,grid,size,single_size=single)
        print('Exported',stem,flush=True)
    import nbformat
    for stem,_,_,_,_ in entries:
        if only and stem not in only:continue
        nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# Reproduce this figure\n\nRuns the publication plotting entry point using the accompanying numerical results.'),
            nbformat.v4.new_code_cell("from pathlib import Path\nimport runpy\nroots=[Path.cwd(),*Path.cwd().parents]\ncandidates=[p/'scripts'/'16_publication_figures.py' for p in roots]+[p/'10_ML分析'/'scripts'/'16_publication_figures.py' for p in roots]\nscript=next(p for p in candidates if p.exists())\nmodule=runpy.run_path(str(script))\nmodule['main'](only=['"+stem+"'])")])
        nbformat.write(nb,FIG/stem/f'{stem}_jupyter_code.ipynb')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--only',nargs='*')
    main(parser.parse_args().only)


