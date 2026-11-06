"""Publication-only rendering helpers; no analysis or training is executed."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.colors import to_hex, to_rgba

SUPPORT = Path(__file__).resolve().parent/'plotting_support'
sys.path.insert(0, str(SUPPORT))
from ml_publication_style import (configure_publication_style, style_axis as _style_axis,
    numeric_ticks, export_subfigures, save_figure_with_subfigures)
from scientific_palettes import get_palette, STYLE_COLORS
from audit_palette_usage import validate_series_colors

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'
QA=ROOT/'results'/'figure_checks'
QA.mkdir(parents=True,exist_ok=True)
configure_publication_style()
plt.rcParams.update({'font.weight':'normal','axes.labelweight':'normal',
    'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11,
    'legend.fontsize':10,'axes.linewidth':1.15,'xtick.major.width':1.1,
    'ytick.major.width':1.1,'xtick.major.size':5,'ytick.major.size':5})
_p=get_palette('nature_new')
# Preserve colour meanings across every exported panel.
PAL=[_p[1],_p[0],_p[2],_p[3],_p[4],_p[5],_p[6],_p[7]]
INK='#282828'
ROLE=validate_series_colors('nature_new',{'random':PAL[1],'group':PAL[0],
    'P':PAL[3],'RE':PAL[2],'SC':PAL[1],'SC+RE':PAL[0],
    'naive':PAL[5],'FE':PAL[0],'OLS':PAL[2]})
EDGE={}

def edge(c): return INK

def style_axis(ax,xlabel=None,ylabel=None,tick_size=12,label_size=14):
    _style_axis(ax,xlabel,ylabel,tick_size,label_size)
    for text in ax.get_xticklabels()+ax.get_yticklabels()+[ax.xaxis.label,ax.yaxis.label]:
        text.set_fontweight('normal')
    ax.tick_params(width=1.1,length=5)
    for s in ('left','bottom'): ax.spines[s].set_linewidth(1.15)

def normal_text(fig):
    for item in fig.findobj(Text): item.set_fontweight('normal')

def align_outer_boundaries(fig, axes, grid):
    """Measure and recalibrate each panel's full text/legend footprint this run."""
    rows,cols=grid
    # Fixed outer regions make full visual footprints align along rows/columns.
    # Apply margins measured from the actual renderer, including legends/labels.
    left,right,bottom,top=.015,.99,.02,.985
    gx=.028 if cols>1 else 0; gy=.045 if rows>1 else 0
    cw=(right-left-gx*(cols-1))/cols
    ch=(top-bottom-gy*(rows-1))/rows
    for _ in range(3):
        fig.canvas.draw(); renderer=fig.canvas.get_renderer()
        for i,ax in enumerate(axes):
            if not ax.get_visible(): continue
            rr,cc=divmod(i,cols)
            # A wide external legend must not recursively shrink the plotting
            # area; horizontal footprint uses the actual axis/tick/axis-label
            # box, while vertical margins retain the legend height.
            bbox=ax.get_tightbbox(renderer,bbox_extra_artists=[]).transformed(fig.transFigure.inverted())
            full=ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
            pos=ax.get_position()
            ml=pos.x0-bbox.x0; mr=bbox.x1-pos.x1
            mb=pos.y0-full.y0; mt=full.y1-pos.y1
            x0=left+cc*(cw+gx); y0=top-(rr+1)*ch-rr*gy
            ax.set_position([x0+ml,y0+mb,max(.08,cw-ml-mr),max(.08,ch-mb-mt)])
    fig.canvas.draw()
    return [{'panel':i,'tightbbox':list(ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).bounds)} for i,ax in enumerate(axes) if ax.get_visible()]

def audit_colors(fig):
    """Inspect actual data artists plus legend proxies, excluding structural ink."""
    from matplotlib.lines import Line2D
    from matplotlib.collections import PathCollection
    from matplotlib.patches import Rectangle
    allowed={x.lower() for x in get_palette('nature_new')}
    checks=[];bad=[]
    def check(c,kind):
        try:
            rgba=to_rgba(c)
            if rgba[3]==0:return
            h=to_hex(rgba).lower()
            if h in ('#ffffff','#282828','#252525','#000000','#30343b'):return
            checks.append({'kind':kind,'color':h})
            if h not in allowed:bad.append(h)
        except (ValueError,TypeError):pass
    for obj in fig.findobj():
        if hasattr(obj,'get_visible') and not obj.get_visible():continue
        if isinstance(obj,Line2D):
            if obj.get_linestyle() not in (None,'None','none','',' '):check(obj.get_color(),'line')
            if obj.get_marker() not in ('','None',None):check(obj.get_markerfacecolor(),'marker')
        elif isinstance(obj,PathCollection):
            for c in list(obj.get_facecolors())+list(obj.get_edgecolors()):check(c,'scatter')
        elif isinstance(obj,Rectangle):check(obj.get_facecolor(),'bar')
    if bad:raise ValueError(f'Out-of-palette artist colours: {set(bad)}')
    return {'palette':'nature_new','passed':True,'artists':checks,'continuous_scale':'blue-white-rose interpolation for numerical correlations'}

def save_combined_and_standalone(stem,panels,grid,figsize,single_size=(5.8,4.8),gridspec_kw=None):
    folder=FIG/stem;sub=folder/'subfigures';sub.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(*grid,figsize=figsize,gridspec_kw=gridspec_kw or {})
    axs=list(axs.ravel()) if hasattr(axs,'ravel') else [axs]
    for f,ax in zip(panels,axs):f(ax)
    for ax in axs[len(panels):]:ax.set_visible(False)
    normal_text(fig)
    footprints=align_outer_boundaries(fig,axs,grid)
    audit=audit_colors(fig)
    # Adapted helper writes combined files; independent redraws below supply
    # panel exports so no cropped substitute is used for a standalone figure.
    save_figure_with_subfigures(fig,folder,stem,dpi=600,close=True,export_panels=False)
    for i,f in enumerate(panels):
        fg,ax=plt.subplots(figsize=single_size);f(ax);normal_text(fg)
        align_outer_boundaries(fg,[ax],(1,1));audit_colors(fg)
        tag='abcdefghijklmnopqrstuvwxyz'[i]
        fg.savefig(sub/f'{stem}_{tag}.png',dpi=600,bbox_inches='tight',facecolor='white')
        fg.savefig(sub/f'{stem}_{tag}.pdf',bbox_inches='tight',facecolor='white',metadata={'Creator':'CHONG LIU','Author':'CHONG LIU'})
        plt.close(fg)
    (sub/'README.md').write_text('# Independent subfigures\n\nEach panel is independently redrawn on its own canvas; none is cropped from the combined figure.\n',encoding='utf-8')
    (QA/f'{stem}.json').write_text(json.dumps({'alignment':footprints,'palette':audit,'panels':len(panels),'ordinary_text':True},indent=2),encoding='utf-8')
    return folder
