#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
■ 金手指 Golden Finger — 散户/机构 + Delta 可视化
Authors: 3123007918 张懿哲 · 3123007910 李沐鑫 · 3223007924 黄凡绮
"""

import sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.special import erf as sp_erf

# ══ 中文字体 + 负号 ══════════════════════════════════════════
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong',
    'Arial Unicode MS', 'WenQuanYi Micro Hei', 'DejaVu Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10

from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation          # ← 关键：用原生动画
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

try:
    import mplcursors
    HAS_CURSOR = True
except ImportError:
    HAS_CURSOR = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QPushButton, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# ══════════════════════════════════════════════════════════════
#  参数
# ══════════════════════════════════════════════════════════════
class Tranche:
    def __init__(self, issue, guar, PR, fee):
        self.issue=issue; self.guar=guar; self.PR=PR; self.fee=fee

class P:
    S0=300; L=375; T=3; r=0.045
    A=Tranche(1000,1000,0.70,0.028)
    B=Tranche(1100,1000,1.30,0.040)

COL_A='#27B345'; COL_B='#E6A800'
COL_I='#7D1AB5'; COL_D='#E65C00'; COL_S='#1A4FCC'

# ══════════════════════════════════════════════════════════════
#  数学函数
# ══════════════════════════════════════════════════════════════
def payoff_call(ST, tr):
    cy = 0.25 if ST>=375 else (min(ST/300-1,0.25) if ST>300 else 0.0)
    return max(cy*tr.PR - tr.fee, -tr.fee)

def c_ret_final(ST, tr):
    cy = 0.25 if ST>=375 else (min(ST/300-1,0.25) if ST>300 else 0.0)
    return (tr.guar + cy*tr.PR*1000 - tr.issue)/tr.issue - tr.fee

def c_ret_path(ST, t, T, tr):
    r_f = c_ret_final(ST, tr)
    gap = (tr.guar - tr.issue)/tr.issue
    sm  = max(0.0, min(1.0, (300-ST)/50))
    return min(r_f - gap*(1-t/T)*(1-sm), c_ret_final(375, tr))

def i_ret(ST):
    return 0.034 - (0.018*np.exp(-((ST-375)/14)**2) + 0.004)

def calc_delta(S, tau, sig, rr):
    if tau<=0: return 0.0
    te = max(tau, 0.05)
    d1 = (np.log(S/300)+(rr+0.5*sig**2)*te)/(sig*np.sqrt(te))
    d3 = (np.log(S/375)+(rr+0.5*sig**2)*te)/(sig*np.sqrt(te))
    d  = 0.5*(1+sp_erf(d1/np.sqrt(2))) - 0.5*(1+sp_erf(d3/np.sqrt(2)))
    if tau<0.05: d *= tau/0.05
    if S>=375:   d  = 0.0
    if S<280:    d *= max(0.0,(S-250)/30)
    return float(max(0.0, min(d, 1.2)))

def sim_scene(Sfin, vv, seed):
    N=250; t=np.linspace(0,P.T,N); dt=P.T/N
    np.random.seed(seed)
    S   = 300*np.exp(np.cumsum((np.log(Sfin/300)/P.T-0.5*vv**2)*dt + vv*np.sqrt(dt)*np.random.randn(N)))
    rA  = np.zeros(N); rB=np.zeros(N)
    capA=c_ret_final(375,P.A); capB=c_ret_final(375,P.B)
    ev=-1; KO=False
    for k in range(N):
        if KO: rA[k]=capA*100; rB[k]=capB*100
        elif S[k]>=375: KO=True; ev=k; rA[k]=capA*100; rB[k]=capB*100
        else:
            rA[k]=c_ret_path(S[k],t[k],P.T,P.A)*100
            rB[k]=c_ret_path(S[k],t[k],P.T,P.B)*100
    return t, S, rA, rB, ev

def attach_cursor(lines):
    if HAS_CURSOR and lines:
        cur = mplcursors.cursor(lines, hover=True)
        @cur.connect("add")
        def _(sel):
            x,y = sel.target
            sel.annotation.set_text(f'x={x:.3f}\ny={y:.3f}')
            sel.annotation.get_bbox_patch().set(fc='#FFFDE0', alpha=0.92)

def plot_scene(fig, rect, t, S, ap, bp, ev, title, legend=False):
    ax  = fig.add_axes(rect)
    ax2 = ax.twinx()
    ax.axhline(375,ls='--',color='r',lw=1.2)
    ax.axhline(300,ls='--',color='k',lw=1.0)
    hS, = ax.plot(t,S,color=COL_S,lw=2.0,label='St')
    ax.set_ylabel('St'); ax.set_xlim(0,P.T)
    ax.set_ylim(min(min(S)-10,220), max(max(S)+10,405))
    fA0=-P.A.fee*100; fB0=-P.B.fee*100
    ax2.axhline(0,ls=':',color='k',lw=0.7)
    ax2.axhline(fA0,ls='--',color=COL_A,lw=0.9)
    ax2.axhline(fB0,ls='--',color=COL_B,lw=0.9)
    ax2.text(P.T+0.04,fA0,f'A费\n{fA0:.1f}%',color=COL_A,fontsize=7,fontweight='bold',va='center')
    ax2.text(P.T+0.04,fB0,f'B费\n{fB0:.1f}%',color=COL_B,fontsize=7,fontweight='bold',va='center')
    hA, = ax2.plot(t,ap,color=COL_A,lw=2.4,label='A 产品总收益')
    hB, = ax2.plot(t,bp,color=COL_B,lw=2.4,label='B 产品总收益')
    ax2.set_ylabel('收益率 (%)'); ax2.set_ylim(-17,22)
    if ev>=0:
        ax.plot(t[ev],375,'p',ms=18,mfc='r',mec='k',mew=1.5)
        ax.text(t[ev]+0.05,380,'⚡敲出',color='r',fontweight='bold',
                bbox=dict(facecolor='yellow',edgecolor='red',alpha=0.9),fontsize=8)
    ax.set_xlabel('t (年)'); ax.set_title(title,fontweight='bold',fontsize=10)
    if legend:
        ax.legend([hS,hA,hB],['St','A 产品总收益','B 产品总收益'],loc='lower left',fontsize=7)
    attach_cursor([hS,hA,hB])

# ══════════════════════════════════════════════════════════════
#  主窗口
# ══════════════════════════════════════════════════════════════
class GoldenFingerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('■ 金手指 Golden Finger — 散户/机构 + Delta 可视化')
        self.resize(1460, 900)
        self.setStyleSheet('QMainWindow{background:#EEEEF6;}')

        self._anim = None   # 持有 FuncAnimation 引用，防止被GC销毁
        self.canvas  = None
        self.toolbar = None

        c=QWidget(); self.setCentralWidget(c)
        hl=QHBoxLayout(c); hl.setContentsMargins(8,8,8,8); hl.setSpacing(8)
        hl.addWidget(self._build_left())

        self.plot_area=QWidget()
        self.plot_area.setStyleSheet('background:white;border-radius:8px;')
        self.plot_layout=QVBoxLayout(self.plot_area)
        self.plot_layout.setContentsMargins(2,2,2,2)
        hl.addWidget(self.plot_area,1)

    # ── 左侧面板 ───────────────────────────────────────────
    def _build_left(self):
        frame=QFrame()
        frame.setFixedWidth(238)
        frame.setStyleSheet('''
            QFrame{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
                stop:0 #1A1A2E,stop:1 #16213E);border-radius:10px;}
        ''')
        vb=QVBoxLayout(frame)
        vb.setContentsMargins(12,14,12,14); vb.setSpacing(0)

        title_box=QFrame()
        title_box.setStyleSheet('QFrame{background:rgba(255,255,255,0.07);border-radius:8px;border:none;}')
        tb=QVBoxLayout(title_box); tb.setContentsMargins(8,6,8,6); tb.setSpacing(2)
        t1=QLabel('■ 金手指 Golden Finger')
        t1.setStyleSheet('color:#FFD700;font-size:14px;font-weight:bold;background:transparent;')
        t2=QLabel('散户 / 机构 + Delta 可视化')
        t2.setStyleSheet('color:#AAC4FF;font-size:10px;background:transparent;')
        tb.addWidget(t1); tb.addWidget(t2)
        vb.addWidget(title_box); vb.addSpacing(10)

        def sec(txt):
            l=QLabel(txt)
            l.setStyleSheet('color:#90CAF9;font-size:10px;font-weight:bold;'
                           'background:transparent;margin-top:8px;margin-bottom:2px;')
            return l

        cs='''QComboBox{background:#0F3460;color:white;border:1px solid #3A6EA5;
                border-radius:5px;padding:4px 8px;font-size:11px;}
              QComboBox:hover{border:1px solid #64B5F6;}
              QComboBox QAbstractItemView{background:#0F3460;color:white;
                selection-background-color:#1565C0;}
              QComboBox::drop-down{border:none;}'''
        ss='''QSlider::groove:horizontal{height:4px;background:#2A4A6A;border-radius:2px;}
              QSlider::handle:horizontal{width:16px;height:16px;margin:-6px 0;
                background:#64B5F6;border-radius:8px;border:2px solid #1565C0;}
              QSlider::sub-page:horizontal{background:#1E88E5;border-radius:2px;}'''
        ls='color:#E3F2FD;font-size:11px;background:transparent;'

        vb.addWidget(sec('① 模块'))
        self.dd_mod=QComboBox(); self.dd_mod.addItems(['1-Payoff情景','2-3D曲面','3-双轴对比','4-动画+Delta'])
        self.dd_mod.setCurrentIndex(3); self.dd_mod.setStyleSheet(cs); vb.addWidget(self.dd_mod)

        vb.addWidget(sec('② 档位'))
        self.dd_tr=QComboBox(); self.dd_tr.addItems(['仅A','仅B','A+B对比'])
        self.dd_tr.setCurrentIndex(2); self.dd_tr.setStyleSheet(cs); vb.addWidget(self.dd_tr)

        vb.addWidget(sec('③ 情景'))
        self.dd_sc=QComboBox(); self.dd_sc.addItems(['下跌→250','上涨→350','敲出→395','随机'])
        self.dd_sc.setCurrentIndex(2); self.dd_sc.setStyleSheet(cs); vb.addWidget(self.dd_sc)

        vb.addWidget(sec('④ 速度'))
        self.lbl_sp=QLabel('速度: 5.0'); self.lbl_sp.setStyleSheet(ls); vb.addWidget(self.lbl_sp)
        self.sld_sp=QSlider(Qt.Horizontal); self.sld_sp.setRange(10,100); self.sld_sp.setValue(50)
        self.sld_sp.setStyleSheet(ss)
        self.sld_sp.valueChanged.connect(lambda v:self.lbl_sp.setText(f'速度: {v/10:.1f}'))
        vb.addWidget(self.sld_sp)

        vb.addWidget(sec('⑤ 波动率 σ'))
        self.lbl_vol=QLabel('σ = 0.170'); self.lbl_vol.setStyleSheet(ls); vb.addWidget(self.lbl_vol)
        self.sld_vol=QSlider(Qt.Horizontal); self.sld_vol.setRange(5,40); self.sld_vol.setValue(17)
        self.sld_vol.setStyleSheet(ss)
        self.sld_vol.valueChanged.connect(lambda v:self.lbl_vol.setText(f'σ = {v/100:.3f}'))
        vb.addWidget(self.sld_vol)

        vb.addSpacing(10)
        self.btn_run=QPushButton('▶  运行 / 播放')
        self.btn_run.setStyleSheet('''
            QPushButton{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
                stop:0 #43A047,stop:1 #2E7D32);color:white;font-weight:bold;
                font-size:14px;border-radius:7px;padding:10px;border:none;}
            QPushButton:hover{background:#388E3C;}
            QPushButton:pressed{background:#1B5E20;}''')
        self.btn_run.clicked.connect(self.run_module); vb.addWidget(self.btn_run)
        vb.addSpacing(5)

        btn_clr=QPushButton('🗑  清除画布')
        btn_clr.setStyleSheet('''
            QPushButton{background:rgba(255,255,255,0.10);color:#CFD8DC;
                border:1px solid rgba(255,255,255,0.2);border-radius:6px;
                padding:6px;font-size:11px;}
            QPushButton:hover{background:rgba(255,255,255,0.18);}''')
        btn_clr.clicked.connect(self.clear_plot); vb.addWidget(btn_clr)

        vb.addSpacing(10)
        sep=QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet('background:rgba(255,255,255,0.15);max-height:1px;border:none;')
        vb.addWidget(sep); vb.addSpacing(6)

        info=QLabel(
            '<b style="color:#FFD700">── 产品参数 ──</b><br>'
            '<span style="color:#B0BEC5">S₀=K=300 │ L=375 │ T=3年 │ r=4.5%</span><br><br>'
            '<span style="color:#A5D6A7"><b>Tranche A</b>: 上限 +14.7%</span><br>'
            '<span style="color:#90A4AE">&nbsp;发行1000 保本1000 PR70% 费2.8%</span><br><br>'
            '<span style="color:#FFF176"><b>Tranche B</b>: 上限 +16.45%</span><br>'
            '<span style="color:#90A4AE">&nbsp;发行1100 保本1000 PR130% 费4%</span><br>'
            '<span style="color:#90A4AE">&nbsp;保险垫 90.91%</span>')
        info.setStyleSheet('background:rgba(255,255,255,0.05);padding:8px;'
                          'border-radius:6px;font-size:10px;line-height:160%;')
        info.setWordWrap(True); vb.addWidget(info); vb.addSpacing(6)

        tip=QLabel('💡 悬浮曲线显示精确值\n💡 3D图鼠标拖拽旋转\n💡 工具栏缩放/平移/保存')
        tip.setStyleSheet('color:#80CBC4;font-size:9px;background:rgba(0,128,100,0.15);'
                         'padding:6px;border-radius:5px;line-height:160%;')
        tip.setWordWrap(True); vb.addWidget(tip)
        vb.addStretch()

        sep2=QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet('background:rgba(255,255,255,0.15);max-height:1px;border:none;')
        vb.addWidget(sep2); vb.addSpacing(6)

        ab_box=QFrame()
        ab_box.setStyleSheet('background:rgba(255,215,0,0.10);border-radius:6px;border:none;')
        ab=QVBoxLayout(ab_box); ab.setContentsMargins(6,6,6,6); ab.setSpacing(4)
        al0=QLabel('◆  Authors  ◆')
        al0.setStyleSheet('color:#FFD700;font-size:12px;font-weight:bold;background:transparent;')
        al0.setAlignment(Qt.AlignCenter); ab.addWidget(al0)
        for name in ['3123007918  张懿哲','3123007910  李沐鑫','3223007924  黄凡绮']:
            l=QLabel(name)
            l.setStyleSheet('color:#E0E0E0;font-size:11px;background:transparent;')
            l.setAlignment(Qt.AlignCenter); ab.addWidget(l)
        vb.addWidget(ab_box)
        return frame

    # ── 画布管理 ──────────────────────────────────────────
    def clear_plot(self):
        self._anim = None          # 先停动画
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater(); self.canvas=None
        if self.toolbar:
            self.plot_layout.removeWidget(self.toolbar)
            self.toolbar.deleteLater(); self.toolbar=None

    def _show(self, fig):
        self.clear_plot()
        cv=FigureCanvas(fig)
        cv.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tb=NavigationToolbar(cv, self)
        self.plot_layout.addWidget(tb)
        self.plot_layout.addWidget(cv)
        self.canvas=cv; self.toolbar=tb
        cv.draw()

    def run_module(self):
        m=self.dd_mod.currentIndex()+1
        {1:self._mod1,2:self._mod2,3:self._mod3,4:self._mod4}[m](
            self.dd_sc.currentIndex()+1,
            self.dd_tr.currentIndex()+1,
            self.sld_sp.value()/10.0,
            self.sld_vol.value()/100.0)

    # ══ 模块1 ══════════════════════════════════════════════
    def _mod1(self, sc, tr, sp, vol):
        fig=Figure(figsize=(15,9),facecolor='white')
        fig.suptitle('■ 金手指 Golden Finger — 模块1: Payoff 情景分析',
                     fontsize=14,fontweight='bold',color='#B01515',y=0.99)
        ax1=fig.add_axes([0.06,0.56,0.41,0.37])
        ax1.set_facecolor('#FAFAFA'); ax1.grid(True,alpha=0.35)
        S=np.linspace(200,455,500)
        rA=np.array([payoff_call(s,P.A)*100 for s in S])
        rB=np.array([payoff_call(s,P.B)*100 for s in S])
        hA,=ax1.plot(S,rA,lw=2.8,color=COL_A,label=f'A  PR={P.A.PR*100:.0f}%  fee={P.A.fee*100:.1f}%')
        hB,=ax1.plot(S,rB,lw=2.8,color=COL_B,label=f'B  PR={P.B.PR*100:.0f}%  fee={P.B.fee*100:.1f}%')
        ax1.axvline(300,ls='--',color='k',lw=1.2); ax1.axvline(375,ls='--',color='r',lw=1.4)
        ax1.axhline(0,ls=':',color='k',lw=0.8)
        ax1.text(302,30.5,'K=300',color='k',fontweight='bold',fontsize=8)
        ax1.text(377,30.5,'L=375',color='r',fontweight='bold',fontsize=8)
        fA0=-P.A.fee*100; fB0=-P.B.fee*100
        bepA=300*(1+P.A.fee/P.A.PR); bepB=300*(1+P.B.fee/P.B.PR)
        ax1.axhline(fA0,ls='--',color=COL_A,lw=1.0); ax1.axhline(fB0,ls='--',color=COL_B,lw=1.0)
        ax1.text(455,fA0,f'A起始\n{fA0:.1f}%',color=COL_A,fontsize=7,fontweight='bold',va='center')
        ax1.text(455,fB0,f'B起始\n{fB0:.1f}%',color=COL_B,fontsize=7,fontweight='bold',va='center')
        for bep,col,lbl in [(bepA,COL_A,'A'),(bepB,COL_B,'B')]:
            ax1.axvline(bep,ymax=0.48,ls='-.',color=col,lw=0.9)
            ax1.text(bep,-9.2,f'{lbl}盈亏\n平衡\n{bep:.0f}',color=col,fontsize=7,ha='center')
        ax1.set_xlabel('S_T'); ax1.set_ylabel('Payoff (%)')
        ax1.set_title(f'(a) 看涨期权 Payoff  A:+{payoff_call(375,P.A)*100:.1f}%  '
                      f'B:+{payoff_call(375,P.B)*100:.1f}%',fontweight='bold',fontsize=10)
        ax1.legend(loc='upper left',fontsize=8); ax1.set_ylim(-11,33)
        attach_cursor([hA,hB])
        t1,S1,ad1,bd1,_  = sim_scene(240,0.16,11)
        t2,S2,ad2,bd2,_  = sim_scene(355,0.14,22)
        t3,S3,ad3,bd3,ev = sim_scene(395,0.24,33)
        plot_scene(fig,[0.555,0.56,0.41,0.37],t1,S1,ad1,bd1,-1,'(b) 情景1: 破位下跌',legend=True)
        plot_scene(fig,[0.06,0.07,0.41,0.37], t2,S2,ad2,bd2,-1,'(c) 情景2: 温和上涨')
        plot_scene(fig,[0.555,0.07,0.41,0.37],t3,S3,ad3,bd3,ev,'(d) 情景3: 震荡触障敲出')
        self._show(fig)

    # ══ 模块2 ══════════════════════════════════════════════
    def _mod2(self, *_):
        fig=Figure(figsize=(13,8),facecolor='white')
        ax=fig.add_subplot(111,projection='3d')
        Sa=np.linspace(260,420,60); Va=np.linspace(0.05,0.40,50)
        Sg,Vg=np.meshgrid(Sa,Va)
        sc_a=18*(Sg<375)+38*(Sg>=375)
        HL=14*Vg*np.exp(-((Sg-375)/sc_a)**2)+2.5*Vg
        res=6*Vg*(Sg>=375)*(1-np.exp(-(Sg-375)/60))
        PnL=34-HL*8-res*8
        surf=ax.plot_surface(Sg,Vg,PnL,cmap='turbo',alpha=0.93,edgecolor='none')
        fig.colorbar(surf,ax=ax,shrink=0.5,pad=0.08,label='发行方利润 (元)')
        ax.set_xlabel('黄金价格 S',labelpad=10)
        ax.set_ylabel('波动率 σ',  labelpad=10)
        ax.set_zlabel('发行方利润 (元)',labelpad=10)
        ax.set_title('模块2: 机构利润曲面  (鼠标拖拽旋转 / 工具栏缩放)',fontweight='bold',fontsize=13)
        ax.view_init(elev=30,azim=-40)
        fig.tight_layout()
        self._show(fig)

    # ══ 模块3 ══════════════════════════════════════════════
    def _mod3(self, sc, tr, sp, vol):
        fig=Figure(figsize=(13,7),facecolor='white')
        ax=fig.add_axes([0.08,0.12,0.80,0.78]); ax2=ax.twinx()
        ax.set_facecolor('#FAFAFA'); ax.grid(True,alpha=0.35)
        S=np.linspace(220,430,400)
        rA=np.array([c_ret_final(s,P.A)*100 for s in S])
        rB=np.array([c_ret_final(s,P.B)*100 for s in S])
        iss=np.array([i_ret(s)*100 for s in S])
        lines=[]; labels=[]
        if tr!=2:
            l,=ax.plot(S,rA,lw=2.8,color=COL_A); lines.append(l); labels.append('散户A')
        if tr!=1:
            l,=ax.plot(S,rB,lw=2.8,color=COL_B); lines.append(l); labels.append('散户B')
        lK=ax.axvline(300,ls='--',color='k',lw=1.4)
        lL=ax.axvline(375,ls='--',color='r',lw=1.4)
        ax.set_ylabel('散户收益 (%)',fontsize=11); ax.set_ylim(-20,20)
        ax.set_xlabel('黄金价格 S',fontsize=11)
        lI,=ax2.plot(S,iss,'--',lw=2.8,color=COL_I)
        ax2.set_ylabel('机构利润 (%)',fontsize=11,color=COL_I)
        ax2.tick_params(axis='y',colors=COL_I); ax2.set_ylim(-4,5)
        lines+=[lK,lL,lI]; labels+=['K=300','L=375','机构利润']
        ax.legend(lines,labels,loc='upper center',ncol=3,fontsize=9,framealpha=0.9)
        ax.set_title('模块3: 散户 vs 机构 收益相关性（含保险垫）',fontweight='bold',fontsize=13)
        attach_cursor(lines[:2]+[lI])
        self._show(fig)

    # ══ 模块4: FuncAnimation（彻底解决闪退）══════════════
    def _mod4(self, sc, sp, vol):
        N=250; tp=np.linspace(0,P.T,N); dt=P.T/N
        Sfin,vv={1:(250,vol),2:(350,vol),3:(395,max(vol,0.22)),4:(300,vol)}[sc]
        np.random.seed(7)
        dW=vv*np.sqrt(dt)*np.random.randn(N)
        Sp=300*np.exp(np.cumsum((np.log(Sfin/300)/P.T-0.5*vv**2)*dt+dW))

        capA=c_ret_final(375,P.A)*100; capB=c_ret_final(375,P.B)*100
        preA=np.array([c_ret_path(Sp[k],tp[k],P.T,P.A)*100 for k in range(N)])
        preB=np.array([c_ret_path(Sp[k],tp[k],P.T,P.B)*100 for k in range(N)])
        preI=np.array([(i_ret(Sp[k])-0.0008*k*vv)*100 for k in range(N)])
        deltas=np.array([calc_delta(Sp[k],P.T-tp[k],vv,P.r) for k in range(N)])
        rebal=np.zeros(N); rebal[1:]=(deltas[1:]-deltas[:-1])*10

        KO=-1
        for k in range(N):
            if Sp[k]>=375: KO=k; break
        if KO>=0:
            preA[KO:]=capA; preB[KO:]=capB
            iPnL=preI[KO-1]/100 if KO>0 else 0
            for k in range(KO,N):
                if k>KO: iPnL+=deltas[k-1]*(Sp[k]-Sp[k-1])/300*0.08+P.r*dt*0.3
                preI[k]=iPnL*100

        # ── 建图 ──
        fig=Figure(figsize=(13,9),facecolor='white')
        axP=fig.add_axes([0.08,0.70,0.86,0.25])
        axR=fig.add_axes([0.08,0.40,0.86,0.25])
        axD=fig.add_axes([0.08,0.08,0.86,0.25])
        for ax in [axP,axR,axD]:
            ax.set_facecolor('#FAFAFA'); ax.grid(True,alpha=0.33)

        axP.axhline(375,ls='--',color='r',lw=2); axP.axhline(300,ls='--',color='k',lw=1.3)
        axP.text(0.06,377,'L=375',color='r',fontweight='bold',fontsize=9)
        axP.text(0.06,292,'K=300',color='k',fontweight='bold',fontsize=9)
        lnP,  =axP.plot([],[],'-', color=COL_S,lw=2.5,label='价格路径')
        lnBal,=axP.plot([],[],'o', ms=11,mfc='yellow',mec='k',mew=1.5,label='当前位置')
        axP.set_ylabel('St'); axP.set_xlim(0,P.T)
        axP.set_ylim(min(Sp)-20, max(max(Sp),400)+20)
        axP.set_title(f'模块4 情景{sc} — 黄金价格路径 (σ={vv:.2f})',fontweight='bold',fontsize=11)
        axP.legend(loc='upper left',fontsize=8,ncol=2)

        axR.axhline(0,ls=':',color='k',lw=0.8)
        axR.axhline(capA,ls=':',color=COL_A,lw=1); axR.axhline(capB,ls=':',color=COL_B,lw=1)
        axR.text(P.T+0.04,capA,f'上限A\n{capA:.2f}%',color=COL_A,fontsize=7,va='center')
        axR.text(P.T+0.04,capB,f'上限B\n{capB:.2f}%',color=COL_B,fontsize=7,va='center')
        lnRA,=axR.plot([],[],'-', color=COL_A,lw=2.8,label='散户A')
        lnRB,=axR.plot([],[],'-', color=COL_B,lw=2.8,label='散户B')
        lnRI,=axR.plot([],[],'--',color=COL_I,lw=2.5,label='机构')
        axR.set_ylabel('累计收益率 (%)'); axR.set_ylim(-17,22); axR.set_xlim(0,P.T)
        axR.set_title(f'散户A(绿≤{capA:.2f}%) / B(黄≤{capB:.2f}%) / 机构(紫)',fontweight='bold',fontsize=10)
        axR.legend(loc='upper left',ncol=3,fontsize=8)

        axD.axhline(0,ls=':',color='k',lw=0.8)
        lnDlt,=axD.plot([],[],'-', color=COL_D,lw=2.5,label='Delta 持仓')
        lnReb,=axD.plot([],[],'-', color='#3D70DD',lw=1.3,label='调仓量×10')
        axD.set_ylim(-2.5,1.5); axD.set_xlim(0,P.T)
        axD.set_xlabel('时间 (年)'); axD.set_ylabel('Delta / 调仓')
        axD.set_title('Delta 对冲仓位 & 调仓量×10  (Δ≈N(d₁)−N(d₃))',fontweight='bold',fontsize=10)
        axD.legend(loc='upper right',fontsize=8)

        # ── 必须先 _show 才能获得有效 canvas ──
        self._show(fig)

        # KO 标记状态（用列表避免 Python 闭包只读限制）
        ko_done=[False]

        # ── FuncAnimation 更新函数 ──────────────────────────
        def update(k):
            k = int(k)
            if k >= N: return lnP, lnBal, lnRA, lnRB, lnRI, lnDlt, lnReb

            sl = slice(0, k+1)
            lnP.set_data(tp[sl], Sp[sl])
            lnBal.set_data([tp[k]], [Sp[k]])
            lnRA.set_data(tp[sl], preA[sl])
            lnRB.set_data(tp[sl], preB[sl])
            lnRI.set_data(tp[sl], preI[sl])
            lnDlt.set_data(tp[sl], deltas[sl])
            lnReb.set_data(tp[sl], rebal[sl])

            if KO>=0 and k==KO and not ko_done[0]:
                ko_done[0]=True
                axP.plot(tp[k],375,'p',ms=26,mfc='red',mec='k',mew=2,zorder=10)
                axP.text(tp[k]+0.04,381,'⚡ 敲出!',color='red',fontsize=13,
                         fontweight='bold',
                         bbox=dict(facecolor='yellow',edgecolor='red',alpha=0.95))

            return lnP, lnBal, lnRA, lnRB, lnRI, lnDlt, lnReb

        # interval(ms) = 1000 / (sp * fps_factor)
        interval = max(8, int(600 / (sp * N / P.T)))

        # ── 创建 FuncAnimation，存到 self._anim 防止被 GC ──
        self._anim = FuncAnimation(
            fig,
            update,
            frames=np.arange(1, N),
            interval=interval,
            blit=False,       # blit=True 在 Qt5Agg 双Y轴下有bug
            repeat=False
        )
        # 通知 canvas 重新绘制（触发动画启动）
        self.canvas.draw()

# ══════════════════════════════════════════════════════════════
def main():
    app=QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont('Microsoft YaHei',10))
    win=GoldenFingerApp()
    win.show()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()
