#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
■ 金手指 Golden Finger — 散户/机构 + Delta 可视化
Authors: 3123007918 张懿哲 · 3123007910 李沐鑫 · 3223007924 黄凡绮
"""

import sys, os, warnings, traceback
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.special import erf as sp_erf

import matplotlib
matplotlib.use('TkAgg')   # 最稳定后端，PyInstaller 零问题
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong',
    'Arial Unicode MS', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

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

CA='#27B345'; CB='#E6A800'; CI='#7D1AB5'; CD='#E65C00'; CS='#1A4FCC'

# ══════════════════════════════════════════════════════════════
#  数学函数
# ══════════════════════════════════════════════════════════════
def payoff_call(ST, tr):
    cy=0.25 if ST>=375 else(min(ST/300-1,0.25)if ST>300 else 0.)
    return max(cy*tr.PR-tr.fee,-tr.fee)

def c_ret_final(ST, tr):
    cy=0.25 if ST>=375 else(min(ST/300-1,0.25)if ST>300 else 0.)
    return(tr.guar+cy*tr.PR*1000-tr.issue)/tr.issue-tr.fee

def c_ret_path(ST,t,T,tr):
    rf=c_ret_final(ST,tr); gap=(tr.guar-tr.issue)/tr.issue
    sm=max(0.,min(1.,(300-ST)/50))
    return min(rf-gap*(1-t/T)*(1-sm),c_ret_final(375,tr))

def i_ret(ST):
    return 0.034-(0.018*np.exp(-((ST-375)/14)**2)+0.004)

def calc_delta(S,tau,sig,rr):
    if tau<=0: return 0.
    te=max(tau,0.05)
    d1=(np.log(S/300)+(rr+.5*sig**2)*te)/(sig*np.sqrt(te))
    d3=(np.log(S/375)+(rr+.5*sig**2)*te)/(sig*np.sqrt(te))
    d=.5*(1+sp_erf(d1/np.sqrt(2)))-.5*(1+sp_erf(d3/np.sqrt(2)))
    if tau<0.05: d*=tau/0.05
    if S>=375:   d=0.
    if S<280:    d*=max(0.,(S-250)/30)
    return float(max(0.,min(d,1.2)))

def sim_scene(Sfin,vv,seed):
    N=250; t=np.linspace(0,P.T,N); dt=P.T/N
    np.random.seed(seed)
    S=300*np.exp(np.cumsum((np.log(Sfin/300)/P.T-.5*vv**2)*dt+vv*np.sqrt(dt)*np.random.randn(N)))
    rA=np.zeros(N); rB=np.zeros(N)
    capA=c_ret_final(375,P.A); capB=c_ret_final(375,P.B)
    ev=-1; KO=False
    for k in range(N):
        if KO: rA[k]=capA*100; rB[k]=capB*100
        elif S[k]>=375: KO=True; ev=k; rA[k]=capA*100; rB[k]=capB*100
        else:
            rA[k]=c_ret_path(S[k],t[k],P.T,P.A)*100
            rB[k]=c_ret_path(S[k],t[k],P.T,P.B)*100
    return t,S,rA,rB,ev

def plot_scene(fig,rect,t,S,ap,bp,ev,title,legend=False):
    ax=fig.add_axes(rect); ax2=ax.twinx()
    ax.axhline(375,ls='--',color='r',lw=1.2)
    ax.axhline(300,ls='--',color='k',lw=1.)
    hS,=ax.plot(t,S,color=CS,lw=2.,label='St')
    ax.set_ylabel('St'); ax.set_xlim(0,P.T)
    ax.set_ylim(min(min(S)-10,220),max(max(S)+10,405))
    ax2.axhline(0,ls=':',color='k',lw=.7)
    ax2.axhline(-P.A.fee*100,ls='--',color=CA,lw=.9)
    ax2.axhline(-P.B.fee*100,ls='--',color=CB,lw=.9)
    hA,=ax2.plot(t,ap,color=CA,lw=2.4,label='A')
    hB,=ax2.plot(t,bp,color=CB,lw=2.4,label='B')
    ax2.set_ylabel('收益率 (%)'); ax2.set_ylim(-17,22)
    if ev>=0:
        ax.plot(t[ev],375,'p',ms=16,mfc='r',mec='k',mew=1.5)
        ax.text(t[ev]+.05,380,'⚡敲出',color='r',fontweight='bold',
                bbox=dict(facecolor='yellow',edgecolor='red',alpha=.9),fontsize=8)
    ax.set_xlabel('t (年)'); ax.set_title(title,fontweight='bold',fontsize=10)
    if legend:
        ax.legend([hS,hA,hB],['St','A','B'],loc='lower left',fontsize=7)

# ══════════════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('■ 金手指 Golden Finger — 散户/机构 + Delta 可视化')
        self.geometry('1440x880')
        self.configure(bg='#EEEEF6')
        self.resizable(True, True)

        self._after_id = None   # 动画 after() id
        self._anim_d   = None
        self._canvas_widget = None
        self._toolbar_widget = None

        self._build_ui()

    # ── UI 布局 ────────────────────────────────────────────
    def _build_ui(self):
        # 顶部标题
        title_bar = tk.Frame(self, bg='#1A1A2E', height=42)
        title_bar.pack(fill='x', side='top')
        tk.Label(title_bar,
                 text='■ 金手指 Golden Finger — 散户/机构 + Delta 可视化',
                 font=('Microsoft YaHei',14,'bold'),
                 fg='#FFD700', bg='#1A1A2E').pack(side='left', padx=14, pady=6)

        main = tk.Frame(self, bg='#EEEEF6')
        main.pack(fill='both', expand=True)

        # 左侧控制面板
        left = tk.Frame(main, bg='#1A1A2E', width=230)
        left.pack(side='left', fill='y', padx=(8,4), pady=8)
        left.pack_propagate(False)
        self._build_left(left)

        # 右侧绘图区
        self.plot_frame = tk.Frame(main, bg='white', relief='flat')
        self.plot_frame.pack(side='left', fill='both', expand=True,
                             padx=(4,8), pady=8)

    def _lbl(self, parent, text, fg='#90CAF9', size=10, bold=True):
        f = ('Microsoft YaHei', size, 'bold') if bold else ('Microsoft YaHei', size)
        tk.Label(parent, text=text, font=f, fg=fg,
                 bg='#1A1A2E', anchor='w').pack(fill='x', padx=10, pady=(8,1))

    def _build_left(self, parent):
        # 副标题
        tk.Label(parent, text='散户 / 机构 + Delta 可视化',
                 font=('Microsoft YaHei',9), fg='#AAC4FF',
                 bg='#1A1A2E').pack(padx=10, pady=(10,4))

        sep = tk.Frame(parent, bg='#334466', height=1)
        sep.pack(fill='x', padx=10, pady=4)

        # ① 模块
        self._lbl(parent, '① 模块')
        self.var_mod = tk.StringVar(value='4-动画+Delta')
        cb1 = ttk.Combobox(parent, textvariable=self.var_mod, state='readonly',
            values=['1-Payoff情景','2-3D曲面','3-双轴对比','4-动画+Delta'])
        cb1.pack(fill='x', padx=10, pady=2)

        # ② 档位
        self._lbl(parent, '② 档位')
        self.var_tr = tk.StringVar(value='A+B对比')
        ttk.Combobox(parent, textvariable=self.var_tr, state='readonly',
            values=['仅A','仅B','A+B对比']).pack(fill='x', padx=10, pady=2)

        # ③ 情景
        self._lbl(parent, '③ 情景')
        self.var_sc = tk.StringVar(value='敲出→395')
        ttk.Combobox(parent, textvariable=self.var_sc, state='readonly',
            values=['下跌→250','上涨→350','敲出→395','随机']).pack(fill='x', padx=10, pady=2)

        # ④ 速度
        self._lbl(parent, '④ 速度: 5.0')
        self.lbl_sp = parent.winfo_children()[-1]
        self.var_sp = tk.DoubleVar(value=5.0)
        tk.Scale(parent, from_=1, to=10, resolution=0.5, orient='horizontal',
                 variable=self.var_sp, bg='#1A1A2E', fg='#64B5F6',
                 troughcolor='#2A4A6A', highlightthickness=0,
                 command=lambda v: self.lbl_sp.config(
                     text=f'④ 速度: {float(v):.1f}')
                 ).pack(fill='x', padx=10, pady=2)

        # ⑤ σ
        self._lbl(parent, '⑤ σ: 0.170')
        self.lbl_vol = parent.winfo_children()[-1]
        self.var_vol = tk.DoubleVar(value=0.17)
        tk.Scale(parent, from_=0.05, to=0.40, resolution=0.005, orient='horizontal',
                 variable=self.var_vol, bg='#1A1A2E', fg='#64B5F6',
                 troughcolor='#2A4A6A', highlightthickness=0,
                 command=lambda v: self.lbl_vol.config(
                     text=f'⑤ σ: {float(v):.3f}')
                 ).pack(fill='x', padx=10, pady=2)

        sep2 = tk.Frame(parent, bg='#334466', height=1)
        sep2.pack(fill='x', padx=10, pady=8)

        # ▶ 运行按钮
        tk.Button(parent, text='▶  运行 / 播放',
                  font=('Microsoft YaHei',13,'bold'),
                  bg='#2E7D32', fg='white', activebackground='#388E3C',
                  relief='flat', padx=8, pady=8,
                  command=self.run_module
                  ).pack(fill='x', padx=10, pady=3)

        tk.Button(parent, text='🗑  清除画布',
                  font=('Microsoft YaHei',10),
                  bg='#2A3A5A', fg='#CFD8DC', activebackground='#334466',
                  relief='flat', padx=6, pady=5,
                  command=self.clear_plot
                  ).pack(fill='x', padx=10, pady=2)

        sep3 = tk.Frame(parent, bg='#334466', height=1)
        sep3.pack(fill='x', padx=10, pady=8)

        # 产品参数
        info_text = (
            '── 产品参数 ──\n'
            'S₀=K=300  L=375\n'
            'T=3年  r=4.5%\n\n'
            'Tranche A: 上限+14.7%\n'
            '  发行1000 保本1000\n'
            '  PR70%  费2.8%\n\n'
            'Tranche B: 上限+16.45%\n'
            '  发行1100 保本1000\n'
            '  PR130% 费4%\n'
            '  保险垫90.91%'
        )
        tk.Label(parent, text=info_text, font=('Microsoft YaHei',9),
                 fg='#B0BEC5', bg='#0F1E3A', justify='left',
                 padx=8, pady=8, relief='flat'
                 ).pack(fill='x', padx=10, pady=2)

        sep4 = tk.Frame(parent, bg='#334466', height=1)
        sep4.pack(fill='x', padx=10, pady=8)

        # 作者区
        author_frame = tk.Frame(parent, bg='#2A1A00')
        author_frame.pack(fill='x', padx=10, pady=4)
        tk.Label(author_frame, text='◆  Authors  ◆',
                 font=('Microsoft YaHei',12,'bold'),
                 fg='#FFD700', bg='#2A1A00').pack(pady=(6,2))
        for nm in ['3123007918  张懿哲','3123007910  李沐鑫','3223007924  黄凡绮']:
            tk.Label(author_frame, text=nm,
                     font=('Microsoft YaHei',11),
                     fg='#E0E0E0', bg='#2A1A00').pack(pady=1)
        tk.Label(author_frame, text='', bg='#2A1A00').pack(pady=3)

    # ── 清除画布 ──────────────────────────────────────────
    def clear_plot(self):
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        self._anim_d = None
        for w in self.plot_frame.winfo_children():
            w.destroy()
        self._canvas_widget = None
        self._toolbar_widget = None

    def _show_fig(self, fig):
        """在右侧显示 matplotlib 图，含导航工具栏"""
        self.clear_plot()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()
        toolbar.pack(side='top', fill='x')
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self._canvas_widget = canvas

    # ── 分发运行 ──────────────────────────────────────────
    def run_module(self):
        mod_str = self.var_mod.get()
        sc_map  = {'下跌→250':1,'上涨→350':2,'敲出→395':3,'随机':4}
        tr_map  = {'仅A':1,'仅B':2,'A+B对比':3}
        mod = int(mod_str[0])
        sc  = sc_map.get(self.var_sc.get(), 3)
        tr  = tr_map.get(self.var_tr.get(), 3)
        sp  = self.var_sp.get()
        vol = self.var_vol.get()
        {1:self._mod1, 2:self._mod2, 3:self._mod3, 4:self._mod4}[mod](sc,tr,sp,vol)

    # ══ 模块1 ══════════════════════════════════════════════
    def _mod1(self, sc, tr, sp, vol):
        fig=Figure(figsize=(15,9),facecolor='white')
        fig.suptitle('■ 金手指 Golden Finger — 模块1: Payoff 情景分析',
                     fontsize=13,fontweight='bold',color='#B01515',y=0.99)
        ax1=fig.add_axes([0.06,0.56,0.41,0.37])
        ax1.set_facecolor('#FAFAFA'); ax1.grid(True,alpha=0.35)
        S=np.linspace(200,455,500)
        rA=np.array([payoff_call(s,P.A)*100 for s in S])
        rB=np.array([payoff_call(s,P.B)*100 for s in S])
        ax1.plot(S,rA,lw=2.8,color=CA,label=f'A PR={P.A.PR*100:.0f}% fee={P.A.fee*100:.1f}%')
        ax1.plot(S,rB,lw=2.8,color=CB,label=f'B PR={P.B.PR*100:.0f}% fee={P.B.fee*100:.1f}%')
        ax1.axvline(300,ls='--',color='k',lw=1.2); ax1.axvline(375,ls='--',color='r',lw=1.4)
        ax1.axhline(0,ls=':',color='k',lw=0.8)
        ax1.text(302,30,'K=300',color='k',fontweight='bold',fontsize=8)
        ax1.text(377,30,'L=375',color='r',fontweight='bold',fontsize=8)
        ax1.axhline(-P.A.fee*100,ls='--',color=CA,lw=1.)
        ax1.axhline(-P.B.fee*100,ls='--',color=CB,lw=1.)
        bepA=300*(1+P.A.fee/P.A.PR); bepB=300*(1+P.B.fee/P.B.PR)
        for bep,col,lbl in[(bepA,CA,'A'),(bepB,CB,'B')]:
            ax1.axvline(bep,ymax=0.48,ls='-.',color=col,lw=0.9)
            ax1.text(bep,-9,f'{lbl}\n平衡\n{bep:.0f}',color=col,fontsize=7,ha='center')
        ax1.set_xlabel('S_T'); ax1.set_ylabel('Payoff (%)')
        ax1.set_title(f'(a) 看涨期权 Payoff  A:+{payoff_call(375,P.A)*100:.1f}%  '
                      f'B:+{payoff_call(375,P.B)*100:.1f}%',fontweight='bold',fontsize=10)
        ax1.legend(loc='upper left',fontsize=8); ax1.set_ylim(-11,33)
        t1,S1,ad1,bd1,_ =sim_scene(240,0.16,11)
        t2,S2,ad2,bd2,_ =sim_scene(355,0.14,22)
        t3,S3,ad3,bd3,ev=sim_scene(395,0.24,33)
        plot_scene(fig,[0.555,0.56,0.41,0.37],t1,S1,ad1,bd1,-1,'(b) 情景1: 破位下跌',legend=True)
        plot_scene(fig,[0.06,0.07,0.41,0.37], t2,S2,ad2,bd2,-1,'(c) 情景2: 温和上涨')
        plot_scene(fig,[0.555,0.07,0.41,0.37],t3,S3,ad3,bd3,ev,'(d) 情景3: 震荡触障敲出')
        self._show_fig(fig)

    # ══ 模块2 ══════════════════════════════════════════════
    def _mod2(self, *_):
        fig=Figure(figsize=(13,8),facecolor='white')
        ax=fig.add_subplot(111,projection='3d')
        Sa=np.linspace(260,420,60); Va=np.linspace(0.05,0.40,50)
        Sg,Vg=np.meshgrid(Sa,Va)
        sca=18*(Sg<375)+38*(Sg>=375)
        HL=14*Vg*np.exp(-((Sg-375)/sca)**2)+2.5*Vg
        res=6*Vg*(Sg>=375)*(1-np.exp(-(Sg-375)/60))
        PnL=34-HL*8-res*8
        surf=ax.plot_surface(Sg,Vg,PnL,cmap='turbo',alpha=0.93,edgecolor='none')
        fig.colorbar(surf,ax=ax,shrink=0.5,pad=0.08,label='发行方利润 (元)')
        ax.set_xlabel('黄金价格 S',labelpad=10)
        ax.set_ylabel('波动率 σ',labelpad=10)
        ax.set_zlabel('发行方利润 (元)',labelpad=10)
        ax.set_title('模块2: 机构利润曲面  (鼠标拖拽旋转)',fontweight='bold',fontsize=13)
        ax.view_init(elev=30,azim=-40); fig.tight_layout()
        self._show_fig(fig)

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
        if tr!=2: l,=ax.plot(S,rA,lw=2.8,color=CA); lines.append(l); labels.append('散户A')
        if tr!=1: l,=ax.plot(S,rB,lw=2.8,color=CB); lines.append(l); labels.append('散户B')
        lK=ax.axvline(300,ls='--',color='k',lw=1.4)
        lL=ax.axvline(375,ls='--',color='r',lw=1.4)
        ax.set_ylabel('散户收益 (%)',fontsize=11); ax.set_ylim(-20,20)
        ax.set_xlabel('黄金价格 S',fontsize=11)
        lI,=ax2.plot(S,iss,'--',lw=2.8,color=CI)
        ax2.set_ylabel('机构利润 (%)',fontsize=11,color=CI)
        ax2.tick_params(axis='y',colors=CI); ax2.set_ylim(-4,5)
        lines+=[lK,lL,lI]; labels+=['K=300','L=375','机构利润']
        ax.legend(lines,labels,loc='upper center',ncol=3,fontsize=9,framealpha=0.9)
        ax.set_title('模块3: 散户 vs 机构 收益相关性（含保险垫）',fontweight='bold',fontsize=13)
        self._show_fig(fig)

    # ══ 模块4：tkinter after() 驱动，零崩溃风险 ══════════
    def _mod4(self, sc, tr, sp, vol):
        N=250; tp=np.linspace(0,P.T,N); dt=P.T/N
        Sfin,vv={1:(250,vol),2:(350,vol),3:(395,max(vol,0.22)),4:(300,vol)}[sc]
        np.random.seed(7)
        dW=vv*np.sqrt(dt)*np.random.randn(N)
        Sp=300*np.exp(np.cumsum((np.log(Sfin/300)/P.T-.5*vv**2)*dt+dW))
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

        # 建图（空线）
        fig=Figure(figsize=(13,9),facecolor='white')
        axP=fig.add_axes([0.08,0.70,0.86,0.25])
        axR=fig.add_axes([0.08,0.40,0.86,0.25])
        axD=fig.add_axes([0.08,0.08,0.86,0.25])
        for ax in[axP,axR,axD]:
            ax.set_facecolor('#FAFAFA'); ax.grid(True,alpha=0.33)

        axP.axhline(375,ls='--',color='r',lw=2); axP.axhline(300,ls='--',color='k',lw=1.3)
        axP.text(0.06,377,'L=375',color='r',fontweight='bold',fontsize=9)
        axP.text(0.06,292,'K=300',color='k',fontweight='bold',fontsize=9)
        lnP, =axP.plot([],[],'-', color=CS,lw=2.5,label='价格路径')
        lnBal,=axP.plot([],[],'o', ms=10,mfc='yellow',mec='k',mew=1.5)
        axP.set_ylabel('St'); axP.set_xlim(0,P.T)
        axP.set_ylim(min(Sp)-20,max(max(Sp),400)+20)
        axP.set_title(f'模块4 情景{sc} — 黄金价格路径 (σ={vv:.2f})',fontweight='bold',fontsize=11)
        axP.legend(loc='upper left',fontsize=8,ncol=2)

        axR.axhline(0,ls=':',color='k',lw=0.8)
        axR.axhline(capA,ls=':',color=CA,lw=1); axR.axhline(capB,ls=':',color=CB,lw=1)
        axR.text(P.T+0.04,capA,f'上限A\n{capA:.2f}%',color=CA,fontsize=7,va='center')
        axR.text(P.T+0.04,capB,f'上限B\n{capB:.2f}%',color=CB,fontsize=7,va='center')
        lnRA,=axR.plot([],[],'-', color=CA,lw=2.8,label='散户A')
        lnRB,=axR.plot([],[],'-', color=CB,lw=2.8,label='散户B')
        lnRI,=axR.plot([],[],'--',color=CI,lw=2.5,label='机构')
        axR.set_ylabel('累计收益率 (%)'); axR.set_ylim(-17,22); axR.set_xlim(0,P.T)
        axR.set_title(f'散户A(绿≤{capA:.2f}%) / B(黄≤{capB:.2f}%) / 机构(紫)',
                      fontweight='bold',fontsize=10)
        axR.legend(loc='upper left',ncol=3,fontsize=8)

        axD.axhline(0,ls=':',color='k',lw=0.8)
        lnDlt,=axD.plot([],[],'-', color=CD,lw=2.5,label='Delta 持仓')
        lnReb,=axD.plot([],[],'-', color='#3D70DD',lw=1.3,label='调仓量×10')
        axD.set_ylim(-2.5,1.5); axD.set_xlim(0,P.T)
        axD.set_xlabel('时间 (年)'); axD.set_ylabel('Delta / 调仓')
        axD.set_title('Delta 对冲仓位 & 调仓量×10',fontweight='bold',fontsize=10)
        axD.legend(loc='upper right',fontsize=8)

        self._show_fig(fig)
        canvas = self._canvas_widget

        interval_ms = max(10, int(600/(sp*N/P.T)))
        ko_done = [False]

        # 用 tkinter 原生 after() 驱动动画
        def tick(k=1):
            if k >= N:
                return
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
                axP.plot(tp[k],375,'p',ms=24,mfc='red',mec='k',mew=2,zorder=10)
                axP.text(tp[k]+0.04,381,'⚡ 敲出!',color='red',fontsize=13,
                         fontweight='bold',
                         bbox=dict(facecolor='yellow',edgecolor='red',alpha=0.95))
            canvas.draw()
            self._after_id = self.after(interval_ms, tick, k+1)

        self._after_id = self.after(interval_ms, tick, 1)

# ══════════════════════════════════════════════════════════════
def main():
    try:
        app = App()
        app.mainloop()
    except Exception:
        try:
            messagebox.showerror('错误', traceback.format_exc())
        except Exception:
            pass

if __name__ == '__main__':
    main()
