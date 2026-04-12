#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
■ 金手指 Golden Finger — 散户/机构 + Delta 可视化
Authors: 3123007918 张懿哲 · 3123007910 李沐鑫 · 3223007924 黄凡绮
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.special import erf as sp_erf

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

try:
    import mplcursors
    HAS_MPLCURSORS = True
except ImportError:
    HAS_MPLCURSORS = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QPushButton, QFrame, QSizePolicy, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

# ══════════════════════════════════════════════════════════════
#  参数
# ══════════════════════════════════════════════════════════════
class Tranche:
    def __init__(self, issue, guar, PR, fee):
        self.issue = issue; self.guar = guar; self.PR = PR; self.fee = fee

class P:
    S0 = 300; L = 375; T = 3; r = 0.045
    A = Tranche(1000, 1000, 0.70, 0.028)
    B = Tranche(1100, 1000, 1.30, 0.040)

# ── 颜色 ──
COL_A = '#27B345'
COL_B = '#E6A800'
COL_I = '#7D1AB5'
COL_D = '#E65C00'
COL_S = '#1A4FCC'

# ══════════════════════════════════════════════════════════════
#  数学函数
# ══════════════════════════════════════════════════════════════
def payoff_call(ST, tr):
    if ST >= 375:  cy = 0.25
    elif ST > 300: cy = min(ST / 300 - 1, 0.25)
    else:          cy = 0.0
    return max(cy * tr.PR - tr.fee, -tr.fee)

def c_ret_final(ST, tr):
    if ST >= 375:  cy = 0.25
    elif ST > 300: cy = min(ST / 300 - 1, 0.25)
    else:          cy = 0.0
    return (tr.guar + cy * tr.PR * 1000 - tr.issue) / tr.issue - tr.fee

def c_ret_path(ST, t, T, tr):
    r_final = c_ret_final(ST, tr)
    cushion_gap = (tr.guar - tr.issue) / tr.issue
    smooth_S = max(0.0, min(1.0, (300 - ST) / 50))
    unrealized_frac = (1 - t / T) * (1 - smooth_S)
    r = r_final - cushion_gap * unrealized_frac
    return min(r, c_ret_final(375, tr))

def i_ret(ST):
    return 0.034 - (0.018 * np.exp(-((ST - 375) / 14) ** 2) + 0.004)

def calc_delta(S, tau, sig, r_rate):
    if tau <= 0: return 0.0
    tau_eff = max(tau, 0.05)
    d1 = (np.log(S / 300) + (r_rate + 0.5 * sig ** 2) * tau_eff) / (sig * np.sqrt(tau_eff))
    d3 = (np.log(S / 375) + (r_rate + 0.5 * sig ** 2) * tau_eff) / (sig * np.sqrt(tau_eff))
    Nd1 = 0.5 * (1 + sp_erf(d1 / np.sqrt(2)))
    Nd3 = 0.5 * (1 + sp_erf(d3 / np.sqrt(2)))
    d = Nd1 - Nd3
    if tau < 0.05: d *= tau / 0.05
    if S >= 375:   d = 0.0
    if S < 280:    d *= max(0.0, (S - 250) / 30)
    return float(max(0.0, min(d, 1.2)))

def sim_scene(Sfin, vv, seed):
    N = 250
    t = np.linspace(0, P.T, N)
    dt = P.T / N
    np.random.seed(seed)
    drift = np.log(Sfin / 300) / P.T
    dW = vv * np.sqrt(dt) * np.random.randn(N)
    S = 300 * np.exp(np.cumsum((drift - 0.5 * vv ** 2) * dt + dW))
    rA = np.zeros(N); rB = np.zeros(N)
    capA = c_ret_final(375, P.A); capB = c_ret_final(375, P.B)
    ev = -1; KO = False
    for k in range(N):
        if KO:
            rA[k] = capA * 100; rB[k] = capB * 100
        elif S[k] >= 375:
            KO = True; ev = k
            rA[k] = capA * 100; rB[k] = capB * 100
        else:
            rA[k] = c_ret_path(S[k], t[k], P.T, P.A) * 100
            rB[k] = c_ret_path(S[k], t[k], P.T, P.B) * 100
    return t, S, rA, rB, ev

# ══════════════════════════════════════════════════════════════
#  辅助：为轴添加 mplcursors 交互提示
# ══════════════════════════════════════════════════════════════
def _attach_cursor(lines):
    if HAS_MPLCURSORS and lines:
        cur = mplcursors.cursor(lines, hover=True)
        @cur.connect("add")
        def _(sel):
            x, y = sel.target
            sel.annotation.set_text(f'x={x:.3f}\ny={y:.3f}')
            sel.annotation.get_bbox_patch().set(fc='lightyellow', alpha=0.9)

# ══════════════════════════════════════════════════════════════
#  场景绘图（模块1 用）
# ══════════════════════════════════════════════════════════════
def plot_scene(fig, rect, t, S, aprod, bprod, ev, title, show_legend=False):
    ax  = fig.add_axes(rect)
    ax2 = ax.twinx()

    ax.axhline(375, ls='--', color='r', lw=1.2)
    ax.axhline(300, ls='--', color='k', lw=1.0)
    hS, = ax.plot(t, S, '-', color=COL_S, lw=2.0, label='S_t')
    ax.set_ylabel('S_t', color='#101066'); ax.tick_params(axis='y', colors='#101066')
    s_lo = min(min(S) - 10, 220)
    s_hi = max(max(S) + 10, 405)
    ax.set_ylim(s_lo, s_hi); ax.set_xlim(0, P.T)

    ax2.axhline(0, ls=':', color='k', lw=0.7)
    fA0 = -P.A.fee * 100; fB0 = -P.B.fee * 100
    ax2.axhline(fA0, ls='--', color=COL_A, lw=0.9)
    ax2.axhline(fB0, ls='--', color=COL_B, lw=0.9)
    ax2.text(P.T + 0.04, fA0, f'A费\n{fA0:.1f}%',
             color=COL_A, fontsize=6.5, fontweight='bold', va='center')
    ax2.text(P.T + 0.04, fB0, f'B费\n{fB0:.1f}%',
             color=COL_B, fontsize=6.5, fontweight='bold', va='center')
    hA, = ax2.plot(t, aprod, '-', color=COL_A, lw=2.4, label='A 产品总收益')
    hB, = ax2.plot(t, bprod, '-', color=COL_B, lw=2.4, label='B 产品总收益')
    ax2.set_ylabel('收益率 (%)'); ax2.set_ylim(-17, 22)

    if ev >= 0:
        ax.plot(t[ev], 375, 'p', ms=18, mfc='r', mec='k', mew=1.5)
        ax.text(t[ev] + 0.05, 380, '⚡敲出', color='r', fontweight='bold',
                bbox=dict(facecolor='yellow', edgecolor='red', alpha=0.9), fontsize=8.5)

    ax.set_xlabel('t (年)')
    ax.set_title(title, fontweight='bold', fontsize=10)
    if show_legend:
        ax.legend([hS, hA, hB], ['S_t', 'A 产品总收益', 'B 产品总收益'],
                  loc='lower left', fontsize=7)
    _attach_cursor([hS, hA, hB])
    return ax, ax2

# ══════════════════════════════════════════════════════════════
#  主窗口
# ══════════════════════════════════════════════════════════════
class GoldenFingerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('■ 金手指 Golden Finger — 散户/机构 + Delta 可视化')
        self.resize(1440, 880)
        self.setStyleSheet('background:#F3F3F8;')

        self._timer  = QTimer(self)
        self._timer.timeout.connect(self._anim_step)
        self._anim_k = 0
        self._anim_d = None
        self.canvas  = None
        self.toolbar = None

        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(8)

        root.addWidget(self._build_left())

        self.plot_area = QWidget(); self.plot_area.setStyleSheet('background:white;')
        self.plot_layout = QVBoxLayout(self.plot_area)
        self.plot_layout.setContentsMargins(2, 2, 2, 2)
        root.addWidget(self.plot_area, 1)

    # ─── 左侧控制面板 ───────────────────────────────────────────
    def _build_left(self):
        frame = QFrame()
        frame.setFixedWidth(232)
        frame.setStyleSheet('''
            QFrame { background:white; border:1px solid #C8C8D4; border-radius:8px; }
            QLabel { border:none; }
            QComboBox { border:1px solid #BBBBCC; border-radius:4px; padding:2px 4px; }
            QSlider::groove:horizontal { height:5px; background:#DDE; border-radius:3px; }
            QSlider::handle:horizontal { width:14px; height:14px; margin:-5px 0;
                                         background:#5599EE; border-radius:7px; }
        ''')
        vb = QVBoxLayout(frame)
        vb.setContentsMargins(10, 10, 10, 10); vb.setSpacing(5)

        ttl = QLabel('■ 金手指 Golden Finger\n散户/机构 + Delta 可视化')
        ttl.setStyleSheet('color:#B01515;font-size:13px;font-weight:bold;line-height:150%;')
        ttl.setWordWrap(True); vb.addWidget(ttl)
        vb.addWidget(self._sep())

        # ① 模块
        vb.addWidget(self._blabel('① 模块:'))
        self.dd_mod = QComboBox()
        self.dd_mod.addItems(['1-Payoff情景', '2-3D曲面', '3-双轴对比', '4-动画+Delta'])
        self.dd_mod.setCurrentIndex(3); vb.addWidget(self.dd_mod)

        # ② 档位
        vb.addWidget(self._blabel('② 档位:'))
        self.dd_tr = QComboBox()
        self.dd_tr.addItems(['仅A', '仅B', 'A+B对比'])
        self.dd_tr.setCurrentIndex(2); vb.addWidget(self.dd_tr)

        # ③ 情景
        vb.addWidget(self._blabel('③ 情景:'))
        self.dd_sc = QComboBox()
        self.dd_sc.addItems(['下跌→250', '上涨→350', '敲出→395', '随机'])
        self.dd_sc.setCurrentIndex(2); vb.addWidget(self.dd_sc)

        # ④ 速度
        self.lbl_sp = self._blabel('④ 速度: 5.0')
        vb.addWidget(self.lbl_sp)
        self.sld_sp = QSlider(Qt.Horizontal)
        self.sld_sp.setRange(10, 100); self.sld_sp.setValue(50)
        self.sld_sp.valueChanged.connect(lambda v: self.lbl_sp.setText(f'④ 速度: {v/10:.1f}'))
        vb.addWidget(self.sld_sp)

        # ⑤ σ
        self.lbl_vol = self._blabel('⑤ σ: 0.170')
        vb.addWidget(self.lbl_vol)
        self.sld_vol = QSlider(Qt.Horizontal)
        self.sld_vol.setRange(5, 40); self.sld_vol.setValue(17)
        self.sld_vol.valueChanged.connect(lambda v: self.lbl_vol.setText(f'⑤ σ: {v/100:.3f}'))
        vb.addWidget(self.sld_vol)

        vb.addSpacing(4)
        # ▶ 运行
        self.btn_run = QPushButton('▶ 运行 / 播放')
        self.btn_run.setStyleSheet('''
            QPushButton{background:#2E8844;color:white;font-weight:bold;font-size:14px;
                        border-radius:6px;padding:9px;border:none;}
            QPushButton:hover{background:#246B36;}
            QPushButton:pressed{background:#1C5229;}
        ''')
        self.btn_run.clicked.connect(self.run_module); vb.addWidget(self.btn_run)

        btn_clr = QPushButton('🗑  清除画布')
        btn_clr.setStyleSheet('''
            QPushButton{background:#D5D5E8;border-radius:5px;padding:6px;border:none;font-size:11px;}
            QPushButton:hover{background:#BEBECE;}
        ''')
        btn_clr.clicked.connect(self.clear_plot); vb.addWidget(btn_clr)

        vb.addWidget(self._sep())

        # 产品参数说明
        info = QLabel(
            '<b>── 产品参数 ──</b><br>'
            'S₀ = K = 300 &nbsp; L = 375<br>'
            'T = 3 年 &nbsp; r = 4.5%<br><br>'
            '<b>Tranche A</b>: 上限 <b>+14.7%</b><br>'
            '&nbsp; 发行1000 保本1000<br>'
            '&nbsp; PR 70%  费 2.8%<br><br>'
            '<b>Tranche B</b>: 上限 <b>+16.45%</b><br>'
            '&nbsp; 发行1100 保本1000<br>'
            '&nbsp; PR 130% 费 4%<br>'
            '&nbsp; 保险垫 90.91%'
        )
        info.setStyleSheet('background:#F8F8FB;font-size:9.5px;padding:7px;border-radius:5px;line-height:160%;')
        info.setWordWrap(True); vb.addWidget(info)

        if HAS_MPLCURSORS:
            tip = QLabel('💡 悬浮曲线可查看精确数值\n💡 3D图可鼠标拖拽旋转\n💡 工具栏支持缩放/平移/保存')
        else:
            tip = QLabel('💡 3D图可鼠标拖拽旋转\n💡 工具栏支持缩放/平移/保存')
        tip.setStyleSheet('background:#EEFAEE;font-size:8.5px;padding:5px;border-radius:4px;color:#1A5E2A;')
        tip.setWordWrap(True); vb.addWidget(tip)

        vb.addStretch()

        author = QLabel(
            '◆ Authors ◆<br>'
            '3123007918 张懿哲<br>'
            '3123007910 李沐鑫<br>'
            '3223007924 黄凡绮'
        )
        author.setStyleSheet('color:#4A0A60;font-size:8px;font-style:italic;font-weight:bold;')
        author.setAlignment(Qt.AlignCenter); vb.addWidget(author)
        return frame

    def _sep(self):
        s = QFrame(); s.setFrameShape(QFrame.HLine)
        s.setStyleSheet('color:#CCCCDD;'); return s

    def _blabel(self, text):
        l = QLabel(text); l.setFont(QFont('', -1, QFont.Bold)); return l

    # ─── 画布管理 ───────────────────────────────────────────────
    def clear_plot(self):
        self._timer.stop()
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater(); self.canvas = None
        if self.toolbar:
            self.plot_layout.removeWidget(self.toolbar)
            self.toolbar.deleteLater(); self.toolbar = None

    def _show(self, fig):
        self.clear_plot()
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavigationToolbar(canvas, self)
        self.plot_layout.addWidget(toolbar)
        self.plot_layout.addWidget(canvas)
        self.canvas = canvas; self.toolbar = toolbar
        canvas.draw()

    # ─── 分发运行 ───────────────────────────────────────────────
    def run_module(self):
        self._timer.stop()
        mod = self.dd_mod.currentIndex() + 1
        sc  = self.dd_sc.currentIndex() + 1
        tr  = self.dd_tr.currentIndex() + 1
        sp  = self.sld_sp.value() / 10.0
        vol = self.sld_vol.value() / 100.0
        {1: self._mod1, 2: self._mod2, 3: self._mod3, 4: self._mod4}[mod](sc, tr, sp, vol)

    # ══════════════════════════════════════════════════════════
    #  模块 1 — Payoff 情景
    # ══════════════════════════════════════════════════════════
    def _mod1(self, sc, tr, sp, vol):
        fig = Figure(figsize=(15, 9), facecolor='white')
        fig.suptitle('■ 金手指 Golden Finger — 模块1: Payoff 情景分析',
                     fontsize=14, fontweight='bold', color='#B01515', y=0.99)

        ax1 = fig.add_axes([0.06, 0.56, 0.41, 0.37])
        ax1.set_facecolor('#FAFAFA'); ax1.grid(True, alpha=0.35)

        S = np.linspace(200, 455, 500)
        rA = np.array([payoff_call(s, P.A) * 100 for s in S])
        rB = np.array([payoff_call(s, P.B) * 100 for s in S])

        hA, = ax1.plot(S, rA, lw=2.8, color=COL_A,
                       label=f'A  PR={P.A.PR*100:.0f}%  fee={P.A.fee*100:.1f}%')
        hB, = ax1.plot(S, rB, lw=2.8, color=COL_B,
                       label=f'B  PR={P.B.PR*100:.0f}%  fee={P.B.fee*100:.1f}%')
        ax1.axvline(300, ls='--', color='k', lw=1.2)
        ax1.axvline(375, ls='--', color='r', lw=1.4)
        ax1.axhline(0,   ls=':',  color='k', lw=0.8)
        ax1.text(302, 30.5, 'K=300', color='k', fontweight='bold', fontsize=8)
        ax1.text(377, 30.5, 'L=375', color='r', fontweight='bold', fontsize=8)

        feeA = -P.A.fee * 100; feeB = -P.B.fee * 100
        bepA = 300 * (1 + P.A.fee / P.A.PR)
        bepB = 300 * (1 + P.B.fee / P.B.PR)
        ax1.axhline(feeA, ls='--', color=COL_A, lw=1.0)
        ax1.axhline(feeB, ls='--', color=COL_B, lw=1.0)
        ax1.text(455, feeA, f'A起始\n{feeA:.1f}%',
                 color=COL_A, fontsize=7, fontweight='bold', va='center')
        ax1.text(455, feeB, f'B起始\n{feeB:.1f}%',
                 color=COL_B, fontsize=7, fontweight='bold', va='center')
        for bep, col, lbl in [(bepA, COL_A, 'A'), (bepB, COL_B, 'B')]:
            ax1.axvline(bep, ymax=0.48, ls='-.', color=col, lw=0.9)
            ax1.text(bep, -9.2, f'{lbl}盈亏\n平衡\n{bep:.0f}', color=col,
                     fontsize=7, ha='center')
        ax1.text(202, 29,
                 f'注: B(PR={P.B.PR*100:.0f}%) > A(PR={P.A.PR*100:.0f}%)\n'
                 f'S>306后B恒高于A',
                 fontsize=7, color='#444', va='top',
                 bbox=dict(facecolor='#F5F5F5', edgecolor='#CCC', alpha=0.9))
        ax1.set_xlabel('S_T'); ax1.set_ylabel('Payoff (%)')
        ax1.set_title(
            f'(a) 看涨期权 Payoff  '
            f'A:+{payoff_call(375,P.A)*100:.1f}%  B:+{payoff_call(375,P.B)*100:.1f}%',
            fontweight='bold', fontsize=10)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.set_ylim(-11, 33)
        _attach_cursor([hA, hB])

        t1, S1, ad1, bd1, _  = sim_scene(240, 0.16, 11)
        t2, S2, ad2, bd2, _  = sim_scene(355, 0.14, 22)
        t3, S3, ad3, bd3, ev = sim_scene(395, 0.24, 33)
        plot_scene(fig, [0.555, 0.56, 0.41, 0.37], t1, S1, ad1, bd1, -1,
                   '(b) 情景1: 破位下跌', show_legend=True)
        plot_scene(fig, [0.06,  0.07, 0.41, 0.37], t2, S2, ad2, bd2, -1,
                   '(c) 情景2: 温和上涨')
        plot_scene(fig, [0.555, 0.07, 0.41, 0.37], t3, S3, ad3, bd3, ev,
                   '(d) 情景3: 震荡触障敲出')
        self._show(fig)

    # ══════════════════════════════════════════════════════════
    #  模块 2 — 3D 曲面
    # ══════════════════════════════════════════════════════════
    def _mod2(self, *_):
        fig = Figure(figsize=(13, 8), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')

        S_arr = np.linspace(260, 420, 60)
        V_arr = np.linspace(0.05, 0.40, 50)
        Sg, Vg = np.meshgrid(S_arr, V_arr)
        sc_arr = 18 * (Sg < 375) + 38 * (Sg >= 375)
        HL = 14 * Vg * np.exp(-((Sg - 375) / sc_arr) ** 2) + 2.5 * Vg
        res = 6 * Vg * (Sg >= 375) * (1 - np.exp(-(Sg - 375) / 60))
        PnL = 34 - HL * 8 - res * 8

        surf = ax.plot_surface(Sg, Vg, PnL, cmap='turbo', alpha=0.93,
                               edgecolor='none', linewidth=0)
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08, label='发行方利润 (元)')
        ax.set_xlabel('黄金价格 S', labelpad=10)
        ax.set_ylabel('波动率 σ',   labelpad=10)
        ax.set_zlabel('发行方利润 (元)', labelpad=10)
        ax.set_title('模块2: 机构利润曲面  （可鼠标拖拽旋转 / 工具栏缩放）',
                     fontweight='bold', fontsize=13)
        ax.view_init(elev=30, azim=-40)
        fig.tight_layout()
        self._show(fig)

    # ══════════════════════════════════════════════════════════
    #  模块 3 — 双轴对比
    # ══════════════════════════════════════════════════════════
    def _mod3(self, sc, tr, sp, vol):
        fig = Figure(figsize=(13, 7), facecolor='white')
        ax  = fig.add_axes([0.08, 0.12, 0.80, 0.78])
        ax2 = ax.twinx()
        ax.set_facecolor('#FAFAFA'); ax.grid(True, alpha=0.35)

        S = np.linspace(220, 430, 400)
        rA  = np.array([c_ret_final(s, P.A) * 100 for s in S])
        rB  = np.array([c_ret_final(s, P.B) * 100 for s in S])
        iss = np.array([i_ret(s) * 100 for s in S])

        lines = []; labels = []
        if tr != 2:
            l, = ax.plot(S, rA, lw=2.8, color=COL_A, label='散户A')
            lines.append(l); labels.append('散户A')
        if tr != 1:
            l, = ax.plot(S, rB, lw=2.8, color=COL_B, label='散户B')
            lines.append(l); labels.append('散户B')

        lK = ax.axvline(300, ls='--', color='k', lw=1.4, label='K=300')
        lL = ax.axvline(375, ls='--', color='r', lw=1.4, label='L=375')
        ax.set_ylabel('散户收益 (%)', fontsize=11)
        ax.set_ylim(-20, 20); ax.set_xlabel('黄金价格 S', fontsize=11)
        ax.text(302, 18.5, 'K=300', fontsize=8, color='k', fontweight='bold')
        ax.text(377, 18.5, 'L=375', fontsize=8, color='r', fontweight='bold')

        lI, = ax2.plot(S, iss, '--', lw=2.8, color=COL_I, label='机构利润')
        ax2.set_ylabel('机构利润 (%)', fontsize=11, color=COL_I)
        ax2.tick_params(axis='y', colors=COL_I)
        ax2.set_ylim(-4, 5)

        lines += [lK, lL, lI]
        labels += ['K=300', 'L=375', '机构利润']
        ax.legend(lines, labels, loc='upper center', ncol=3, fontsize=9,
                  framealpha=0.9)
        ax.set_title('模块3: 散户 vs 机构 收益相关性（含保险垫）',
                     fontweight='bold', fontsize=13)
        _attach_cursor(lines[:2] + [lI])
        self._show(fig)

    # ══════════════════════════════════════════════════════════
    #  模块 4 — 动画 + Delta
    # ══════════════════════════════════════════════════════════
    def _mod4(self, sc, sp, vol):
        N = 250
        tp = np.linspace(0, P.T, N)
        dt = P.T / N

        if sc == 1:   Sfin = 250; vv = vol
        elif sc == 2: Sfin = 350; vv = vol
        elif sc == 3: Sfin = 395; vv = max(vol, 0.22)
        else:         Sfin = 300; vv = vol

        drift = np.log(Sfin / 300) / P.T
        np.random.seed(7)
        dW = vv * np.sqrt(dt) * np.random.randn(N)
        Sp = 300 * np.exp(np.cumsum((drift - 0.5 * vv ** 2) * dt + dW))

        capA = c_ret_final(375, P.A) * 100
        capB = c_ret_final(375, P.B) * 100

        preA = np.array([c_ret_path(Sp[k], tp[k], P.T, P.A) * 100 for k in range(N)])
        preB = np.array([c_ret_path(Sp[k], tp[k], P.T, P.B) * 100 for k in range(N)])
        preI = np.array([(i_ret(Sp[k]) - 0.0008 * k * vv) * 100 for k in range(N)])

        deltas = np.array([calc_delta(Sp[k], P.T - tp[k], vv, P.r) for k in range(N)])
        rebal  = np.zeros(N)
        rebal[1:] = (deltas[1:] - deltas[:-1]) * 10

        # KO 锁定
        KO_idx = -1
        for k in range(N):
            if Sp[k] >= 375: KO_idx = k; break
        if KO_idx >= 0:
            preA[KO_idx:] = capA; preB[KO_idx:] = capB
            iPnL = preI[KO_idx - 1] / 100 if KO_idx > 0 else 0
            for k in range(KO_idx, N):
                if k > KO_idx:
                    dS = Sp[k] - Sp[k - 1]
                    iPnL += deltas[k - 1] * (dS / 300) * 0.08 + P.r * dt * 0.3
                preI[k] = iPnL * 100

        # ── 建图 ──
        fig = Figure(figsize=(13, 9), facecolor='white')
        axP = fig.add_axes([0.08, 0.70, 0.86, 0.25])
        axR = fig.add_axes([0.08, 0.40, 0.86, 0.25])
        axD = fig.add_axes([0.08, 0.08, 0.86, 0.25])
        for ax in [axP, axR, axD]:
            ax.set_facecolor('#FAFAFA'); ax.grid(True, alpha=0.33)

        # 价格图
        axP.axhline(375, ls='--', color='r', lw=2.0)
        axP.axhline(300, ls='--', color='k', lw=1.3)
        axP.text(0.06, 377, 'L=375', color='r', fontweight='bold', fontsize=9)
        axP.text(0.06, 292, 'K=300', color='k', fontweight='bold', fontsize=9)
        self.ln_price, = axP.plot([], [], '-', color=COL_S, lw=2.5, label='价格路径')
        self.ln_ball,  = axP.plot([], [], 'o', ms=11, mfc='yellow', mec='k', mew=1.5, label='当前位置')
        axP.set_ylabel('S_t'); axP.set_xlim(0, P.T)
        axP.set_ylim(min(Sp) - 20, max(max(Sp), 400) + 20)
        axP.set_title(f'模块4 情景{sc} — 黄金价格路径 (σ={vv:.2f})', fontweight='bold', fontsize=11)
        axP.legend(loc='upper left', fontsize=8, ncol=2)

        # 收益图
        axR.axhline(0,    ls=':', color='k', lw=0.8)
        axR.axhline(capA, ls=':', color=COL_A, lw=1.0)
        axR.axhline(capB, ls=':', color=COL_B, lw=1.0)
        axR.text(P.T + 0.04, capA, f'上限A\n{capA:.2f}%', color=COL_A, fontsize=7, va='center')
        axR.text(P.T + 0.04, capB, f'上限B\n{capB:.2f}%', color=COL_B, fontsize=7, va='center')
        self.ln_rA, = axR.plot([], [], '-',  color=COL_A, lw=2.8, label='散户A')
        self.ln_rB, = axR.plot([], [], '-',  color=COL_B, lw=2.8, label='散户B')
        self.ln_rI, = axR.plot([], [], '--', color=COL_I, lw=2.5, label='机构')
        axR.set_ylabel('累计收益率 (%)'); axR.set_ylim(-17, 22); axR.set_xlim(0, P.T)
        axR.set_title(f'散户A (绿 ≤{capA:.2f}%) / B (黄 ≤{capB:.2f}%) / 机构 (紫)',
                      fontweight='bold', fontsize=10)
        axR.legend(loc='upper left', ncol=3, fontsize=8)

        # Delta 图
        axD.axhline(0, ls=':', color='k', lw=0.8)
        self.ln_dlt, = axD.plot([], [], '-', color=COL_D, lw=2.5, label='Delta 持仓')
        self.ln_reb, = axD.plot([], [], '-', color='#3D70DD', lw=1.3, label='调仓量×10')
        axD.set_ylim(-2.5, 1.5); axD.set_xlim(0, P.T)
        axD.set_xlabel('时间 (年)'); axD.set_ylabel('Delta / 调仓')
        axD.set_title('Delta 对冲仓位 & 调仓量×10  (Δ ≈ N(d₁)−N(d₃))',
                      fontweight='bold', fontsize=10)
        axD.legend(loc='upper right', fontsize=8)

        self._show(fig)

        # 动画状态
        self._anim_d = dict(
            tp=tp, Sp=Sp, preA=preA, preB=preB, preI=preI,
            deltas=deltas, rebal=rebal, N=N,
            KO_idx=KO_idx, axP=axP, sc=sc, vv=vv,
            ko_marked=False
        )
        self._anim_k = 1
        interval = max(8, int(80 / sp))
        self._timer.start(interval)

    def _anim_step(self):
        d = self._anim_d
        if d is None: return
        k = self._anim_k
        if k >= d['N']:
            self._timer.stop(); return

        tp = d['tp'][:k + 1]; Sp = d['Sp'][:k + 1]
        self.ln_price.set_data(tp, Sp)
        self.ln_ball.set_data([tp[-1]], [Sp[-1]])
        self.ln_rA.set_data(tp, d['preA'][:k + 1])
        self.ln_rB.set_data(tp, d['preB'][:k + 1])
        self.ln_rI.set_data(tp, d['preI'][:k + 1])
        self.ln_dlt.set_data(tp, d['deltas'][:k + 1])
        self.ln_reb.set_data(tp, d['rebal'][:k + 1])

        # 敲出标记
        if d['KO_idx'] >= 0 and k == d['KO_idx'] and not d['ko_marked']:
            d['ko_marked'] = True
            axP = d['axP']
            axP.plot(tp[-1], 375, 'p', ms=26, mfc='red', mec='k', mew=2,
                     zorder=10)
            axP.text(tp[-1] + 0.04, 380, '⚡ 敲出!', color='red',
                     fontsize=13, fontweight='bold',
                     bbox=dict(facecolor='yellow', edgecolor='red', alpha=0.95))

        self.canvas.draw_idle()
        self._anim_k += 1


# ══════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    # 中文字体
    font = QFont('Microsoft YaHei', 10)
    if not font.exactMatch():
        font = QFont('SimHei', 10)
    app.setFont(font)
    win = GoldenFingerApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
