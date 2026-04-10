"""
figure_selection_criteria.py

Illustrates the 3 photon-selection cases used to decide which photons
are passed to xifusim for simulation.

Thresholds (v5_20250621):
  close_dist_toxifusim = 200  samples   (≈ 1.5 ms)
  secondary_samples    = 1563 samples   (≈ 12 ms)
  HR_samples           = 8192 samples   (≈ 63 ms)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Colour palette ──────────────────────────────────────────────────────────
COL = {
    'prox': '#1f77b4',   # blue  — case 1: close neighbor ≤ 200 samples
    'D':    '#ff7f0e',   # orange — case 2: follows selected predecessor
    'E':    '#9467bd',   # purple — case 3: precedes a close pair
    'none': '#aaaaaa',   # grey  — not selected
}

STEM_TOP  = 0.48
LABEL_OFF = 0.08


# ── Helper drawing functions ────────────────────────────────────────────────

def photon(ax, x, case, top=STEM_TOP, lw=2.5):
    """Selected photon: solid stem + filled circle."""
    c = COL[case]
    ax.plot([x, x], [0, top], color=c, lw=lw, solid_capstyle='round', zorder=4)
    ax.plot(x, top, 'o', color=c, ms=9, zorder=5)


def photon_skip(ax, x, top=STEM_TOP, lw=1.5):
    """Unselected photon: dashed stem + open circle."""
    c = COL['none']
    ax.plot([x, x], [0, top], color=c, lw=lw, ls='--',
            solid_capstyle='round', zorder=4)
    ax.plot(x, top, 'o', color=c, ms=9, zorder=5,
            mfc='white', mew=1.5)


def label(ax, x, text, case, top=STEM_TOP):
    c = COL[case]
    ax.text(x, top + LABEL_OFF, text, ha='center', va='bottom',
            fontsize=10, color=c, fontweight='bold',
            multialignment='center')


def bracket(ax, x1, x2, y, text, color='#444444', fs=8.5):
    """Double-headed arrow + label below the timeline."""
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='<->',
                                color=color, lw=1.3, mutation_scale=9))
    ax.text((x1 + x2) / 2, y - 0.055, text, ha='center', va='top',
            fontsize=fs, color=color, multialignment='center')


def axis_break(ax, xb):
    """Schematic break marks on the timeline."""
    for x in [xb - 0.15, xb + 0.15]:
        ax.plot([x - 0.15, x, x + 0.15], [-0.05, 0.05, -0.05],
                color='black', lw=1.2, clip_on=False)


def case_label(ax, text):
    ax.text(0.01, 0.97, text, transform=ax.transAxes,
            fontsize=10, va='top', style='italic', color='#222222')


def setup(ax, xlim=(-0.5, 10.5), ylim=(-0.72, 0.95)):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axhline(0, color='black', lw=1.8, zorder=3)
    ax.set_yticks([])
    ax.set_xticks([])
    for sp in ['top', 'right', 'left', 'bottom']:
        ax.spines[sp].set_visible(False)


# ── Build figure ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), constrained_layout=True)
# fig.suptitle('Photon selection criteria for xifusim simulation',
#              fontsize=13, fontweight='bold')

# ── Panel 0 · Case 1: close neighbors (≤ 200 samples) ─────────────────────
# Shows: close pair at start, close triplet in the middle, close pair at end.
# Illustrates that the same rule applies regardless of position in sequence.
ax = axes[0]
setup(ax)

# close pair at start of sequence
x0, x1 = 0.3, 1.1
photon(ax, x0, 'prox');  label(ax, x0, '$P_0$', 'prox')
photon(ax, x1, 'prox');  label(ax, x1, '$P_1$', 'prox')
bracket(ax, x0, x1, -0.14, '≤ 200', COL['prox'])
ax.text((x0 + x1) / 2, -0.50, 'start of\nsequence',
        ha='center', va='top', fontsize=7.5, color='#666666', style='italic')

axis_break(ax, 2.6)
axis_break(ax, 3.2)

# close triplet in the middle
x2, x3, x4 = 4.1, 4.9, 5.7
photon(ax, x2, 'prox');  label(ax, x2, '$P_2$', 'prox')
photon(ax, x3, 'prox');  label(ax, x3, '$P_3$', 'prox')
photon(ax, x4, 'prox');  label(ax, x4, '$P_4$', 'prox')
bracket(ax, x2, x3, -0.14, '≤ 200', COL['prox'])
bracket(ax, x3, x4, -0.14, '≤ 200', COL['prox'])
ax.text((x2 + x4) / 2, -0.50, 'middle of\nsequence',
        ha='center', va='top', fontsize=7.5, color='#666666', style='italic')

axis_break(ax, 7.0)
axis_break(ax, 7.6)

# close pair at end of sequence
x5, x6 = 8.7, 9.5
photon(ax, x5, 'prox');  label(ax, x5, '$P_5$', 'prox')
photon(ax, x6, 'prox');  label(ax, x6, '$P_6$', 'prox')
bracket(ax, x5, x6, -0.14, '≤ 200', COL['prox'])
ax.text((x5 + x6) / 2, -0.50, 'end of\nsequence',
        ha='center', va='top', fontsize=7.5, color='#666666', style='italic')

case_label(ax, 'Case 1 — Any photon with a neighbor within close_dist = 200 samples  '
               '(first, middle, or last in the sequence)')

# ── Panel 1 · Case 2: follows a selected predecessor (≤ HR_samples) ────────
ax = axes[1]
setup(ax)

xD = [0.3, 1.1, 5.2, 9.8]

# seed close pair (selected by case 1)
photon(ax, xD[0], 'prox');  label(ax, xD[0], '$P_0$', 'prox')
photon(ax, xD[1], 'prox');  label(ax, xD[1], '$P_1$', 'prox')
bracket(ax, xD[0], xD[1], -0.14, '≤ 200', COL['prox'])

# P2: selected by case 2 because P1 was selected
photon(ax, xD[2], 'D');
label(ax, xD[2], '$P_2$\n[case 2]', 'D')
bracket(ax, xD[1], xD[2], -0.14,
        '4100 ≤ 8192 samples  ($P_1$ selected → case 2 fires)', COL['D'])

# P3: not selected (gap too large)
photon_skip(ax, xD[3]);  label(ax, xD[3], '$P_3$', 'none')
bracket(ax, xD[2], xD[3], -0.43,
        '9300 > 8192 samples  (case 2 does not fire)', COL['none'])

case_label(ax, 'Case 2 — Photon following an already-selected predecessor  '
               '(gap ≤ HR_samples = 8192 samples)')

# ── Panel 2 · Case 3: precedes a close pair (≤ secondary_samples) ──────────
ax = axes[2]
setup(ax)

xE = [0.5, 3.3, 5.5, 6.3]

# P0: isolated, not selected
photon_skip(ax, xE[0]);  label(ax, xE[0], '$P_0$', 'none')

# P1: selected by case 3
photon(ax, xE[1], 'E');
label(ax, xE[1], '$P_1$\n[case 3]', 'E')

# P2, P3: close pair → selected by case 1
photon(ax, xE[2], 'prox');  label(ax, xE[2], '$P_2$', 'prox')
photon(ax, xE[3], 'prox');  label(ax, xE[3], '$P_3$', 'prox')

# P2–P3 close (case 1, top bracket)
bracket(ax, xE[2], xE[3], -0.14, '≤ 200', COL['prox'])
# P1 to P2 (case 3 condition 1, middle bracket)
bracket(ax, xE[1], xE[2], -0.31,
        '800 ≤ 1563  (secondary_samples)', COL['E'])
# P1 to P3 (case 3 condition 2, bottom bracket)
bracket(ax, xE[1], xE[3], -0.54,
        '950 ≤ 1763  (secondary_samples + close_dist)', COL['E'])

case_label(ax, 'Case 3 — Photon preceding a moderately-close pair  '
               '(gap to $P_{i+1}$ ≤ secondary_samples = 1563  and  gap to $P_{i+2}$ ≤ 1763 samples)')

axes[2].set_xlabel('Time  (schematic, not to scale)', fontsize=10, labelpad=6)

# ── Legend ──────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=COL['prox'],
                   label='Case 1 — close neighbor ≤ 200 samples'),
    mpatches.Patch(color=COL['D'],
                   label='Case 2 — follows selected photon, gap ≤ 8192 samples'),
    mpatches.Patch(color=COL['E'],
                   label='Case 3 — precedes a close pair, gap ≤ 1563 samples'),
    mpatches.Patch(color=COL['none'],
                   label='Not selected'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=2,
           fontsize=9.5, bbox_to_anchor=(0.5, -0.07), frameon=True,
           edgecolor='#cccccc')

plt.savefig('Figures/figure_selection_criteria.png', dpi=150, bbox_inches='tight')
plt.savefig('Figures/figure_selection_criteria.pdf', bbox_inches='tight')
print("Saved to Figures/figure_selection_criteria.png / .pdf")
plt.show()
