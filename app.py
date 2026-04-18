import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

from analysis.ingestion import read_csv_robust, infer_types
from analysis.metrics import rmse, mae, pearson_r

# ──────────────────────────────────────────────
# Configuración general
# ──────────────────────────────────────────────
st.set_page_config(page_title="Comparador de CSVs", layout="wide")
st.title("📊 Comparador de CSVs")

BW_DEFAULT  = 1.0          # multiplicador de ancho de banda (Scott)
KDE_SAMPLES = 100_000      # submuestreo máximo para KDE
GRID        = np.linspace(1, 255, 255)
COLOR_CYCLE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
               "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas índice y reemplaza 0→NaN en columnas numéricas."""
    idx_cols = [c for c in df.columns
                if c.lower().startswith("unnamed") or c.lower() == "index"]
    df = df.drop(columns=idx_cols, errors="ignore")
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df[num] = df[num].replace(0, np.nan)
    return df


def compute_kde_counts(x: np.ndarray, bw_mult: float = BW_DEFAULT):
    """Devuelve (grid, counts) del KDE en conteos reales."""
    if x.size > KDE_SAMPLES:
        x = np.random.default_rng(42).choice(x, size=KDE_SAMPLES, replace=False)
    kde = gaussian_kde(x)
    kde.set_bandwidth(bw_method=kde.factor * bw_mult)
    density = kde(GRID)
    dx = GRID[1] - GRID[0]
    counts = density * len(x) * dx
    return GRID, counts


def ax_style(ax, title="", xlabel="Nivel de gris (0–255)", ylabel="Conteo estimado"):
    ax.set_xlim(0, 255)
    ax.set_xticks(np.arange(0, 256, 50))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25)
    # Escala Y uniforme
    y_top = ax.get_ylim()[1]
    y_step = max(1, int(np.ceil(y_top / 10)))
    ax.set_yticks(np.arange(0, y_top + y_step, y_step))


def mark_max(ax, g, counts, color, label=None):
    idx = np.argmax(counts)
    ax.scatter(g[idx], counts[idx], color=color, s=100, marker="^",
               zorder=5, edgecolor="black", linewidth=0.8)
    if label:
        ax.annotate(f"({g[idx]:.0f}, {counts[idx]:.1f})",
                    xy=(g[idx], counts[idx]),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                    arrowprops=dict(arrowstyle="-", color=color, alpha=0.7))


# ──────────────────────────────────────────────
# Carga de archivos
# ──────────────────────────────────────────────
st.markdown("Sube **uno o varios .csv o .xlsx** para comenzar.")
files = st.file_uploader("Archivos (.csv / .xlsx)", type=["csv","xlsx"],
                         accept_multiple_files=True)

if not files:
    st.info("Carga al menos un archivo para comenzar.")
    st.stop()

# Leer y limpiar todos los archivos
data: dict[str, pd.DataFrame] = {}
for f in files:
    df = read_csv_robust(f)
    df = infer_types(df)
    df = clean_df(df)
    data[f.name] = df

# ──────────────────────────────────────────────
# Vista previa
# ──────────────────────────────────────────────
with st.expander("👀 Vista previa de los archivos", expanded=False):
    tabs_prev = st.tabs(list(data.keys()))
    for tab, name in zip(tabs_prev, data.keys()):
        with tab:
            df0 = data[name]
            st.caption(f"{df0.shape[0]} filas × {df0.shape[1]} columnas")
            st.dataframe(df0.head(20), use_container_width=True)


# ══════════════════════════════════════════════
# SECCIÓN 1: Análisis por archivo
# ══════════════════════════════════════════════
st.markdown("---")
st.header("🔬 Análisis por archivo")

tabs_files = st.tabs(list(data.keys()))

for tab, fname in zip(tabs_files, data.keys()):
    with tab:
        df = data[fname]
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        if not num_cols:
            st.warning("No se encontraron columnas numéricas.")
            continue

        st.caption(f"{len(num_cols)} columnas numéricas: {', '.join(num_cols)}")

        # ── 1a) FDO de todas las columnas ──────────────────────────────
        st.markdown("### FDO — todas las columnas")
        fig1, ax1 = plt.subplots(figsize=(12, 5))

        for i, col in enumerate(num_cols):
            x = df[col].dropna().to_numpy(dtype=float)
            if x.size < 2:
                continue
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            g, counts = compute_kde_counts(x)
            ax1.plot(g, counts, color=color, linewidth=1.8, label=f"{col} (N={len(x)})")
            mark_max(ax1, g, counts, color, label=(len(num_cols) <= 10))

        ax_style(ax1, title=f"FDO por columna — {fname}")
        ax1.legend(loc="upper left", fontsize=7, ncol=max(1, len(num_cols)//12))
        st.pyplot(fig1, clear_figure=True)

        # ── 1b) FDO del promedio por fila ──────────────────────────────
        st.markdown("### FDO — promedio por fila")
        row_means = df[num_cols].mean(axis=1).dropna()

        if row_means.size >= 2:
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            g, counts = compute_kde_counts(row_means.to_numpy(dtype=float))
            ax2.plot(g, counts, color="#1f77b4", linewidth=2.2,
                     label=f"Promedio por fila (N={row_means.size})")
            mark_max(ax2, g, counts, "#1f77b4", label=True)
            ax_style(ax2, title=f"FDO promedio por fila — {fname}",
                     xlabel="Nivel de gris promedio (0–255)")
            ax2.legend(loc="upper left", fontsize=8)
            st.pyplot(fig2, clear_figure=True)
        else:
            st.info("No hay suficientes datos para calcular el promedio por fila.")

        # ── 1c) Boxplots ───────────────────────────────────────────────
        st.markdown("### Boxplots por columna")
        fig3, ax3 = plt.subplots(figsize=(max(8, len(num_cols)*0.9), 4))
        long_df = df[num_cols].melt(var_name="Columna", value_name="Valor")
        sns.boxplot(data=long_df, x="Columna", y="Valor", ax=ax3,
                    color="#FFD54F", showmeans=True,
                    meanprops={"marker":"^","markerfacecolor":"green",
                               "markeredgecolor":"green","markersize":6})
        ax3.set_ylim(0, 255)
        ax3.set_yticks(np.arange(0, 256, 25))
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
        ax3.set_ylabel("Nivel de gris")
        ax3.grid(axis="y", alpha=0.3)
        st.pyplot(fig3, clear_figure=True)

        # ── 1d) Correlación ────────────────────────────────────────────
        if len(num_cols) >= 2:
            st.markdown("### Matriz de correlación (Pearson)")
            corr = df[num_cols].corr()
            fig4, ax4 = plt.subplots(figsize=(min(12, len(num_cols)*0.8+2),
                                              min(10, len(num_cols)*0.7+2)))
            sns.heatmap(corr, vmin=-1, vmax=1, cmap="viridis",
                        annot=(len(num_cols) <= 20), fmt=".2f", ax=ax4)
            ax4.set_title("Correlación entre columnas")
            st.pyplot(fig4, clear_figure=True)


# ══════════════════════════════════════════════
# SECCIÓN 2: Comparación entre archivos
# ══════════════════════════════════════════════
if len(data) < 2:
    st.stop()

st.markdown("---")
st.header("📈 Comparación entre archivos")

# Columnas comunes
common_cols = None
for df in data.values():
    num = set(c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
    common_cols = num if common_cols is None else common_cols & num
common_cols = sorted(common_cols or [])

# ── 2a) FDO superpuesta de promedios por fila ──────────────────────
st.markdown("### FDO superpuesta — promedios por fila")
fig_a, ax_a = plt.subplots(figsize=(12, 5))

for i, (fname, df) in enumerate(data.items()):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    row_means = df[num_cols].mean(axis=1).dropna()
    if row_means.size < 2:
        continue
    color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
    g, counts = compute_kde_counts(row_means.to_numpy(dtype=float))
    ax_a.plot(g, counts, color=color, linewidth=2.5, alpha=0.85,
              label=f"{fname} (N={row_means.size})")
    mark_max(ax_a, g, counts, color, label=True)

ax_style(ax_a, title="FDO superpuesta: promedios por fila",
         xlabel="Nivel de gris promedio (0–255)")
ax_a.legend(loc="upper left", fontsize=8)
st.pyplot(fig_a, clear_figure=True)

# ── 2b) Comparación de promedios por columna ──────────────────────
if common_cols:
    st.markdown("### Promedios por columna — todos los archivos")
    fig_b, ax_b = plt.subplots(figsize=(max(8, len(common_cols)*0.7), 5))

    for i, (fname, df) in enumerate(data.items()):
        means = df[common_cols].mean(numeric_only=True)
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        ax_b.plot(common_cols, means.values, marker="o", color=color,
                  linewidth=2, label=fname)

    ax_b.set_ylabel("Promedio (nivel de gris)")
    ax_b.set_xlabel("Columna")
    ax_b.set_xticks(range(len(common_cols)))
    ax_b.set_xticklabels(common_cols, rotation=45, ha="right")
    ax_b.grid(alpha=0.3)
    ax_b.legend(loc="upper right", fontsize=8)
    ax_b.set_title("Comparación de promedios por columna (archivos superpuestos)")
    st.pyplot(fig_b, clear_figure=True)
else:
    st.info("No hay columnas numéricas en común entre todos los archivos.")


"""
FASE 3 — Validación de Estandarización PRNU  (versión mejorada)
================================================================
REEMPLAZA todo el bloque de Fase 3 en tu app.py existente.
Métricas: CoV, d de Cohen, prueba t de Welch, JSD, Δ media
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import gaussian_kde, ttest_ind
from scipy.spatial.distance import jensenshannon

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
GRID         = np.linspace(1, 255, 255)
COLOR_HETERO = "#E24B4A"   # rojo  → heterogéneo
COLOR_PRNU   = "#1D9E75"   # verde → PRNU


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de carga y limpieza
# ─────────────────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    idx = [c for c in df.columns if c.lower().startswith("unnamed") or c.lower() == "index"]
    df  = df.drop(columns=idx, errors="ignore")
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df[num] = df[num].replace(0, np.nan)
    return df


def _load_files(files) -> dict[str, pd.DataFrame]:
    out = {}
    for f in files:
        try:
            df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
        except Exception:
            continue
        out[f.name] = _clean(df)
    return out


def _row_means(dfs: dict) -> np.ndarray:
    """Promedio por fila de cada archivo, concatenados."""
    arrays = []
    for df in dfs.values():
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        rm  = df[num].mean(axis=1).dropna().to_numpy(dtype=float)
        arrays.append(rm)
    return np.concatenate(arrays) if arrays else np.array([])


def _col_means(dfs: dict) -> np.ndarray:
    """Promedio por columna (ojo) de todos los archivos."""
    means = []
    for df in dfs.values():
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        means.extend(df[num].mean(skipna=True).tolist())
    return np.array([m for m in means if not np.isnan(m)])


def _kde_counts(x: np.ndarray) -> np.ndarray:
    if x.size < 2:
        return np.zeros_like(GRID)
    if x.size > 80_000:
        x = np.random.default_rng(42).choice(x, 80_000, replace=False)
    kde = gaussian_kde(x)
    dx  = GRID[1] - GRID[0]
    return kde(GRID) * len(x) * dx


# ─────────────────────────────────────────────────────────────────────────────
# Métricas estadísticas
# ─────────────────────────────────────────────────────────────────────────────

def coef_variacion(x: np.ndarray) -> float:
    m = np.nanmean(x)
    return (np.nanstd(x) / m * 100) if m != 0 else np.nan


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """d de Cohen para dos grupos independientes (pooled SD)."""
    nx, ny   = len(x), len(y)
    pooled_s = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 +
                        (ny - 1) * np.std(y, ddof=1)**2) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_s if pooled_s > 0 else np.nan


def welch_t(x: np.ndarray, y: np.ndarray):
    """Prueba t de Welch. Devuelve (t, p)."""
    t, p = ttest_ind(x, y, equal_var=False)
    return float(t), float(p)


def jsd_entre_grupos(x_a: np.ndarray, x_b: np.ndarray) -> float:
    ka = _kde_counts(x_a) + 1e-10
    kb = _kde_counts(x_b) + 1e-10
    ka /= ka.sum(); kb /= kb.sum()
    return float(jensenshannon(ka, kb))


def interpretar_cohen(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:   return "Efecto negligible"
    elif ad < 0.5: return "Efecto pequeño"
    elif ad < 0.8: return "Efecto moderado"
    else:          return "Efecto grande ✓"


def interpretar_p(p: float) -> str:
    if p < 0.001:  return "p < 0.001 — diferencia muy significativa ✓"
    elif p < 0.01: return "p < 0.01 — diferencia significativa ✓"
    elif p < 0.05: return "p < 0.05 — diferencia significativa ✓"
    else:          return "p ≥ 0.05 — no significativo"


def interpretar_jsd(jsd: float) -> str:
    if jsd < 0.05:  return "Distribuciones casi idénticas"
    elif jsd < 0.15: return "Diferencia leve"
    elif jsd < 0.30: return "Diferencia moderada ✓"
    else:            return "Distribuciones muy distintas ✓"


# ─────────────────────────────────────────────────────────────────────────────
# Gráficas
# ─────────────────────────────────────────────────────────────────────────────

def fig_fdo_superpuesta(rm_h: np.ndarray, rm_p: np.ndarray) -> plt.Figure:
    """FDO de promedios por fila — más sensible a desplazamientos entre grupos."""
    fig, ax = plt.subplots(figsize=(11, 4.5))

    for vals, color, label in [
        (rm_h, COLOR_HETERO, f"Heterogéneo  (N={len(rm_h)},  μ={np.mean(rm_h):.1f})"),
        (rm_p, COLOR_PRNU,   f"PRNU          (N={len(rm_p)},  μ={np.mean(rm_p):.1f})"),
    ]:
        if vals.size < 2:
            continue
        counts = _kde_counts(vals)
        ax.plot(GRID, counts, color=color, linewidth=2.4, label=label)
        ax.fill_between(GRID, counts, alpha=0.10, color=color)
        idx = np.argmax(counts)
        ax.scatter(GRID[idx], counts[idx], color=color, s=100, marker="^",
                   zorder=5, edgecolor="black", linewidth=0.8)
        ax.annotate(f"pico={GRID[idx]:.0f}",
                    xy=(GRID[idx], counts[idx]), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8))

    # Línea de diferencia de medias
    ax.axvline(np.mean(rm_h), color=COLOR_HETERO, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(np.mean(rm_p), color=COLOR_PRNU,   linestyle="--", linewidth=1.2, alpha=0.7)

    delta = np.mean(rm_h) - np.mean(rm_p)
    ax.set_xlim(0, 255); ax.set_xticks(np.arange(0, 256, 25))
    ax.set_xlabel("Nivel de gris promedio por fila (0–255)", fontsize=10)
    ax.set_ylabel("Conteo estimado (KDE)", fontsize=10)
    ax.set_title(f"FDO — Promedios por fila: Heterogéneo vs. PRNU\n"
                 f"Desplazamiento de medias: Δμ = {delta:+.1f} niveles de gris", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    return fig


def fig_boxplot_comparativo(rm_h: np.ndarray, rm_p: np.ndarray,
                             d: float, t: float, p: float) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    # ── Izquierda: boxplot + strip ─────────────────────────────────────────
    ax = axes[0]
    df_plot = pd.DataFrame({
        "Grupo":  (["Heterogéneo"] * len(rm_h) + ["PRNU"] * len(rm_p)),
        "Valor":  np.concatenate([rm_h, rm_p]),
    })
    palette = {"Heterogéneo": COLOR_HETERO, "PRNU": COLOR_PRNU}

    sns.boxplot(data=df_plot, x="Grupo", y="Valor", ax=ax,
                palette=palette, showmeans=True, width=0.45,
                meanprops={"marker": "^", "markerfacecolor": "black",
                           "markeredgecolor": "black", "markersize": 8},
                flierprops={"marker": "o", "markersize": 2, "alpha": 0.3})

    # Puntos semitransparentes encima
    rng = np.random.default_rng(0)
    for i, (vals, grupo) in enumerate([(rm_h, "Heterogéneo"), (rm_p, "PRNU")]):
        sample = vals if len(vals) <= 300 else rng.choice(vals, 300, replace=False)
        jitter = rng.uniform(-0.15, 0.15, size=len(sample))
        ax.scatter(np.full(len(sample), i) + jitter, sample,
                   alpha=0.25, s=8, color=palette[grupo], zorder=2)

    # CoV encima de cada caja
    for i, vals in enumerate([rm_h, rm_p]):
        cov = coef_variacion(vals)
        ax.text(i, ax.get_ylim()[1] * 0.98, f"CoV = {cov:.1f}%",
                ha="center", va="top", fontsize=9, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=COLOR_HETERO if i == 0 else COLOR_PRNU,
                          alpha=0.9))

    ax.set_ylabel("Nivel de gris promedio por fila", fontsize=10)
    ax.set_title("Distribución de promedios por fila\n(puntos = muestra aleatoria de 300)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # ── Derecha: panel de métricas ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")

    metricas = [
        ("d de Cohen", f"{d:.3f}", interpretar_cohen(d),
         "#2E75B6" if abs(d) >= 0.5 else "#888"),
        ("t de Welch", f"{t:.2f}", interpretar_p(p),
         "#2E75B6" if p < 0.05 else "#888"),
        ("Δ media (H−P)", f"{np.mean(rm_h)-np.mean(rm_p):+.1f} GL",
         "Sesgo sistemático" if abs(np.mean(rm_h)-np.mean(rm_p)) > 5 else "Sin sesgo relevante",
         "#E24B4A" if abs(np.mean(rm_h)-np.mean(rm_p)) > 5 else "#888"),
    ]

    y = 0.92
    ax2.text(0.5, 1.0, "Estadísticos clave", ha="center", va="top",
             fontsize=11, fontweight="bold", transform=ax2.transAxes)
    for nombre, valor, interp, color in metricas:
        ax2.text(0.05, y, nombre, ha="left", va="top", fontsize=9,
                 color="#555", transform=ax2.transAxes)
        ax2.text(0.05, y - 0.08, valor, ha="left", va="top", fontsize=14,
                 fontweight="bold", color=color, transform=ax2.transAxes)
        ax2.text(0.05, y - 0.17, interp, ha="left", va="top", fontsize=8,
                 color="#777", style="italic", transform=ax2.transAxes)
        ax2.plot([0.02, 0.98], [y - 0.22, y - 0.22],
             color="#ddd", linewidth=0.8, transform=ax2.transAxes)
        y -= 0.30

    fig.tight_layout()
    return fig


def fig_bland_altman(rm_h: np.ndarray, rm_p: np.ndarray) -> plt.Figure:
    n = min(len(rm_h), len(rm_p))
    if n < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Datos insuficientes", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    rng  = np.random.default_rng(42)
    idx_h = rng.choice(len(rm_h), n, replace=False)
    idx_p = rng.choice(len(rm_p), n, replace=False)
    a, b  = rm_h[idx_h], rm_p[idx_p]

    avg    = (a + b) / 2
    diff   = a - b
    mean_d = np.mean(diff)
    std_d  = np.std(diff)
    loa_hi = mean_d + 1.96 * std_d
    loa_lo = mean_d - 1.96 * std_d
    pct_dentro = np.mean((diff >= loa_lo) & (diff <= loa_hi)) * 100

    fig, ax = plt.subplots(figsize=(10, 5))

    # Color por zona: dentro/fuera de límites
    dentro = (diff >= loa_lo) & (diff <= loa_hi)
    ax.scatter(avg[dentro],  diff[dentro],  color="#378ADD", alpha=0.45, s=18,
               edgecolors="none", label="Dentro de límites")
    ax.scatter(avg[~dentro], diff[~dentro], color=COLOR_HETERO, alpha=0.70, s=22,
               edgecolors="none", marker="x", label="Fuera de límites")

    ax.axhline(mean_d, color="black",      linewidth=1.6, linestyle="-",
               label=f"Sesgo = {mean_d:.2f} GL")
    ax.axhline(loa_hi, color=COLOR_HETERO, linewidth=1.3, linestyle="--",
               label=f"+1.96 SD = {loa_hi:.2f}")
    ax.axhline(loa_lo, color=COLOR_HETERO, linewidth=1.3, linestyle="--",
               label=f"−1.96 SD = {loa_lo:.2f}")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.fill_between([avg.min(), avg.max()], loa_lo, loa_hi,
                    color=COLOR_HETERO, alpha=0.06)

    # Anotación del sesgo
    ax.text(avg.max() * 0.98, mean_d + std_d * 0.15,
            f"Sesgo = {mean_d:.1f}", ha="right", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))
    ax.text(0.02, 0.03,
            f"{pct_dentro:.1f}% de puntos dentro de los límites de acuerdo (esperado: 95%)",
            transform=ax.transAxes, fontsize=8.5, color="#444",
            bbox=dict(facecolor="#f9f9f9", edgecolor="#ccc", boxstyle="round,pad=0.4"))

    ax.set_xlabel("Promedio de ambas mediciones (nivel de gris)", fontsize=10)
    ax.set_ylabel("Diferencia  Heterogéneo − PRNU  (GL)", fontsize=10)
    ax.set_title("Gráfico de Bland-Altman\n"
                 "Un sesgo alejado de cero confirma que los equipos no son comparables sin PRNU",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_tabla_metricas(cov_h, cov_p, d, t, p, jsd, delta_mu) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")

    p_str = "< 0.001" if p < 0.001 else f"{p:.4f}"

    rows = [
        ["CoV  (dispersión relativa)",
         f"{cov_h:.2f} %", f"{cov_p:.2f} %",
         "↓ Mejor" if cov_p < cov_h else "↑ Peor"],
        ["d de Cohen  (tamaño de efecto)",
         f"{abs(d):.3f}", "—",
         interpretar_cohen(d)],
        ["t de Welch  (significancia)",
         f"t = {t:.2f}", f"p = {p_str}",
         interpretar_p(p)],
        ["Δ media  (sesgo sistemático)",
         f"{delta_mu:+.1f} GL", "—",
         "Sesgo detectado ✓" if abs(delta_mu) > 5 else "Sin sesgo relevante"],
        ["JSD  (similitud distribuciones)",
         "—", f"{jsd:.4f}",
         interpretar_jsd(jsd)],
    ]
    cols = ["Métrica", "Heterogéneo", "PRNU / Resultado", "Interpretación"]

    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    tbl.scale(1, 1.9)

    # Encabezado
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#1A3A5C")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Colorear filas alternadas y celdas de resultado
    row_colors = ["#F0F4F8", "#FFFFFF"] * 10
    for i in range(1, len(rows) + 1):
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(row_colors[i])
        # Columna heterogéneo = rojo suave
        tbl[i, 1].set_facecolor("#FDDCDC")
        # Columna PRNU/resultado = verde suave
        tbl[i, 2].set_facecolor("#DDFBEE")

    ax.set_title("Resumen de métricas — Fase 3  (Validación PRNU)",
                 fontweight="bold", fontsize=11, pad=14)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.header("🧪 Fase 3 — Validación de Estandarización PRNU")
st.markdown(
    "Sube **dos conjuntos de archivos** del mismo grado de retinopatía: "
    "uno con datos mezclados (heterogéneo) y otro ya estandarizado por PRNU. "
    "La app compara estadísticamente ambos escenarios."
)

col_h, col_p = st.columns(2)
with col_h:
    st.markdown("#### 🔴 Datos heterogéneos (revueltos)")
    files_h = st.file_uploader("Archivos sin estandarizar", type=["csv", "xlsx"],
                                accept_multiple_files=True, key="fase3_h")
with col_p:
    st.markdown("#### 🟢 Datos estandarizados por PRNU")
    files_p = st.file_uploader("Archivos por clúster PRNU", type=["csv", "xlsx"],
                                accept_multiple_files=True, key="fase3_p")

if files_h and files_p:
    with st.spinner("Calculando métricas y generando gráficas..."):
        dfs_h = _load_files(files_h)
        dfs_p = _load_files(files_p)

        rm_h = _row_means(dfs_h)   # promedios por fila — más sensibles
        rm_p = _row_means(dfs_p)

        if rm_h.size < 2 or rm_p.size < 2:
            st.error("Alguno de los grupos no tiene suficientes datos numéricos.")
            st.stop()

        cov_h    = coef_variacion(rm_h)
        cov_p    = coef_variacion(rm_p)
        d        = cohen_d(rm_h, rm_p)
        t, p     = welch_t(rm_h, rm_p)
        jsd      = jsd_entre_grupos(rm_h, rm_p)
        delta_mu = float(np.mean(rm_h) - np.mean(rm_p))

    # ── Tabla resumen ──────────────────────────────────────────────────────
    st.subheader("📊 Métricas comparativas")
    st.pyplot(fig_tabla_metricas(cov_h, cov_p, d, t, p, jsd, delta_mu),
              clear_figure=True)

    with st.expander("ℹ️ ¿Qué significa cada métrica?"):
        st.markdown("""
        | Métrica | Qué mide | Cuándo indica diferencia |
        |---|---|---|
        | **CoV** | Dispersión relativa dentro de cada grupo | CoV heterogéneo > CoV PRNU |
        | **d de Cohen** | Tamaño del efecto entre grupos (independiente del N) | \|d\| ≥ 0.5 = moderado, ≥ 0.8 = grande |
        | **t de Welch** | Si las medias son estadísticamente distintas | p < 0.05 = diferencia significativa |
        | **Δ media** | Desplazamiento absoluto en niveles de gris | \|Δ\| > 5 GL = sesgo detectable |
        | **JSD** | Diferencia entre distribuciones completas (forma) | JSD > 0.15 = distribuciones distinguibles |
        """)

    st.markdown("---")

    # ── FDO superpuesta ────────────────────────────────────────────────────
    st.subheader("📈 FDO — Promedios por fila: Heterogéneo vs. PRNU")
    st.pyplot(fig_fdo_superpuesta(rm_h, rm_p), clear_figure=True)
    st.caption(
        "Se grafica la distribución del **promedio por fila** (un valor por paciente/imagen). "
        "Las líneas punteadas marcan la media de cada grupo. "
        "Si los equipos introducen sesgo, las curvas y sus picos estarán desplazados."
    )

    # ── Boxplot + métricas ─────────────────────────────────────────────────
    st.subheader("📦 Boxplot comparativo + estadísticos")
    st.pyplot(fig_boxplot_comparativo(rm_h, rm_p, d, t, p), clear_figure=True)
    st.caption(
        "Izquierda: distribución real (caja + puntos individuales). "
        "El CoV sobre cada caja mide la dispersión relativa. "
        "Derecha: los tres estadísticos clave para cuantificar la diferencia entre grupos."
    )

    # ── Bland-Altman ───────────────────────────────────────────────────────
    st.subheader("📐 Gráfico de Bland-Altman")
    st.pyplot(fig_bland_altman(rm_h, rm_p), clear_figure=True)
    st.caption(
        "Compara pares de observaciones (muestreo aleatorio si N difiere). "
        "Un sesgo medio alejado de cero confirma que los grupos no son directamente "
        "comparables sin estandarización PRNU. Se espera que ~95% de los puntos "
        "estén dentro de los límites de acuerdo."
    )

    # ── Exportar ───────────────────────────────────────────────────────────
    st.subheader("💾 Exportar métricas")
    p_str = "< 0.001" if p < 0.001 else f"{p:.6f}"
    df_export = pd.DataFrame({
        "Métrica": ["CoV Heterogéneo (%)", "CoV PRNU (%)",
                    "d de Cohen", "t de Welch", "p-valor (Welch)",
                    "Δ media (GL)", "JSD"],
        "Valor":   [round(cov_h, 4), round(cov_p, 4),
                    round(d, 4), round(t, 4), p_str,
                    round(delta_mu, 2), round(jsd, 4)],
        "Interpretación": [
            "Dispersión relativa heterogéneo",
            "Dispersión relativa PRNU",
            interpretar_cohen(d),
            "Estadístico t (Welch, grupos independientes)",
            interpretar_p(p),
            "Sesgo sistemático en niveles de gris",
            interpretar_jsd(jsd),
        ]
    })
    st.download_button(
        label="⬇️ Descargar métricas como CSV",
        data=df_export.to_csv(index=False).encode("utf-8"),
        file_name="fase3_metricas.csv",
        mime="text/csv"
    )

elif files_h or files_p:
    st.info("Carga archivos en **ambos** grupos para activar el análisis comparativo.")
else:
    st.info("Carga los dos grupos de archivos para comenzar.")