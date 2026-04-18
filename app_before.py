import streamlit as st
import pandas as pd
import numpy as np
from analysis.ingestion import read_csv_robust, infer_types
from analysis.viz import fig_scatter_identity
from analysis.metrics import rmse, mae, pearson_r
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde

st.set_page_config(page_title="Comparador de CSVs", layout="wide")
st.title("📊 Comparador de CSVs — MVP")

st.markdown("Sube **uno o varios .csv o .xlsx** para comenzar.")

files = st.file_uploader("Archivos (.csv / .xlsx)", type=["csv", "xlsx"], accept_multiple_files=True)

if files:
    dfs = {}
    for f in files:
        df = read_csv_robust(f)
        df = infer_types(df)
        dfs[f.name] = df

    st.subheader("👀 Vista previa")
    tabs = st.tabs(list(dfs.keys()))
    for tab, name in zip(tabs, dfs.keys()):
        with tab:
            st.caption(f"{name} — {dfs[name].shape[0]} filas × {dfs[name].shape[1]} columnas")
            st.dataframe(dfs[name].head(20), use_container_width=True)

else:
    st.info("Carga al menos un CSV.")


data = dfs


# =======================
# Análisis por archivo: FDO/KDE, Promedios, Boxplots, Histogramas, Correlación, ECDF
# =======================


st.subheader("🔬 Análisis por archivo (FDO/KDE, promedios, caja, hist, correlación, ECDF)")

if data:
    # Un tab por archivo
    tabs_files = st.tabs(list(data.keys()))
    for tab, fname in zip(tabs_files, data.keys()):
        with tab:
            df0 = data[fname].copy()

            # -------------------------
            # 1) Limpieza: eliminar ceros y columnas índice "sucias"
            # -------------------------
            # Heurística: columnas típicas de índice a descartar
            idx_like = [c for c in df0.columns if c.lower().startswith("unnamed") or c.lower() in {"index"}]
            if idx_like:
                df0 = df0.drop(columns=idx_like)

            # Reemplaza 0 por NaN sólo en columnas numéricas (cero representa fuera de vasos)
            num_cols_all = [c for c in df0.columns if pd.api.types.is_numeric_dtype(df0[c])]
            df0[num_cols_all] = df0[num_cols_all].replace(0, np.nan)

            st.caption(f"Columnas numéricas detectadas: {len(num_cols_all)}")
            # Usar todas las columnas numéricas
            sel_cols = num_cols_all

            # Parámetros comunes de visualización
            c1, c2, c3 = st.columns([1,1,1])
            kde_samples = c1.number_input("Submuestreo para KDE (máx. puntos)", min_value=10000, max_value=200000, value=100000, step=10000, key=f"samp_{fname}")
            bw_mult = c2.slider("Ancho de banda (×Scott)", 0.25, 3.0, 1.0, 0.05, key=f"bw_{fname}")
            show_counts = c3.selectbox("Eje Y (KDE)", options=["Conteos (reales)", "Densidad"], index=0, key=f"yaxis_{fname}")

            # -------------------------
            # 2) FDO / KDE (por columnas seleccionadas)
            #    - Calculamos KDE con scipy (gaussian_kde)
            #    - Convertimos densidad -> conteos: counts = density * N * dx
            #    - Marcamos máximos y mínimos
            # -------------------------
            st.markdown("### FDO / KDE (Función de Distribución Oximétrica)")
            fig_kde, ax_kde = plt.subplots(figsize=(12, 6))
            grid = np.linspace(1, 255, 255)  # eje horizontal fijo 0–255 (evitamos 0 por limpieza)

            color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3","C4","C5"])

            max_yvalue = 0
            legend_lines = []
            for i, col in enumerate(sel_cols):
                x = df0[col].dropna().to_numpy(dtype=float)
                if x.size == 0:
                    continue
                # Submuestreo si excede límite
                if x.size > kde_samples:
                    rng = np.random.default_rng(42)
                    x = rng.choice(x, size=kde_samples, replace=False)

                # KDE con método de Scott y multiplicador
                kde = gaussian_kde(x)  # Scott por defecto; luego ajustamos
                kde.set_bandwidth(bw_method=kde.factor * bw_mult)
                density = kde(grid)

                # conversion a conteos
                dx = grid[1] - grid[0]
                N = len(x)
                counts = density * N * dx

                # Plot
                yplot = counts if show_counts == "Conteos (reales)" else density
                label = f"{col} (N={N})"
                color = color_cycle[i % len(color_cycle)]
                ax_kde.plot(grid, yplot, color=color, linewidth=2, label=label)
                
                # Marcar máximo y mínimo
                max_idx = np.argmax(yplot)
                min_idx = np.argmin(yplot)
                ax_kde.scatter(grid[max_idx], yplot[max_idx], color=color, s=100, marker='^', 
                             zorder=5, edgecolor='black', linewidth=1)
                ax_kde.scatter(grid[min_idx], yplot[min_idx], color=color, s=100, marker='v', 
                             zorder=5, edgecolor='black', linewidth=1)

                # Anotación del máximo de cada serie tal como antes, justo encima del triángulo
                lines_count = len(sel_cols)
                show_label = (lines_count <= 12) or (i % 2 == 0)
                if show_label:
                    ax_kde.annotate(f"Máx: ({grid[max_idx]:.0f}, {yplot[max_idx]:.1f})",
                                    xy=(grid[max_idx], yplot[max_idx]),
                                    xytext=(0, 12), textcoords='offset points',
                                    ha='center', va='bottom',
                                    color=color, fontsize=8, alpha=0.9,
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                                    arrowprops=dict(arrowstyle='-', color=color, alpha=0.7))

                max_yvalue = max(max_yvalue, yplot.max())

            ax_kde.set_xlim(0, 255)

            # Escala uniforme cada 50 en x
            ax_kde.set_xticks(np.arange(0, 256, 50))
            
            # Escala uniforme en y con un máximo de 10 ticks para evitar encimado
            if show_counts == "Conteos (reales)":
                ax_kde.set_ylabel("Conteo estimado de píxeles")
                y_max = int(np.ceil(max_yvalue / 10) * 10)
                y_step = max(1, int(np.ceil(y_max / 10)))
                ax_kde.set_yticks(np.arange(0, y_max + y_step, y_step))
            else:
                ax_kde.set_ylabel("Densidad (KDE)")
                y_max = float(ax_kde.get_ylim()[1])
                y_step = max(0.1, (y_max / 10))
                ax_kde.set_yticks(np.arange(0, y_max + y_step, y_step))
            
            ax_kde.set_xlabel("Nivel de gris (0–255)")
            ax_kde.grid(alpha=0.25)
            ax_kde.legend(loc="upper left", fontsize=8)
            st.pyplot(fig_kde, clear_figure=True)


            # -------------------------
            # KDE de promedios por fila
            # -------------------------
            st.markdown("### KDE de promedios por fila")
            row_means = df0[sel_cols].mean(axis=1).dropna()
            if row_means.size > 0:
                fig_kde_row, ax_kde_row = plt.subplots(figsize=(12, 6))
                grid = np.linspace(1, 255, 255)
                
                # Submuestreo si excede límite
                x_row = row_means.to_numpy(dtype=float)
                if x_row.size > kde_samples:
                    rng = np.random.default_rng(42)
                    x_row = rng.choice(x_row, size=kde_samples, replace=False)
                
                # KDE con ancho de banda ajustado
                kde_row = gaussian_kde(x_row)
                kde_row.set_bandwidth(bw_method=kde_row.factor * bw_mult)
                density_row = kde_row(grid)
                
                dx = grid[1] - grid[0]
                N_row = len(x_row)
                counts_row = density_row * N_row * dx
                
                yplot_row = counts_row if show_counts == "Conteos (reales)" else density_row
                ax_kde_row.plot(grid, yplot_row, color='blue', linewidth=2, label=f'Promedios por fila (N={N_row})')
                
                # Marcar máximo
                max_idx_row = np.argmax(yplot_row)
                ax_kde_row.scatter(grid[max_idx_row], yplot_row[max_idx_row], color='blue', s=100, marker='^', 
                                 zorder=5, edgecolor='black', linewidth=1)
                # Anotación solo en un gráfico de fila para evitar densidad de textos
                dy_row = 5
                ax_kde_row.annotate(f"Máx: ({grid[max_idx_row]:.0f}, {yplot_row[max_idx_row]:.1f})", 
                                  xy=(grid[max_idx_row], yplot_row[max_idx_row]), 
                                  xytext=(5, dy_row), textcoords='offset points', 
                                  fontsize=8, alpha=0.7, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
                
                ax_kde_row.set_xlim(0, 255)
                ax_kde_row.set_xticks(np.arange(0, 256, 50))
                if show_counts == "Conteos (reales)":
                    ax_kde_row.set_ylabel("Conteo estimado de píxeles")
                    y_max_row = int(np.ceil(yplot_row.max() / 10) * 10)
                    y_step_row = max(1, int(np.ceil(y_max_row / 10)))
                    ax_kde_row.set_yticks(np.arange(0, y_max_row + y_step_row, y_step_row))
                else:
                    ax_kde_row.set_ylabel("Densidad (KDE)")
                    y_max_row = float(ax_kde_row.get_ylim()[1])
                    y_step_row = max(0.1, y_max_row / 10)
                    ax_kde_row.set_yticks(np.arange(0, y_max_row + y_step_row, y_step_row))
                ax_kde_row.set_xlabel("Nivel de gris promedio (0–255)")
                ax_kde_row.grid(alpha=0.25)
                ax_kde_row.legend(loc="upper left", fontsize=8)
                st.pyplot(fig_kde_row, clear_figure=True)
            else:
                st.info("No hay datos suficientes para calcular promedios por fila.")

            # -------------------------
            # 4) Boxplots (con media como triángulo verde) y escala uniforme
            # -------------------------
            st.markdown("### Boxplots por columna")
            fig_box, ax_box = plt.subplots(figsize=(10, 4))
            # Preparamos datos en formato largo para seaborn
            long_df = df0[sel_cols].melt(var_name="Columna", value_name="Valor")
            # Color amarillo
            sns.boxplot(data=long_df, x="Columna", y="Valor", ax=ax_box, color="#FFD54F", showmeans=True,
                        meanprops={"marker":"^", "markerfacecolor":"green", "markeredgecolor":"green", "markersize":6})
            
            # Marcar máximo y mínimo globales
            max_val = long_df["Valor"].max()
            min_val = long_df["Valor"].min()
            
            ax_box.axhline(max_val, color='green', linestyle='--', linewidth=1, alpha=0.5, label=f'Máx: {max_val:.1f}')
            ax_box.axhline(min_val, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Mín: {min_val:.1f}')
            
            ax_box.set_ylim(0, 255)
            ax_box.set_yticks(np.arange(0, 256, 10))
            ax_box.set_xticklabels(ax_box.get_xticklabels(), rotation=45, ha="right")
            ax_box.set_ylabel("Nivel de gris")
            ax_box.grid(axis="y", alpha=0.3)
            ax_box.legend(loc="upper right", fontsize=8)
            st.pyplot(fig_box, clear_figure=True)

            # -------------------------
            # 6) Matriz de correlación
            # -------------------------
            st.markdown("### Matriz de correlación (Pearson)")
            if len(sel_cols) >= 2:
                corr = df0[sel_cols].corr()
                fig_c, ax_c = plt.subplots(figsize=(6, 5))
                sns.heatmap(corr, vmin=-1, vmax=1, cmap="viridis", annot=True, fmt=".2f", ax=ax_c)
                ax_c.set_title("Correlación entre columnas")
                st.pyplot(fig_c, clear_figure=True)
            else:
                st.info("Selecciona ≥2 columnas para ver la correlación.")

            # -------------------------
            # 7) ECDF (gráfica complementaria)
            # -------------------------
            st.markdown("### ECDF (Función de distribución acumulada empírica)")
            fig_ecdf, ax_ecdf = plt.subplots(figsize=(10, 4))
            for i, col in enumerate(sel_cols):
                xv = df0[col].dropna().astype(float).sort_values().to_numpy()
                if xv.size == 0:
                    continue
                y = np.arange(1, xv.size + 1) / xv.size
                ax_ecdf.step(xv, y, where="post", color=color_cycle[i % len(color_cycle)], label=col, linewidth=1.8)
            ax_ecdf.set_xlim(0, 255)
            ax_ecdf.set_ylim(0, 1)
            ax_ecdf.set_xlabel("Nivel de gris (0–255)")
            ax_ecdf.set_ylabel("Proporción acumulada")
            ax_ecdf.grid(alpha=0.3)
            ax_ecdf.legend(loc="lower right", fontsize=8)
            st.pyplot(fig_ecdf, clear_figure=True)


# =======================
# Comparación de KDEs entre archivos
# =======================

if len(data) >= 2:
    st.subheader("📈 Comparación de KDEs (superpuestos)")
    st.markdown("Visualiza KDEs de la misma columna de múltiples archivos superpuestos.")
    
    # Selector de columnas comunes
    all_num_cols = {}
    for fname, df_file in data.items():
        df_temp = df_file.copy()
        idx_like = [c for c in df_temp.columns if c.lower().startswith("unnamed") or c.lower() in {"index"}]
        if idx_like:
            df_temp = df_temp.drop(columns=idx_like)
        num_cols = [c for c in df_temp.columns if pd.api.types.is_numeric_dtype(df_temp[c])]
        all_num_cols[fname] = num_cols
    
    # Encontrar columnas comunes
    common_cols_all = set(all_num_cols[list(all_num_cols.keys())[0]])
    for fname in list(all_num_cols.keys())[1:]:
        common_cols_all &= set(all_num_cols[fname])
    
    common_cols_list = sorted(list(common_cols_all))
    
    if common_cols_list:
        col_compare = st.selectbox("Columna a comparar", options=common_cols_list, key="col_compare_kde")
        
        c1, c2 = st.columns([1, 1])
        kde_samples_comp = c1.number_input("Submuestreo (máx. puntos)", min_value=10000, max_value=200000, value=100000, step=10000, key="samp_comp")
        bw_mult_comp = c2.slider("Ancho de banda (×Scott)", 0.25, 3.0, 1.0, 0.05, key="bw_comp")
        
        # Graficar KDEs superpuestos
        fig_kde_comp, ax_kde_comp = plt.subplots(figsize=(12, 6))
        grid = np.linspace(1, 255, 255)
        
        color_cycle_comp = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3","C4","C5"])
        max_yvalue_comp = 0
        
        for i, fname in enumerate(data.keys()):
            df_file = data[fname].copy()
            idx_like = [c for c in df_file.columns if c.lower().startswith("unnamed") or c.lower() in {"index"}]
            if idx_like:
                df_file = df_file.drop(columns=idx_like)
            
            # Filtrar ceros
            num_cols_temp = [c for c in df_file.columns if pd.api.types.is_numeric_dtype(df_file[c])]
            df_file[num_cols_temp] = df_file[num_cols_temp].replace(0, np.nan)
            
            x = df_file[col_compare].dropna().to_numpy(dtype=float)
            if x.size == 0:
                continue
            
            # Submuestreo
            if x.size > kde_samples_comp:
                rng = np.random.default_rng(42)
                x = rng.choice(x, size=kde_samples_comp, replace=False)
            
            # KDE
            kde = gaussian_kde(x)
            kde.set_bandwidth(bw_method=kde.factor * bw_mult_comp)
            density = kde(grid)
            
            # Conversión a conteos
            dx = grid[1] - grid[0]
            N = len(x)
            counts = density * N * dx
            
            color = color_cycle_comp[i % len(color_cycle_comp)]
            label = f"{fname} (N={N})"
            ax_kde_comp.plot(grid, counts, color=color, linewidth=2.5, label=label, alpha=0.8)
            
            # Marcar máximo con coordenadas
            max_idx = np.argmax(counts)
            ax_kde_comp.scatter(grid[max_idx], counts[max_idx], color=color, s=120, marker='^', 
                               zorder=5, edgecolor='black', linewidth=1)
            lines_count = len(data)
            show_label = (lines_count <= 12) or (i % 2 == 0)
            if show_label:
                ax_kde_comp.annotate(f"({grid[max_idx]:.0f}, {counts[max_idx]:.1f})", 
                                    xy=(grid[max_idx], counts[max_idx]), 
                                    xytext=(0, 12), textcoords='offset points',
                                    ha='center', va='bottom',
                                    fontsize=8, alpha=0.9, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                                    arrowprops=dict(arrowstyle='-', color=color, alpha=0.8))
            
            max_yvalue_comp = max(max_yvalue_comp, counts.max())
        
        ax_kde_comp.set_xlim(0, 255)
        ax_kde_comp.set_xticks(np.arange(0, 256, 50))
        ax_kde_comp.set_ylabel("Conteo estimado de píxeles")
        ax_kde_comp.set_xlabel("Nivel de gris (0–255)")
        
        # Escala uniforme en y con máximo 10 ticks para evitar encimados
        y_max = int(np.ceil(max_yvalue_comp / 10) * 10)
        y_step = max(1, int(np.ceil(y_max / 10)))
        ax_kde_comp.set_yticks(np.arange(0, y_max + y_step, y_step))
        
        ax_kde_comp.grid(alpha=0.25)
        ax_kde_comp.legend(loc="upper left", fontsize=9)
        ax_kde_comp.set_title(f"KDE Superpuesto: {col_compare}")
        st.pyplot(fig_kde_comp, clear_figure=True)
    else:
        st.info("No hay columnas numéricas en común entre todos los archivos.")

# =======================
# Comparación de promedios por columna entre archivos
# =======================

if len(data) >= 2:
    st.subheader("📊 Comparación de promedios por columna")
    st.markdown("Visualiza los promedios de cada columna superpuestos para múltiples archivos.")
    
    # Usar las columnas comunes calculadas anteriormente
    if common_cols_list:
        fig_comp_means, ax_comp_means = plt.subplots(figsize=(12, 6))
        
        color_cycle_means = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3","C4","C5"])
        
        for i, fname in enumerate(data.keys()):
            df_file = data[fname].copy()
            idx_like = [c for c in df_file.columns if c.lower().startswith("unnamed") or c.lower() in {"index"}]
            if idx_like:
                df_file = df_file.drop(columns=idx_like)
            
            # Filtrar ceros
            num_cols_temp = [c for c in df_file.columns if pd.api.types.is_numeric_dtype(df_file[c])]
            df_file[num_cols_temp] = df_file[num_cols_temp].replace(0, np.nan)
            
            # Calcular promedios por columna
            means = df_file[common_cols_list].mean(numeric_only=True)
            
            color = color_cycle_means[i % len(color_cycle_means)]
            ax_comp_means.plot(common_cols_list, means.values, marker='o', color=color, linewidth=2, label=fname)
        
        ax_comp_means.set_ylabel("Promedio (nivel de gris)")
        ax_comp_means.set_xlabel("Columnas")
        ax_comp_means.set_xticks(range(len(common_cols_list)))
        ax_comp_means.set_xticklabels(common_cols_list, rotation=45, ha="right")
        ax_comp_means.grid(alpha=0.3)
        ax_comp_means.legend(loc="upper right", fontsize=9)
        ax_comp_means.set_title("Comparación de promedios por columna")
        st.pyplot(fig_comp_means, clear_figure=True)
    else:
        st.info("No hay columnas numéricas en común entre todos los archivos.")

# =======================
# Comparación de KDEs de promedios por fila entre archivos
# =======================

if len(data) >= 2:
    st.subheader("📈 Comparación de KDEs de promedios por fila")
    st.markdown("Visualiza KDEs de los promedios por fila de múltiples archivos superpuestos.")
    
    c1, c2 = st.columns([1, 1])
    kde_samples_row_comp = c1.number_input("Submuestreo (máx. puntos)", min_value=10000, max_value=200000, value=100000, step=10000, key="samp_row_comp")
    bw_mult_row_comp = c2.slider("Ancho de banda (×Scott)", 0.25, 3.0, 1.0, 0.05, key="bw_row_comp")
    
    fig_kde_row_comp, ax_kde_row_comp = plt.subplots(figsize=(12, 6))
    grid = np.linspace(1, 255, 255)
    
    color_cycle_row = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3","C4","C5"])
    max_yvalue_row_comp = 0
    
    for i, fname in enumerate(data.keys()):
        df_file = data[fname].copy()
        idx_like = [c for c in df_file.columns if c.lower().startswith("unnamed") or c.lower() in {"index"}]
        if idx_like:
            df_file = df_file.drop(columns=idx_like)
        
        num_cols_temp = [c for c in df_file.columns if pd.api.types.is_numeric_dtype(df_file[c])]
        df_file[num_cols_temp] = df_file[num_cols_temp].replace(0, np.nan)
        
        # Calcular promedios por fila
        row_means_comp = df_file[num_cols_temp].mean(axis=1).dropna()
        if row_means_comp.size == 0:
            continue
        
        x_row = row_means_comp.to_numpy(dtype=float)
        # Submuestreo
        if x_row.size > kde_samples_row_comp:
            rng = np.random.default_rng(42)
            x_row = rng.choice(x_row, size=kde_samples_row_comp, replace=False)
        
        # KDE
        kde_row_comp = gaussian_kde(x_row)
        kde_row_comp.set_bandwidth(bw_method=kde_row_comp.factor * bw_mult_row_comp)
        density_row_comp = kde_row_comp(grid)
        
        dx = grid[1] - grid[0]
        N_row_comp = len(x_row)
        counts_row_comp = density_row_comp * N_row_comp * dx
        
        color = color_cycle_row[i % len(color_cycle_row)]
        label = f"{fname} (N={N_row_comp})"
        ax_kde_row_comp.plot(grid, counts_row_comp, color=color, linewidth=2.5, label=label, alpha=0.8)
        
        # Marcar máximo
        max_idx_row_comp = np.argmax(counts_row_comp)
        ax_kde_row_comp.scatter(grid[max_idx_row_comp], counts_row_comp[max_idx_row_comp], color=color, s=120, marker='^', 
                               zorder=5, edgecolor='black', linewidth=1)
        lines_count_row = len(data)
        show_label_row = (lines_count_row <= 12) or (i % 2 == 0)
        if show_label_row:
            ax_kde_row_comp.annotate(f"({grid[max_idx_row_comp]:.0f}, {counts_row_comp[max_idx_row_comp]:.1f})", 
                                    xy=(grid[max_idx_row_comp], counts_row_comp[max_idx_row_comp]), 
                                    xytext=(0, 12), textcoords='offset points',
                                    ha='center', va='bottom',
                                    fontsize=8, alpha=0.9, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                                    arrowprops=dict(arrowstyle='-', color=color, alpha=0.8))
        
        max_yvalue_row_comp = max(max_yvalue_row_comp, counts_row_comp.max())
    
    ax_kde_row_comp.set_xlim(0, 255)
    ax_kde_row_comp.set_xticks(np.arange(0, 256, 50))
    ax_kde_row_comp.set_ylabel("Conteo estimado de píxeles")
    ax_kde_row_comp.set_xlabel("Nivel de gris promedio (0–255)")
    
    y_max_row = int(np.ceil(max_yvalue_row_comp / 10) * 10)
    y_step_row_comp = max(1, int(np.ceil(y_max_row / 10)))
    ax_kde_row_comp.set_yticks(np.arange(0, y_max_row + y_step_row_comp, y_step_row_comp))
    
    ax_kde_row_comp.grid(alpha=0.25)
    ax_kde_row_comp.legend(loc="upper left", fontsize=9)
    ax_kde_row_comp.set_title("KDE Superpuesto: Promedios por fila")
    st.pyplot(fig_kde_row_comp, clear_figure=True)