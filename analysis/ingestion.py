from io import StringIO
import pandas as pd

def read_csv_robust(file) -> pd.DataFrame:
    """Lee un CSV intentando varios encodings y delimitadores.
    También soporta archivos Excel (.xls/.xlsx) detectando la extensión
    del nombre del archivo (p. ej. UploadedFile.name en Streamlit).
    """
    # Si el archivo tiene nombre y es Excel, usar read_excel
    name = getattr(file, "name", "")
    if isinstance(name, str) and name.lower().endswith(('.xls', '.xlsx')):
        try:
            file.seek(0)
            return pd.read_excel(file)
        except Exception:
            try:
                file.seek(0)
                return pd.read_excel(file, engine="openpyxl")
            except Exception:
                pass

    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        try:
            return pd.read_csv(file, encoding="latin1")
        except Exception:
            pass
    except Exception:
        pass

    # fallback: leer como texto y reparsear
    try:
        file.seek(0)
        text = file.read().decode(errors="ignore")
        return pd.read_csv(StringIO(text))
    except Exception:
        # último recurso: delimitador ';'
        try:
            file.seek(0)
            return pd.read_csv(file, sep=";")
        except Exception as e:
            raise RuntimeError(f"No se pudo leer el CSV: {e}")

def infer_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parsea fechas y convierte columnas numéricas cuando sea razonable."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            try:
                parsed = pd.to_datetime(out[c], errors="raise")
                if parsed.notna().mean() > 0.6:
                    out[c] = parsed
                    continue
            except Exception:
                pass
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out
