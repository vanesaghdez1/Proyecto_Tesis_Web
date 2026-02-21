import os, sys, platform, streamlit as st

st.set_page_config(page_title="Prueba", layout="wide")
st.title("✅ Hola, Streamlit está corriendo")
st.write("Si ves este texto, todo está bien.")

st.subheader("Contexto de ejecución")
st.code(f"""
cwd: {os.getcwd()}
archivos en cwd: {os.listdir()}
python: {sys.executable}
python version: {platform.python_version()}
sys.path (3 primeros):
{sys.path[:3]}
""")