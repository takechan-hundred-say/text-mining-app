import os
import subprocess
import tempfile
import streamlit as st


def run_r_script(script_path, *args):
    r_path = st.session_state.R_PATH
    if not r_path:
        st.error("❌ R がインストールされていません。セットアップを実行してください。")
        return None
    result = subprocess.run([r_path, script_path] + list(args), capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        st.error(f"Rスクリプトエラー:\n{result.stderr}")
        return None
    return True


def draw_frequency_chart_r(df_result, top_n=30):
    st.write("### 📈 頻出語句グラフ（R版）")
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "word_freq.csv"); png_path = os.path.join(tmpdir, "word_freq.png")
        df_result.to_csv(csv_path, index=False, encoding='utf-8-sig')
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        script_path = os.path.join(project_root, "r_scripts", "word_frequency.R")
        ok = run_r_script(script_path, csv_path, png_path, str(top_n))
        if ok and os.path.exists(png_path):
            st.image(png_path, width='stretch')
            with open(png_path, "rb") as f:
                st.download_button("🖼️ グラフをPNGでダウンロード", f.read(), file_name="word_frequency_r.png", mime="image/png")
