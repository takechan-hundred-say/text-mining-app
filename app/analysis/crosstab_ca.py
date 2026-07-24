import os
import io
import tempfile
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from janome.tokenizer import Tokenizer
from collections import Counter
from sklearn.decomposition import TruncatedSVD
import numpy as np
from app.r_bridge.r_runner import run_r_script
from app.utils.tokenize import create_user_dict_file


def analyze_metadata(df, meta_cols):
    st.write("### 👥 属性データの分布")
    st.write("選択された属性データ（年齢、性別、スコアなど）の偏りや傾向を確認します。")
    if not meta_cols:
        st.info("属性データが選択されていません。"); return
    for col in meta_cols:
        st.markdown(f"#### 📌 {col} の分布")
        valid_data = df[col].dropna()
        if valid_data.empty:
            st.warning(f"「{col}」には有効なデータがありません。"); continue
        col1, col2 = st.columns([1, 1.2])
        if pd.api.types.is_numeric_dtype(valid_data):
            with col1:
                st.write("【記述統計】"); st.dataframe(valid_data.describe().to_frame(name="統計量"), width='stretch')
            with col2:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(valid_data, bins=10, color='#4A90E2', edgecolor='black', alpha=0.7)
                ax.set_ylabel("人数 / 件数"); ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
                st.pyplot(fig)
        else:
            with col1:
                st.write("【度数分布】"); st.dataframe(valid_data.value_counts().to_frame(name="件数"), width='stretch')
            with col2:
                fig, ax = plt.subplots(figsize=(5, 3))
                value_counts_head = valid_data.value_counts().head(10)
                ax.barh(value_counts_head.index[::-1], value_counts_head.values[::-1], color='#50E3C2')
                ax.set_xlabel("人数 / 件数"); ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
                st.pyplot(fig)
        st.markdown("---")


def draw_crosstab_and_ca(df, text_col, meta_cols, target_pos, synonym_dict, stopwords, compound_words=None):
    st.write("### 🔀 属性別のクロス集計・コレスポンデンス分析")
    if not meta_cols:
        st.info("ファイル読み込み時に属性データ（メタデータ）の列が選択されていません。"); return
    col1, col2 = st.columns(2)
    with col1:
        selected_meta = st.selectbox("比較したい属性を選んでください:", meta_cols)
    with col2:
        top_n = st.slider("集計に含める上位単語数:", 10, 100, 30, step=10, key="crosstab_slider")
    with st.spinner("クロス集計表とマップを作成中..."):
        temp_dict_path = None
        if compound_words:
            temp_dict_path = create_user_dict_file(compound_words)
            t = Tokenizer(udic=temp_dict_path, udic_enc='utf8', udic_type='ipadic')
        else:
            t = Tokenizer()
        all_rows_words = []
        for idx, row in df.iterrows():
            text = str(row[text_col])
            if pd.isna(text) or text.strip() == "":
                all_rows_words.append([]); continue
            words = []
            for token in t.tokenize(text):
                pos = token.part_of_speech.split(',')[0]
                base_form = token.base_form if token.base_form != '*' else token.surface
                if pos in target_pos:
                    base_form = synonym_dict.get(base_form, base_form)
                    if base_form not in stopwords: words.append(base_form)
            all_rows_words.append(words)
        all_words_flat = [w for sublist in all_rows_words for w in sublist]
        if not all_words_flat:
            st.warning("単語が抽出できませんでした。"); return
        top_words = [w for w, c in Counter(all_words_flat).most_common(top_n)]
        crosstab_data = []; meta_values = df[selected_meta].dropna().unique()
        for word in top_words:
            row_data = {"単語": word}
            for meta_value in meta_values:
                indices = df[df[selected_meta] == meta_value].index
                meta_words = [w for i in indices if i < len(all_rows_words) for w in all_rows_words[i]]
                row_data[str(meta_value)] = meta_words.count(word)
            crosstab_data.append(row_data)
        df_crosstab = pd.DataFrame(crosstab_data).set_index("単語")
        st.write(f"#### 📌 「{selected_meta}」別の頻出単語 {top_n}語 の集計表")
        st.dataframe(df_crosstab, width='stretch')
        csv_cross = df_crosstab.to_csv().encode('utf-8-sig')
        st.download_button("📥 クロス集計表をCSVでダウンロード", csv_cross, f"crosstab_{selected_meta}.csv", "text/csv")
        st.markdown("---")
        st.write("#### 🗺️ コレスポンデンス分析（対応分析）マップ")
        st.write("属性（赤色の▲）と単語（青色の●）の位置関係をマップ化します。")
        col_method = st.columns(2)
        with col_method[0]: st.write("**グラフの描画方法を選択：**")
        with col_method[1]:
            ca_method = st.radio("実行方法:", ["Python版", "R版"], key=f"ca_method_{selected_meta}", horizontal=True)
        if ca_method == "R版":
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    draw_ca_with_r(df_crosstab, selected_meta, tmpdir)
                except Exception as e:
                    st.error(f"R版でのCA実行中にエラー: {e}")
        else:
            try:
                X = df_crosstab.values.astype(float); row_sums = X.sum(axis=1); col_sums = X.sum(axis=0)
                valid_rows = row_sums > 0; valid_cols = col_sums > 0
                X = X[valid_rows][:, valid_cols]
                words_labels = df_crosstab.index[valid_rows].tolist(); meta_labels = df_crosstab.columns[valid_cols].tolist()
                total = X.sum(); P = X / total; r = P.sum(axis=1); c = P.sum(axis=0)
                E = np.outer(r, c); Z = (P - E) / np.sqrt(E)
                svd = TruncatedSVD(n_components=2, random_state=42); svd.fit(Z)
                row_coords = svd.transform(Z) / np.sqrt(r[:, np.newaxis])
                col_coords = svd.components_.T * svd.singular_values_ / np.sqrt(c[:, np.newaxis])
                fig_ca, ax_ca = plt.subplots(figsize=(10, 8))
                ax_ca.scatter(row_coords[:, 0], row_coords[:, 1], c='#4A90E2', alpha=0.5, marker='o', s=50)
                for i, txt in enumerate(words_labels):
                    ax_ca.annotate(txt, (row_coords[i, 0], row_coords[i, 1]), color='#333333', fontsize=11, ha='center', va='bottom')
                ax_ca.scatter(col_coords[:, 0], col_coords[:, 1], c='#E94A66', marker='^', s=200, edgecolors='white', linewidth=1.5, zorder=5)
                for i, txt in enumerate(meta_labels):
                    ax_ca.annotate(txt, (col_coords[i, 0], col_coords[i, 1]), color='#E94A66', fontsize=15, fontweight='bold', ha='center', va='bottom')
                ax_ca.axhline(0, color='gray', linestyle='--', alpha=0.3); ax_ca.axvline(0, color='gray', linestyle='--', alpha=0.3)
                ax_ca.set_title(f"「{selected_meta}」と頻出単語の関連性マップ", fontsize=14)
                st.pyplot(fig_ca)
                buf_ca = io.BytesIO(); fig_ca.savefig(buf_ca, format="png", dpi=300, bbox_inches='tight')
                st.download_button("🖼️ マップをPNGで保存", data=buf_ca.getvalue(), file_name=f"correspondence_map_{selected_meta}.png", mime="image/png")
            except Exception as e:
                st.error(f"コレスポンデンス分析の計算中にエラー: {e}")
        if temp_dict_path and os.path.exists(temp_dict_path):
            os.remove(temp_dict_path)


def draw_ca_with_r(df_crosstab, selected_meta, tmpdir):
    csv_path = os.path.join(tmpdir, "crosstab.csv"); png_path = os.path.join(tmpdir, "ca_plot.png")
    df_crosstab.reset_index().to_csv(csv_path, index=False, encoding='utf-8-sig')
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script_path = os.path.join(project_root, "r_scripts", "correspondence_analysis.R")
    ok = run_r_script(script_path, csv_path, png_path, selected_meta)
    if ok and os.path.exists(png_path):
        st.image(png_path, width='stretch')
        with open(png_path, "rb") as f:
            st.download_button("🖼️ CA図をPNGでダウンロード", f.read(), file_name=f"ca_{selected_meta}.png", mime="image/png")
        st.markdown("---"); st.write("#### 📊 コレスポンデンス分析の結果表")
        row_coords_file = os.path.join(tmpdir, "ca_row_coords.csv")
        col_coords_file = os.path.join(tmpdir, "ca_col_coords.csv")
        eigenvalue_file = os.path.join(tmpdir, "ca_eigenvalue.csv")
        tab_row, tab_col, tab_eig = st.tabs(["行座標（単語）", "列座標（属性）", "固有値"])
        with tab_row:
            if os.path.exists(row_coords_file):
                df_row = pd.read_csv(row_coords_file); st.write("**単語の座標と寄与率**"); st.dataframe(df_row, width='stretch')
                with open(row_coords_file, "rb") as f: st.download_button("📥 単語座標をCSVでダウンロード", f.read(), file_name=f"ca_row_coords_{selected_meta}.csv", mime="text/csv", key="dl_row_coords")
            else: st.warning("行座標ファイルが見つかりません")
        with tab_col:
            if os.path.exists(col_coords_file):
                df_col = pd.read_csv(col_coords_file); st.write("**属性値の座標と寄与率**"); st.dataframe(df_col, width='stretch')
                with open(col_coords_file, "rb") as f: st.download_button("📥 属性座標をCSVでダウンロード", f.read(), file_name=f"ca_col_coords_{selected_meta}.csv", mime="text/csv", key="dl_col_coords")
            else: st.warning("列座標ファイルが見つかりません")
        with tab_eig:
            if os.path.exists(eigenvalue_file):
                df_eig = pd.read_csv(eigenvalue_file); st.write("**固有値と寄与率**"); st.dataframe(df_eig, width='stretch')
                with open(eigenvalue_file, "rb") as f: st.download_button("📥 固有値をCSVでダウンロード", f.read(), file_name=f"ca_eigenvalue_{selected_meta}.csv", mime="text/csv", key="dl_eigenvalue")
            else: st.warning("固有値ファイルが見つかりません")
    else:
        st.error("R版でのコレスポンデンス分析の実行に失敗しました")
