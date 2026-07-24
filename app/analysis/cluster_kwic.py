import io
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from app.utils.tokenize import create_user_dict_file


def draw_cluster_analysis(text, df_result, target_pos, synonym_dict, stopwords, compound_words=None):
    st.write("### 🌳/📍 クラスター分析（単語のグループ化）")
    st.write("単語同士が「どのくらい同じ文脈で使われているか」を計算し、グループ化します。")
    col1, col2, col3 = st.columns(3)
    with col1:
        unit_option = st.selectbox("分析単位", ["段落単位（改行）", "一文単位（句点）", "ファイル全体", "単語単位（10語区切り）"])
    with col2:
        top_n = st.slider("分析に含める上位単語数", min_value=10, max_value=100, value=30, step=5)
    with col3:
        plot_type = st.radio("グラフの種類", ["樹形図（階層型）", "散布図（K-means）"])
    if plot_type == "散布図（K-means）":
        n_clusters = st.number_input("クラスター数", min_value=2, max_value=10, value=3)
    with st.spinner("テキストを分割し、クラスターを計算中..."):
        temp_dict_path = None
        if compound_words:
            temp_dict_path = create_user_dict_file(compound_words)
            t = Tokenizer(udic=temp_dict_path, udic_enc='utf8', udic_type='ipadic')
        else:
            t = Tokenizer()
        docs_words = []
        if unit_option == "単語単位（10語区切り）":
            all_valid_words = []
            for token in t.tokenize(text):
                pos = token.part_of_speech.split(',')[0]
                base_form = token.base_form if token.base_form != '*' else token.surface
                if pos in target_pos:
                    base_form = synonym_dict.get(base_form, base_form)
                    if base_form not in stopwords:
                        all_valid_words.append(base_form)
            for i in range(0, len(all_valid_words), 10):
                chunk = all_valid_words[i:i+10]
                if len(chunk) > 0:
                    docs_words.append(" ".join(chunk))
        else:
            if unit_option == "ファイル全体": docs = [text]
            elif unit_option == "段落単位（改行）": docs = [p.strip() for p in text.split('\n') if len(p.strip()) > 0]
            elif unit_option == "一文単位（句点）": docs = [s.strip() + '。' for s in text.replace('\n', '。').split('。') if len(s.strip()) > 0]
            for doc in docs:
                words = []
                for token in t.tokenize(doc):
                    pos = token.part_of_speech.split(',')[0]
                    base_form = token.base_form if token.base_form != '*' else token.surface
                    if pos in target_pos:
                        base_form = synonym_dict.get(base_form, base_form)
                        if base_form not in stopwords:
                            words.append(base_form)
                if len(words) > 0:
                    docs_words.append(" ".join(words))
        if len(docs_words) < 2:
            st.warning("分割された文書が少なすぎます。別の分析単位を選択してください。"); return
        top_words = df_result['語句'].head(top_n).tolist()
        vectorizer = CountVectorizer(vocabulary=top_words)
        X = vectorizer.fit_transform(docs_words).toarray()
        X_T = X.T; valid_indices = X_T.sum(axis=1) > 0; X_T_valid = X_T[valid_indices]
        valid_words = [top_words[i] for i, valid in enumerate(valid_indices) if valid]
        min_clusters = n_clusters if plot_type == "散布図（K-means）" else 2
        if len(valid_words) < min_clusters:
            st.warning("有効な単語数が不足しています。抽出条件を見直してください。"); return
        fig_cluster, ax_cluster = plt.subplots(figsize=(10, 6))
        if plot_type == "樹形図（階層型）":
            Z = linkage(X_T_valid, method='ward', metric='euclidean')
            dendrogram(Z, labels=valid_words, orientation='right', ax=ax_cluster, leaf_font_size=12)
            ax_cluster.set_title(f"単語の樹形図（{unit_option} / 上位{top_n}語）", fontsize=14); ax_cluster.set_xlabel("距離（非類似度）")
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_T_valid)
            pca = PCA(n_components=2, random_state=42); coords = pca.fit_transform(X_T_valid)
            ax_cluster.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap='tab10', alpha=0.7, s=150)
            for i, word in enumerate(valid_words):
                ax_cluster.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=11, alpha=0.9, xytext=(5, 5), textcoords='offset points')
            ax_cluster.set_title(f"単語のクラスター散布図（PCA + K-means / {n_clusters}グループ）", fontsize=14)
            ax_cluster.set_xlabel("第1主成分 (PC1)"); ax_cluster.set_ylabel("第2主成分 (PC2)"); ax_cluster.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_cluster)
        buf_cluster = io.BytesIO(); fig_cluster.savefig(buf_cluster, format="png", dpi=300, bbox_inches='tight')
        st.download_button("🖼️ グラフをPNGで保存", data=buf_cluster.getvalue(), file_name=f"cluster_{'dendrogram' if plot_type=='樹形図（階層型）' else 'scatter'}.png", mime="image/png")
        if temp_dict_path and os.path.exists(temp_dict_path):
            os.remove(temp_dict_path)


def draw_kwic(text, df_result, synonym_dict=None, tokenizer=None):
    from janome.tokenizer import Tokenizer
    st.write("### 🔍 文脈抽出（KWIC）")
    st.write("特定の単語が、テキストの中でどのような文脈（前後関係）で使われているかを確認できます。")
    if synonym_dict is None: synonym_dict = {}
    if tokenizer is None: tokenizer = Tokenizer()
    st.info("💡 頻出語の形態素解析結果に基づいて検索します。同義語・ゆらぎ統一辞書にも対応しています。")
    top_words = df_result['語句'].head(100).tolist()
    col1, col2 = st.columns(2)
    with col1:
        selected_word = st.selectbox("頻出語から選択（トップ100）:", options=["(直接入力する)"] + top_words)
    with col2:
        custom_word = st.text_input("自由に検索したい単語を入力（任意）:")
    target_word = custom_word if custom_word else (selected_word if selected_word != "(直接入力する)" else "")
    if target_word:
        sentences = [s.strip() for s in text.replace('\n', '。').split('。') if s.strip()]
        matched_sentences = []
        search_words = [target_word]
        if synonym_dict:
            for key, value in synonym_dict.items():
                if value == target_word:
                    search_words.append(key)
        for sentence in sentences:
            if not sentence.strip():
                continue
            tokens = [token.base_form for token in tokenizer.tokenize(sentence)]
            tokens_normalized = [synonym_dict.get(token, token) for token in tokens]
            in_text = any(word in sentence for word in search_words)
            in_tokens = any(word in tokens_normalized for word in search_words)
            if in_text or in_tokens:
                matched_sentences.append(sentence + '。')
        if matched_sentences:
            st.success(f"「**{target_word}**」を含む文が **{len(matched_sentences)}件** 見つかりました。")
            html_content = ""
            for i, sentence in enumerate(matched_sentences, 1):
                highlighted_text = sentence.replace(target_word, f"<mark style='background-color: #ffeb3b; font-weight: bold; color: black; padding: 0 4px; border-radius: 3px;'>{target_word}</mark>")
                html_content += f"<div style='padding: 8px; border-bottom: 1px solid #ddd; line-height: 1.6;'><b>{i}.</b> {highlighted_text}</div>"
            st.markdown(f"<div style='max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #fafafa;'>{html_content}</div>", unsafe_allow_html=True)
            df_kwic = pd.DataFrame({"No.": range(1, len(matched_sentences)+1), "文脈": matched_sentences})
            csv = df_kwic.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 抽出結果をCSVでダウンロード", data=csv, file_name=f"kwic_{target_word}.csv", mime="text/csv")
        else:
            st.warning(f"「{target_word}」を含む文は見つかりませんでした。")
    else:
        st.info("👆 確認したい単語を選択するか、入力してください。")
