import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


def draw_ngram(sentences_words, top_n=20):
    st.write("### 🔗 バイグラム（連続する2単語）の出現頻度")
    st.write("「どの単語とどの単語が、連続して使われやすいか（言い回しのクセ）」を可視化します。")
    bigrams = []
    for words in sentences_words:
        if len(words) >= 2:
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                bigrams.append(bigram)
    bigram_counts = Counter(bigrams)
    if not bigram_counts:
        st.warning("バイグラムを生成できるデータがありません。（抽出された単語が少なすぎる可能性があります）")
        return
    df_bigram = pd.DataFrame(bigram_counts.most_common(top_n), columns=["バイグラム（連続する2単語）", "出現回数"])
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.dataframe(df_bigram, width='stretch')
    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        df_bigram_rev = df_bigram.iloc[::-1]
        ax.barh(df_bigram_rev["バイグラム（連続する2単語）"], df_bigram_rev["出現回数"], color='#5D9CEC')
        ax.set_xlabel("出現回数")
        ax.set_title(f"上位{top_n}件のバイグラム")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        st.pyplot(fig)


def draw_tfidf_chart(sentences_words):
    st.write("### TF-IDFによる特徴語抽出")
    st.write("単なる出現回数だけでなく、「そのテキストならではの重要キーワード」を計算して上位20語を表示します。")
    if len(sentences_words) > 0:
        corpus = [" ".join(words) for words in sentences_words if len(words) > 0]
        if len(corpus) > 0:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.sum(axis=0).A1
            df_tfidf = pd.DataFrame({'語句': feature_names, 'TF-IDFスコア': tfidf_scores})
            df_tfidf = df_tfidf.sort_values(by='TF-IDFスコア', ascending=False).reset_index(drop=True)
            fig_tfidf, ax_tfidf = plt.subplots(figsize=(10, 6))
            top20_tfidf = df_tfidf.head(20)
            ax_tfidf.bar(top20_tfidf['語句'], top20_tfidf['TF-IDFスコア'], color='#FFA07A')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('TF-IDFスコア（重要度）')
            plt.tight_layout()
            col1, col2 = st.columns([1, 1])
            with col1:
                st.pyplot(fig_tfidf)
            with col2:
                st.dataframe(top20_tfidf)
        else:
            st.info("計算に必要な単語データが不足しています。")
    else:
        st.info("計算に必要な文データがありません。")
