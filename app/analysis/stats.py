import streamlit as st
import pandas as pd
from janome.tokenizer import Tokenizer
from collections import Counter, defaultdict
import plotly.express as px


def draw_descriptive_stats(text):
    st.write("### 📊 記述統計")
    st.write("テキスト全体の単語数や、品詞別の出現割合を確認します。名詞や動詞以外の品詞も含めた全体の傾向です。")
    t = Tokenizer()
    tokens = t.tokenize(text)
    total_tokens = 0
    unique_tokens = set()
    pos_counts = Counter()
    pos_unique = defaultdict(set)
    for token in tokens:
        word = token.surface
        pos = token.part_of_speech.split(',')[0]
        total_tokens += 1
        unique_tokens.add(word)
        pos_counts[pos] += 1
        pos_unique[pos].add(word)
    st.write("#### 📌 全体サマリー")
    col1, col2 = st.columns(2)
    col1.metric("総出現数（延べ語数）", f"{total_tokens:,} 語")
    col2.metric("総語句数（異なり語数）", f"{len(unique_tokens):,} 語")
    pos_data = []
    unique_total = len(unique_tokens)
    for pos in pos_counts.keys():
        count = pos_counts[pos]
        unique_count = len(pos_unique[pos])
        pos_data.append({
            "品詞": pos,
            "出現数": count,
            "出現数割合(%)": round(count / total_tokens * 100, 1) if total_tokens > 0 else 0,
            "語句数（種類）": unique_count,
            "語句数割合(%)": round(unique_count / unique_total * 100, 1) if unique_total > 0 else 0
        })
    df_pos = pd.DataFrame(pos_data).sort_values(by="出現数", ascending=False).reset_index(drop=True)
    st.write("#### 📌 品詞別の構成")
    col_table, col_chart = st.columns([1, 1.2])
    with col_table:
        st.dataframe(df_pos, width='stretch')
    with col_chart:
        fig = px.pie(df_pos, values='出現数', names='品詞', title='品詞別の出現数割合', hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, width='stretch')
