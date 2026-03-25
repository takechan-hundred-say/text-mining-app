import streamlit as st
import pandas as pd
from janome.tokenizer import Tokenizer
import docx
from collections import Counter, defaultdict
import io
import csv
import tempfile
import os
import sys          # ★ここを追加！  
import pypdf
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx
from itertools import combinations
from pyvis.network import Network
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer # CountVectorizerを追記
from scipy.cluster.hierarchy import linkage, dendrogram # この行を新規追加
from sklearn.cluster import KMeans           # ← この行を追加
from sklearn.decomposition import PCA        # ← この行を追加
from asari.api import Sonar
import plotly.express as px
import zipfile
import urllib.request
import pickle
import signal

# Windows標準のフォント（メイリオ）を指定
plt.rcParams['font.family'] = 'Meiryo'

def load_synonym_dict(uploaded_csv):
    synonym_dict = {}
    if uploaded_csv is not None:
        try:
            df_syn = pd.read_csv(uploaded_csv, header=None)
            for _, row in df_syn.iterrows():
                if pd.notna(row[0]) and pd.notna(row[1]):
                    synonym_dict[str(row[0]).strip()] = str(row[1]).strip()
        except Exception as e:
            st.error(f"同義語辞書の読み込みに失敗しました: {e}")
    return synonym_dict

# ↓ ★ここに新しく追加する（load_synonym_dict のすぐ下）
@st.cache_data
def load_polarity_dict_tohoku():
    import unicodedata

    # ★ exe化（frozen）時と通常実行時でパスを切り替える
    # （ここに先ほどのコードが入っています）
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS   # exe化時は一時展開フォルダ
    else:
        base = os.path.dirname(os.path.abspath(__file__))

    dict_path = os.path.join(base, 'dic', 'pn.csv.m3.120408.trim')

    polarity_dict = {}
    if not os.path.exists(dict_path):
        st.warning(
            f"辞書ファイルが見つかりません: {dict_path}\n"
            "dicフォルダに `pn.csv.m3.120408.trim` を配置してください。"
        )
        return None

    try:
        df_pncsv = pd.read_csv(
            dict_path,
            sep="\t",
            names=["term", "sentiment", "semantic_category"],
            encoding="utf-8"
        )
        for _, row in df_pncsv.iterrows():
            term_nfkc = unicodedata.normalize('NFKC', str(row['term']))
            polarity_dict[term_nfkc] = str(row['sentiment'])
    except Exception as e:
        st.error(f"東北大学極性辞書の読み込みに失敗しました: {e}")
        return None

    return polarity_dict

def extract_text(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    text = ""
    if file_type == 'txt':
        raw_data = uploaded_file.read()
        try:
            text = raw_data.decode('utf-8')
        except UnicodeDecodeError:
            text = raw_data.decode('cp932')
    elif file_type == 'csv':
        df = pd.read_csv(uploaded_file)
        text = " ".join(df.astype(str).values.flatten())
    elif file_type == 'docx':
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif file_type == 'pdf':
        try:
            reader = pypdf.PdfReader(uploaded_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.error(f"PDFの読み込み中にエラーが発生しました: {e}")
    return text

def create_zip_data(text, df_result):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("1_extracted_text.txt", text.encode('utf-8-sig'))
        if not df_result.empty:
            csv_data = df_result.to_csv(index=False).encode('utf-8-sig')
            zip_file.writestr("2_word_frequency.csv", csv_data)
    return zip_buffer.getvalue()

def create_user_dict_file(word_list):
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix='.csv', newline='')
    writer = csv.writer(temp_file)
    for word in word_list:
        word = word.strip()
        if word:
            row = [word, -1, -1, -5000, '名詞', 'カスタム', '*', '*', '*', '*', word, '*', '*']
            writer.writerow(row)
    temp_file.close()
    return temp_file.name

@st.cache_data
def analyze_text(text, option, custom_words_text, custom_dict_content, stop_words_text, stop_words_content, target_pos, synonym_dict):
    tokenizer = None
    temp_dict_path = None

    stopwords = set([w.strip() for w in stop_words_text.replace('\n', ',').split(',') if w.strip()])
    if stop_words_content is not None:
        file_stopwords = [w.strip() for w in stop_words_content.replace('\n', ',').split(',') if w.strip()]
        stopwords.update(file_stopwords)

    words = []
    if option == "1. 画面上で直接、語句定義を入力する" and custom_words_text:
        words = [w.strip() for w in custom_words_text.replace('\n', ',').split(',') if w.strip()]
    elif option == "2. ユーザーが作成した定義ファイルを読み込む" and custom_dict_content:
        words = [w.strip() for w in custom_dict_content.replace('\n', ',').split(',') if w.strip()]
        
    if words:
        temp_dict_path = create_user_dict_file(words)
        tokenizer = Tokenizer(udic=temp_dict_path, udic_enc='utf8', udic_type='ipadic')
    else:
        tokenizer = Tokenizer()

    sentences = text.replace('。', '。\n').split('\n')
    all_words_list = []
    sentences_words = []

    for sentence in sentences:
        if not sentence.strip(): continue
        
        current_sentence_words = []
        if option == "4. 連続する名詞を自動結合する（ルールベース）":
            current_compound = ""
            for token in tokenizer.tokenize(sentence):
                pos = token.part_of_speech.split(',')[0]
                if pos == '名詞':
                    current_compound += token.surface
                else:
                    if current_compound:
                        current_compound = synonym_dict.get(current_compound, current_compound)
                        if current_compound not in stopwords and '名詞' in target_pos:
                            current_sentence_words.append(current_compound)
                            all_words_list.append((current_compound, '名詞'))
                        current_compound = ""
                    
                    if pos in target_pos and pos != '名詞':
                        base_form = token.base_form if token.base_form != '*' else token.surface
                        base_form = synonym_dict.get(base_form, base_form)
                        if base_form not in stopwords:
                            current_sentence_words.append(base_form)
                            all_words_list.append((base_form, pos))
                            
            if current_compound:
                current_compound = synonym_dict.get(current_compound, current_compound)
                if current_compound not in stopwords and '名詞' in target_pos:
                    current_sentence_words.append(current_compound)
                    all_words_list.append((current_compound, '名詞'))
                
        else:
            for token in tokenizer.tokenize(sentence):
                pos = token.part_of_speech.split(',')[0]
                base_form = token.base_form if token.base_form != '*' else token.surface
                
                if pos in target_pos:
                    base_form = synonym_dict.get(base_form, base_form)
                    if base_form not in stopwords:
                        current_sentence_words.append(base_form)
                        all_words_list.append((base_form, pos))
                    
        if current_sentence_words:
            sentences_words.append(list(set(current_sentence_words)))

    if temp_dict_path and os.path.exists(temp_dict_path):
        os.remove(temp_dict_path)

    word_counts = Counter(all_words_list)
    df_data = [{'語句': word, '品詞': pos, '頻度': count} for (word, pos), count in word_counts.items()]
    df = pd.DataFrame(df_data)
    
    if not df.empty:
        df = df.sort_values(by='頻度', ascending=False).reset_index(drop=True)
        
    return df, sentences_words

def draw_frequency_chart(df_result):
    st.write("### 出現頻度トップ20の単語")
    fig, ax = plt.subplots(figsize=(10, 6))
    top20 = df_result.head(20)
    ax.bar(top20['語句'], top20['頻度'], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('出現頻度')
    plt.tight_layout()
    st.pyplot(fig)
    
    buf_bar = io.BytesIO()
    fig.savefig(buf_bar, format="png", dpi=300)
    st.download_button("🖼️ 棒グラフをPNGで保存", data=buf_bar.getvalue(), file_name="bar_chart.png", mime="image/png")

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
        st.dataframe(df_bigram, use_container_width=True)
        
    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        df_bigram_rev = df_bigram.iloc[::-1]
        ax.barh(df_bigram_rev["バイグラム（連続する2単語）"], df_bigram_rev["出現回数"], color='#5D9CEC')
        ax.set_xlabel("出現回数")
        ax.set_title(f"上位{top_n}件のバイグラム")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        st.pyplot(fig)

def draw_wordcloud(df_result):
    import numpy as np
    
    st.write("### ワードクラウド（出現頻度が高いほど大きく表示）")
    wc_text = " ".join([" ".join([word] * count) for word, count in zip(df_result['語句'], df_result['頻度'])])
    
    font_path = r"C:\Windows\Fonts\meiryo.ttc" 
    if not os.path.exists(font_path):
        font_path = None
        
    # 楕円形のマスク（型抜き）を作成
    width, height = 800, 400
    y, x = np.ogrid[:height, :width]
    # 楕円の中心と半径を設定
    center_y, center_x = height / 2, width / 2
    radius_y, radius_x = height / 2, width / 2
    
    # 楕円の外側を255（白・描画しない領域）に設定
    mask = ((x - center_x) / radius_x)**2 + ((y - center_y) / radius_y)**2 > 1
    mask = 255 * mask.astype(int)
        
    # prefer_horizontal=1.0 で横書きのみに設定し、maskを適用
    wc = WordCloud(
        width=width, 
        height=height, 
        background_color='white', 
        font_path=font_path, 
        colormap='viridis', 
        collocations=False,
        prefer_horizontal=1.0,  # 1. 横書きのみ
        mask=mask               # 2. 楕円形のマスクを適用
    ).generate(wc_text)
    
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis("off")
    plt.tight_layout()
    st.pyplot(fig_wc)
    
    buf_wc = io.BytesIO()
    fig_wc.savefig(buf_wc, format="png", dpi=300)
    st.download_button("🖼️ ワードクラウドをPNGで保存", data=buf_wc.getvalue(), file_name="wordcloud.png", mime="image/png")

def draw_cooccurrence_network(df_result, sentences_words):
    st.write("### 共起ネットワーク図（一緒に出現しやすい単語のつながり）")
    
    col1, col2 = st.columns(2)
    with col1:
        num_words = st.slider("ネットワークに含める上位単語の数", min_value=10, max_value=100, value=30, step=5)
    with col2:
        min_edge_weight = st.slider("線を結ぶ最低の共起回数（つながりの強さ）", min_value=1, max_value=20, value=2, step=1)
        
    st.write(f"※頻出上位 {num_words} 語のうち、同じ文に {min_edge_weight} 回以上一緒に出現した単語同士を結んでいます。")
    st.markdown("**【ノードの色】** 🟢 名詞　🔴 動詞　🔵 形容詞　⚪ その他")
        
    top_words = set(df_result.head(num_words)['語句'].tolist())
    edge_list = []
    for words in sentences_words:
        valid_words = [w for w in words if w in top_words]
        if len(valid_words) >= 2:
            for pair in combinations(valid_words, 2):
                edge_list.append(tuple(sorted(pair)))
                
    edge_counts = Counter(edge_list)
    G = nx.Graph()
    for (w1, w2), weight in edge_counts.items():
        if weight >= min_edge_weight: 
            G.add_edge(w1, w2, weight=weight)
            
    if len(G.nodes) > 0:
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False)
        t_color = Tokenizer()
        node_colors_for_static = []
        
        for node in G.nodes():
            tokens = list(t_color.tokenize(node))
            node_color = '#D3D3D3'
            if tokens:
                pos_name = tokens[0].part_of_speech.split(',')[0]
                if pos_name == '名詞':
                    node_color = '#90EE90'
                elif pos_name == '動詞':
                    node_color = '#FFB6C1'
                elif pos_name == '形容詞':
                    node_color = '#ADD8E6'
            
            node_colors_for_static.append(node_color)
            net.add_node(node, label=node, title=f"単語: {node}", color=node_color, size=20)
            
        for u, v, data in G.edges(data=True):
            net.add_edge(u, v, value=data['weight'], title=f"共起回数: {data['weight']}回", color='#A0C4FF')
            
        net.repulsion(node_distance=120, central_gravity=0.05, spring_length=100, spring_strength=0.05)
        
        path = "html_files"
        if not os.path.exists(path):
            os.makedirs(path)
        net.save_graph(f"{path}/network.html")
        
        with open(f"{path}/network.html", 'r', encoding='utf-8') as HtmlFile:
            source_code = HtmlFile.read()
        
        components.html(source_code, height=650)
    else:
        st.warning("指定された条件（単語数・共起回数）では、つながりが見つかりませんでした。スライダーの数値を小さく調整してみてください。")

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

# ============================================================
# テキスト全体の感情分析（東北大学評価極性辞書・一本化版）
# ============================================================
def draw_sentiment_analysis(text):
    import os
    import unicodedata
    from janome.tokenizer import Tokenizer as JanomeTokenizer

    st.write("### 感情分析（ネガポジ判定）")
    st.write("東北大学評価極性辞書（名詞編）を使用して、文章中の各文の感情をポジティブ／ネガティブに判定します。")

    # ---- 辞書の読み込み ----
    dic_path = os.path.join("dic", "pn.csv.m3.120408.trim")
    if not os.path.exists(dic_path):
        st.error(f"辞書ファイルが見つかりません: {dic_path}")
        st.caption("「dic」フォルダに「pn.csv.m3.120408.trim」を配置してください。")
        return

    df_pn = pd.read_csv(dic_path, sep="\t", names=["term", "sentiment", "category"])
    df_pn["term"] = [unicodedata.normalize("NFKC", t) for t in df_pn["term"]]
    pn_dict = {}
    for _, r in df_pn.iterrows():
        s = r["sentiment"]
        if s == "p":
            pn_dict[r["term"]] = 1.0
        elif s == "n":
            pn_dict[r["term"]] = -1.0
        elif s == "e":
            pn_dict[r["term"]] = 0.0

    # ---- 文に分割 ----
    sentences = [s.strip() + "。" for s in
                 text.replace("\n", "。").split("。") if s.strip()]
    if not sentences:
        st.warning("分析する文がありません。")
        return

    t = JanomeTokenizer()
    results = []

    for sentence in sentences:
        if len(sentence) <= 1:
            continue

        tokens = list(t.tokenize(sentence))
        scores = [pn_dict[unicodedata.normalize("NFKC", tk.base_form)]
                  for tk in tokens
                  if unicodedata.normalize("NFKC", tk.base_form) in pn_dict]

        if not scores:
            continue

        avg_score = sum(scores) / len(scores)
        pos_count = scores.count(1.0)
        neg_count = scores.count(-1.0)

        if abs(avg_score) < 0.05:
            label = "😐 ニュートラル"
        elif avg_score > 0:
            label = "😊 ポジティブ"
        else:
            label = "😞 ネガティブ"

        results.append({
            "文": sentence,
            "判定": label,
            "平均スコア": round(avg_score, 3),
            "ポジティブ語数": pos_count,
            "ネガティブ語数": neg_count,
            "推移スコア": avg_score
        })

    df_sentiment = pd.DataFrame(results)

    if df_sentiment.empty:
        st.info("判定できる文がありませんでした。（辞書にヒットする語がありません）")
        return

    pos_count_total = len(df_sentiment[df_sentiment["判定"] == "😊 ポジティブ"])
    neg_count_total = len(df_sentiment[df_sentiment["判定"] == "😞 ネガティブ"])
    neu_count_total = len(df_sentiment[df_sentiment["判定"] == "😐 ニュートラル"])

    col1, col2, col3 = st.columns(3)
    col1.metric("ポジティブな文", f"{pos_count_total}件")
    col2.metric("ネガティブな文", f"{neg_count_total}件")
    col3.metric("ニュートラルな文", f"{neu_count_total}件")

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ['ポジティブ', 'ネガティブ', 'ニュートラル']
    sizes  = [pos_count_total, neg_count_total, neu_count_total]
    colors = ['#ff9999', '#66b3ff', '#d3d3d3']

    if sum(sizes) > 0:
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')

        col_chart, col_table = st.columns([1, 1])
        with col_chart:
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300)
            st.download_button(
                "🖼️ 円グラフを保存",
                data=buf.getvalue(),
                file_name="sentiment_pie.png",
                mime="image/png"
            )
        with col_table:
            st.dataframe(
                df_sentiment[["文", "判定", "平均スコア", "ポジティブ語数", "ネガティブ語数"]],
                height=300
            )
    else:
        st.info("判定できる文がありませんでした。")

    # ---- 感情の推移グラフ ----
    st.markdown("---")
    st.write("#### 📈 文章の展開に伴う感情の推移")
    st.write("横軸が文章の進行（最初から最後）、縦軸が感情スコアを示します。")

    fig_line, ax_line = plt.subplots(figsize=(10, 4))
    ax_line.plot(df_sentiment.index + 1, df_sentiment["推移スコア"],
                 marker='o', linestyle='-', color='#9b59b6')
    ax_line.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_line.set_xlabel("文の順番")
    ax_line.set_ylabel("感情スコア (←ネガティブ / ポジティブ→)")
    ax_line.spines['right'].set_visible(False)
    ax_line.spines['top'].set_visible(False)
    st.pyplot(fig_line)
    plt.close(fig_line)


# ============================================================
# ケース別感情分析（グループ集計対応・Excel 2シート出力）
# ============================================================
def draw_sentiment_by_case(df_meta, text_col, meta_cols):
    """
    ケース別感情分析（完全版）
    - 2方式対応（asari / 東北大学評価極性辞書）
    - 分析対象テキスト列の選択（自由記述が複数列ある場合に対応）
    - ケース別スコア一覧
    - グループ別：記述統計 ＋ 確認用棒グラフ
    - Excel 2シート一括ダウンロード（ケース別＋グループ別）
    """

    st.write("### 📋 ケース別 感情分析")

    # ============================================================
    # ① 分析対象テキスト列の選択
    # ============================================================
    all_cols = list(df_meta.columns)
    text_col_options = [text_col] + [c for c in all_cols if c != text_col]

    selected_text_col = st.selectbox(
        "📝 感情分析する質問列（テキスト列）を選んでください",
        text_col_options,
        index=0,
        key="case_sentiment_text_col"
    )
    st.caption(f"現在選択中: **{selected_text_col}**　（他の自由記述列がある場合は切り替えてください）")

    # ============================================================
    # ② 分析方式の選択
    # ============================================================
    method = st.radio(
        "感情分析の方式を選択してください",
        ["asari（AIモデル）", "東北大学評価極性辞書（名詞編）"],
        horizontal=True,
        key="case_sentiment_method"
    )

    # ============================================================
    # ③ グループ化する属性列の選択
    # ============================================================
    group_col = None
    if meta_cols:
        group_col_input = st.selectbox(
            "📌 グループ化する属性列を選んでください",
            ["（グループ化しない）"] + meta_cols,
            key="case_sentiment_group"
        )
        if group_col_input != "（グループ化しない）":
            group_col = group_col_input
    else:
        st.info("属性列が読み込まれていません。属性列を含むCSV/Excelを読み込むとグループ集計が利用できます。")

    # ============================================================
    # ④ 実行ボタン
    # ============================================================
    if not st.button("▶ ケース別感情分析を実行", key="run_case_sentiment"):
        return

    with st.spinner("ケースごとに感情分析中..."):
        results = []

        # ---- asari 方式 ----
        if method == "asari（AIモデル）":
            try:
                from asari.api import Sonar
                sonar = Sonar()
            except Exception as e:
                st.error(f"asariの起動に失敗しました: {e}")
                return

            for idx, row in df_meta.iterrows():
                case_text = str(row[selected_text_col]) if pd.notna(row[selected_text_col]) else ""
                if not case_text.strip():
                    continue

                # 文に分割して1文ずつ判定
                sentences = [s.strip() + "。" for s in
                             case_text.replace("\n", "。").split("。") if s.strip()]
                pos_scores, neg_scores, trend_scores = [], [], []

                for sentence in sentences:
                    if len(sentence) <= 1:
                        continue
                    try:
                        res = sonar.ping(sentence)
                        pos_p = next(c["confidence"] for c in res["classes"] if c["class_name"] == "positive")
                        neg_p = next(c["confidence"] for c in res["classes"] if c["class_name"] == "negative")
                        pos_scores.append(pos_p)
                        neg_scores.append(neg_p)
                        trend_scores.append(pos_p - neg_p)
                    except:
                        continue

                if not trend_scores:
                    continue

                avg_trend = sum(trend_scores) / len(trend_scores)
                avg_pos   = sum(pos_scores)   / len(pos_scores)
                avg_neg   = sum(neg_scores)   / len(neg_scores)

                if abs(avg_trend) < 0.2:
                    overall = "😐 ニュートラル"
                elif avg_trend > 0:
                    overall = "😊 ポジティブ"
                else:
                    overall = "😞 ネガティブ"

                record = {
                    "ケースNo": idx + 1,
                    "テキスト（先頭50文字）": case_text[:50],
                    "総合判定": overall,
                    "平均ポジティブ率": round(avg_pos, 3),
                    "平均ネガティブ率": round(avg_neg, 3),
                    "平均推移スコア": round(avg_trend, 3),
                }
                for col in meta_cols:
                    record[col] = row[col]
                results.append(record)

            score_col = "平均推移スコア"

        # ---- 東北大学評価極性辞書 方式 ----
        else:
            import os
            import unicodedata
            from janome.tokenizer import Tokenizer as JanomeTokenizer

            dic_path = os.path.join("dic", "pn.csv.m3.120408.trim")
            if not os.path.exists(dic_path):
                st.error(f"辞書ファイルが見つかりません: {dic_path}")
                st.caption("「dic」フォルダに「pn.csv.m3.120408.trim」を配置してください。")
                return

            df_pn = pd.read_csv(dic_path, sep="\t", names=["term", "sentiment", "category"])
            # 全角→半角の正規化
            df_pn["term"] = [unicodedata.normalize("NFKC", t) for t in df_pn["term"]]
            pn_dict = {}
            for _, r in df_pn.iterrows():
                s = r["sentiment"]
                if s == "p":
                    pn_dict[r["term"]] = 1.0
                elif s == "n":
                    pn_dict[r["term"]] = -1.0
                elif s == "e":
                    pn_dict[r["term"]] = 0.0

            t = JanomeTokenizer()

            for idx, row in df_meta.iterrows():
                case_text = str(row[selected_text_col]) if pd.notna(row[selected_text_col]) else ""
                if not case_text.strip():
                    continue

                tokens = list(t.tokenize(case_text))
                scores = [pn_dict[unicodedata.normalize("NFKC", tk.base_form)]
                          for tk in tokens
                          if unicodedata.normalize("NFKC", tk.base_form) in pn_dict]

                if not scores:
                    continue

                avg_score = sum(scores) / len(scores)
                pos_count = scores.count(1.0)
                neg_count = scores.count(-1.0)
                total     = len(scores)

                if abs(avg_score) < 0.05:
                    overall = "😐 ニュートラル"
                elif avg_score > 0:
                    overall = "😊 ポジティブ"
                else:
                    overall = "😞 ネガティブ"

                record = {
                    "ケースNo": idx + 1,
                    "テキスト（先頭50文字）": case_text[:50],
                    "総合判定": overall,
                    "平均スコア": round(avg_score, 3),
                    "ポジティブ語数": pos_count,
                    "ネガティブ語数": neg_count,
                    "判定語数合計": total,
                }
                for col in meta_cols:
                    record[col] = row[col]
                results.append(record)

            score_col = "平均スコア"

    # ============================================================
    # ⑤ 結果が空の場合
    # ============================================================
    if not results:
        st.warning("判定できるケースがありませんでした。テキスト列や辞書ファイルを確認してください。")
        return

    df_result_case = pd.DataFrame(results)

    # ============================================================
    # ⑥ ケース別一覧の表示
    # ============================================================
    st.markdown("---")
    st.write(f"#### 📋 ケース別 感情スコア一覧（{len(df_result_case)}件）")
    st.dataframe(df_result_case, use_container_width=True)

    # ============================================================
    # ⑦ グループ別：記述統計 ＋ 確認用棒グラフ
    # ============================================================
    df_group_stats = None

    if group_col and group_col in df_result_case.columns:

        st.markdown("---")
        st.write(f"#### 📊 「{group_col}」別 感情スコアの記述統計")

        # --- 記述統計 ---
        df_group_stats = (
            df_result_case
            .groupby(group_col)[score_col]
            .describe()
            .round(3)
            .reset_index()
        )
        df_group_stats.columns = [
            group_col, "件数", "平均値", "標準偏差",
            "最小値", "25%ile", "中央値", "75%ile", "最大値"
        ]
        st.dataframe(df_group_stats, use_container_width=True)

        # --- 確認用棒グラフ（matplotlib：シンプル）---
        st.write(f"#### 📊 「{group_col}」別 平均感情スコア（確認用）")

        df_bar = df_result_case.groupby(group_col)[score_col].mean().reset_index()
        df_bar.columns = [group_col, "平均感情スコア"]

        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        colors = ["#6699ff" if v >= 0 else "#ff6666" for v in df_bar["平均感情スコア"]]
        ax_bar.bar(df_bar[group_col].astype(str), df_bar["平均感情スコア"], color=colors)
        ax_bar.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax_bar.set_xlabel(group_col)
        ax_bar.set_ylabel("平均感情スコア")
        ax_bar.set_title(f"{group_col} 別 平均感情スコア（確認用）")
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["top"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    # ============================================================
    # ⑧ 統計的検定（ノンパラメトリック検定・初期設定）
    # ============================================================
    if group_col and group_col in df_result_case.columns:

        from scipy import stats

        st.markdown("---")
        st.write(f"#### 🔬 統計的検定：「{group_col}」別 感情スコアの比較")

        # ---- 検定方式の選択（ノンパラメトリックをデフォルトに）----
        test_type = st.radio(
            "検定方式を選択してください",
            ["ノンパラメトリック検定（推奨）", "パラメトリック検定"],
            index=0,  # ← ノンパラメトリックがデフォルト
            horizontal=True,
            key="test_type_select"
        )
        st.caption(
            "💡 **ノンパラメトリック検定**は正規分布を前提としないため、"
            "社会福祉学領域の自由記述データ分析に適しています。"
        )

        groups       = df_result_case.groupby(group_col)[score_col].apply(list)
        group_names  = list(groups.index)
        group_values = list(groups.values)
        n_groups     = len(group_names)

        df_test_result = None  # 初期化

        if n_groups < 2:
            st.info("グループが1つのみのため、検定はできません。")

        # ============================================================
        # ノンパラメトリック検定
        # ============================================================
        elif test_type == "ノンパラメトリック検定（推奨）":

            if n_groups == 2:
                # ---- Mann-Whitney U 検定（2グループ）----
                st.write("##### 📌 Mann-Whitney U 検定（2グループ・ノンパラメトリック）")

                g1, g2   = group_values[0], group_values[1]
                u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")

                # 効果量 r = Z / √N の計算
                import numpy as np
                n_total = len(g1) + len(g2)
                # 正規近似によるZ値
                z_val = stats.norm.ppf(1 - u_p / 2)
                effect_r = z_val / np.sqrt(n_total)

                st.markdown(f"""
| 項目 | 値 |
|---|---|
| グループ1（{group_names[0]}） | n={len(g1)}、中央値={round(sorted(g1)[len(g1)//2], 3)} |
| グループ2（{group_names[1]}） | n={len(g2)}、中央値={round(sorted(g2)[len(g2)//2], 3)} |
| U統計量 | {round(u_stat, 3)} |
| p値 | **{round(u_p, 4)}** |
| 効果量 r | {round(effect_r, 3)}（目安：小=0.1 / 中=0.3 / 大=0.5） |
| 判定 | {"✅ 有意差あり（p < 0.05）" if u_p < 0.05 else "❌ 有意差なし（p ≥ 0.05）"} |
""")

                if u_p < 0.05:
                    st.success(
                        f"「{group_names[0]}」と「{group_names[1]}」の感情スコアには"
                        f"統計的に有意な差があります（p = {round(u_p, 4)}）。"
                    )
                else:
                    st.info(
                        f"「{group_names[0]}」と「{group_names[1]}」の感情スコアに"
                        f"有意差は認められませんでした（p = {round(u_p, 4)}）。"
                    )

                df_test_result = pd.DataFrame([{
                    "検定手法": "Mann-Whitney U検定",
                    "グループ1": group_names[0],
                    "グループ2": group_names[1],
                    "U統計量": round(u_stat, 3),
                    "p値": round(u_p, 4),
                    "効果量 r": round(effect_r, 3),
                    "有意差（p<0.05）": "あり" if u_p < 0.05 else "なし"
                }])

            else:
                # ---- Kruskal-Wallis 検定（3グループ以上）----
                st.write(f"##### 📌 Kruskal-Wallis 検定（{n_groups}グループ・ノンパラメトリック）")

                h_stat, kw_p = stats.kruskal(*group_values)

                # グループ別の記述統計（中央値ベース）
                import numpy as np
                kw_summary = []
                for name, vals in zip(group_names, group_values):
                    kw_summary.append({
                        "グループ": name,
                        "n": len(vals),
                        "中央値": round(float(np.median(vals)), 3),
                        "平均順位（参考）": None  # 後で埋める
                    })

                # 平均順位の計算
                all_vals    = df_result_case[score_col].values
                all_groups  = df_result_case[group_col].values
                ranks       = stats.rankdata(all_vals)
                for i, name in enumerate(group_names):
                    mask = all_groups == name
                    kw_summary[i]["平均順位（参考）"] = round(float(ranks[mask].mean()), 3)

                st.dataframe(pd.DataFrame(kw_summary), use_container_width=True)

                st.markdown(f"""
| 項目 | 値 |
|---|---|
| H統計量 | {round(h_stat, 3)} |
| p値 | **{round(kw_p, 4)}** |
| 判定 | {"✅ グループ間に有意差あり（p < 0.05）" if kw_p < 0.05 else "❌ グループ間に有意差なし（p ≥ 0.05）"} |
""")

                if kw_p < 0.05:
                    st.success(
                        f"グループ間の感情スコアに統計的に有意な差があります"
                        f"（p = {round(kw_p, 4)}）。"
                    )

                    # ---- Dunn 検定（多重比較・Bonferroni補正）----
                    st.write("##### 📌 多重比較：Dunn検定（Bonferroni補正）")
                    try:
                        from itertools import combinations
                        pair_results = []
                        pairs = list(combinations(range(n_groups), 2))
                        n_pairs = len(pairs)

                        for i, j in pairs:
                            g_i = group_values[i]
                            g_j = group_values[j]
                            # Dunn検定の近似（Mann-Whitney U + Bonferroni補正）
                            _, raw_p = stats.mannwhitneyu(g_i, g_j, alternative="two-sided")
                            adjusted_p = min(raw_p * n_pairs, 1.0)  # Bonferroni補正

                            pair_results.append({
                                "グループ1": group_names[i],
                                "グループ2": group_names[j],
                                "p値（補正前）": round(raw_p, 4),
                                "p値（Bonferroni補正後）": round(adjusted_p, 4),
                                "有意差（p<0.05）": "✅ あり" if adjusted_p < 0.05 else "❌ なし"
                            })

                        df_dunn = pd.DataFrame(pair_results)
                        st.dataframe(df_dunn, use_container_width=True)
                        st.caption(
                            "Bonferroni補正済みp値 < 0.05 の組み合わせが有意差ありのペアです。"
                        )

                        df_test_result = pd.concat([
                            pd.DataFrame([{
                                "検定手法": "Kruskal-Wallis検定",
                                "グループ数": n_groups,
                                "H統計量": round(h_stat, 3),
                                "p値": round(kw_p, 4),
                                "有意差（p<0.05）": "あり"
                            }]),
                            df_dunn
                        ], ignore_index=True)

                    except Exception as e:
                        st.warning(f"多重比較の計算中にエラーが発生しました: {e}")
                        df_test_result = pd.DataFrame([{
                            "検定手法": "Kruskal-Wallis検定",
                            "グループ数": n_groups,
                            "H統計量": round(h_stat, 3),
                            "p値": round(kw_p, 4),
                            "有意差（p<0.05）": "あり"
                        }])
                else:
                    st.info(
                        f"グループ間の感情スコアに有意差は認められませんでした"
                        f"（p = {round(kw_p, 4)}）。"
                    )
                    df_test_result = pd.DataFrame([{
                        "検定手法": "Kruskal-Wallis検定",
                        "グループ数": n_groups,
                        "H統計量": round(h_stat, 3),
                        "p値": round(kw_p, 4),
                        "有意差（p<0.05）": "なし"
                    }])

        # ============================================================
        # パラメトリック検定（選択時のみ）
        # ============================================================
        else:
            if n_groups == 2:
                st.write("##### 📌 対応なしt検定（2グループ）")

                g1, g2 = group_values[0], group_values[1]
                lev_stat, lev_p = stats.levene(g1, g2)
                equal_var = lev_p >= 0.05
                t_stat, t_p = stats.ttest_ind(g1, g2, equal_var=equal_var)

                st.markdown(f"""
| 項目 | 値 |
|---|---|
| グループ1（{group_names[0]}） | n={len(g1)}、平均={round(sum(g1)/len(g1),3)} |
| グループ2（{group_names[1]}） | n={len(g2)}、平均={round(sum(g2)/len(g2),3)} |
| Levene検定（等分散） | F={round(lev_stat,3)}、p={round(lev_p,3)} → {"等分散（Student法）" if equal_var else "不等分散（Welch法）"} |
| t統計量 | {round(t_stat, 3)} |
| p値 | **{round(t_p, 4)}** |
| 判定 | {"✅ 有意差あり（p < 0.05）" if t_p < 0.05 else "❌ 有意差なし（p ≥ 0.05）"} |
""")

                if t_p < 0.05:
                    st.success(f"有意差あり（p = {round(t_p, 4)}）")
                else:
                    st.info(f"有意差なし（p = {round(t_p, 4)}）")

                df_test_result = pd.DataFrame([{
                    "検定手法": "対応なしt検定（Welch法）" if not equal_var else "対応なしt検定（Student法）",
                    "t統計量": round(t_stat, 3),
                    "p値": round(t_p, 4),
                    "有意差（p<0.05）": "あり" if t_p < 0.05 else "なし"
                }])

            else:
                st.write(f"##### 📌 一元配置ANOVA（{n_groups}グループ）")

                f_stat, anova_p = stats.f_oneway(*group_values)

                st.markdown(f"""
| 項目 | 値 |
|---|---|
| F統計量 | {round(f_stat, 3)} |
| p値 | **{round(anova_p, 4)}** |
| 判定 | {"✅ 有意差あり（p < 0.05）" if anova_p < 0.05 else "❌ 有意差なし（p ≥ 0.05）"} |
""")

                if anova_p < 0.05:
                    st.success(f"グループ間に有意差あり（p = {round(anova_p, 4)}）")
                    try:
                        from statsmodels.stats.multicomp import pairwise_tukeyhsd
                        tukey_result = pairwise_tukeyhsd(
                            df_result_case[score_col].values,
                            df_result_case[group_col].values,
                            alpha=0.05
                        )
                        df_tukey = pd.DataFrame(
                            data=tukey_result._results_table.data[1:],
                            columns=tukey_result._results_table.data[0]
                        )
                        st.write("##### 📌 多重比較（Tukey HSD）")
                        st.dataframe(df_tukey, use_container_width=True)
                        st.caption("reject=True の組み合わせが有意差ありのペアです。")
                    except ImportError:
                        st.warning("Tukey HSDには statsmodels が必要です。`pip install statsmodels` を実行してください。")
                else:
                    st.info(f"グループ間に有意差なし（p = {round(anova_p, 4)}）")

                df_test_result = pd.DataFrame([{
                    "検定手法": "一元配置ANOVA",
                    "F統計量": round(f_stat, 3),
                    "p値": round(anova_p, 4),
                    "有意差（p<0.05）": "あり" if anova_p < 0.05 else "なし"
                }])

    else:
        df_test_result = None


    # ============================================================
    # ⑨ Excel 3シート 一括ダウンロード
    # ============================================================
    st.markdown("---")
    st.write("#### 📥 Excelで一括ダウンロード")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # シート1：ケース別
        df_result_case.to_excel(writer, index=False, sheet_name="ケース別感情分析")

        # シート2：グループ別集計 or 全体サマリー
        if df_group_stats is not None:
            df_group_stats.to_excel(writer, index=False, sheet_name="グループ別集計")
        else:
            summary_df = df_result_case[[score_col]].describe().round(3).reset_index()
            summary_df.columns = ["統計量", "値"]
            summary_df.to_excel(writer, index=False, sheet_name="全体サマリー")

        # シート3：検定結果（グループ化した場合のみ）
        if df_test_result is not None:
            df_test_result.to_excel(writer, index=False, sheet_name="統計的検定結果")

    st.download_button(
        label="📥 結果をExcelで一括ダウンロード（ケース別 ＋ グループ別 ＋ 検定結果）",
        data=buffer.getvalue(),
        file_name=f"case_sentiment_{selected_text_col}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_case_sentiment_excel"
    )

def draw_cluster_analysis(text, df_result, target_pos, synonym_dict, stopwords):
    st.write("### 🌳/📍 クラスター分析（単語のグループ化）")
    st.write("単語同士が「どのくらい同じ文脈で使われているか」を計算し、グループ化します。")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        unit_option = st.selectbox(
            "分析単位（テキストの区切り方）",
            ["段落単位（改行）", "一文単位（句点）", "ファイル全体", "単語単位（10語区切り）"]
        )
    with col2:
        top_n = st.slider("分析に含める上位単語数", min_value=10, max_value=100, value=30, step=5)
    with col3:
        plot_type = st.radio("グラフの種類", ["樹形図（階層型）", "散布図（K-means）"])
        
    if plot_type == "散布図（K-means）":
        n_clusters = st.number_input("散布図のクラスター（グループ）数", min_value=2, max_value=10, value=3)

    with st.spinner("テキストを分割し、クラスターを計算中..."):
        # 1. 分析単位の分割と単語抽出
        t = Tokenizer()
        docs_words = []
        
        # 単語単位（10語区切り）の場合は、先に全単語を抽出してチャンク化する
        if unit_option == "単語単位（10語区切り）":
            all_valid_words = []
            for token in t.tokenize(text):
                pos = token.part_of_speech.split(',')[0]
                base_form = token.base_form if token.base_form != '*' else token.surface
                if pos in target_pos:
                    base_form = synonym_dict.get(base_form, base_form)
                    if base_form not in stopwords:
                        all_valid_words.append(base_form)
            # 10語ずつに分割
            for i in range(0, len(all_valid_words), 10):
                chunk = all_valid_words[i:i+10]
                if len(chunk) > 0:
                    docs_words.append(" ".join(chunk))
        else:
            # それ以外の分割単位
            if unit_option == "ファイル全体":
                docs = [text]
            elif unit_option == "段落単位（改行）":
                docs = [p.strip() for p in text.split('\n') if len(p.strip()) > 0]
            elif unit_option == "一文単位（句点）":
                docs = [s.strip() + '。' for s in text.replace('\n', '。').split('。') if len(s.strip()) > 0]
                
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
            st.warning("分割された文書が少なすぎます。別の分析単位を選択してください。")
            return

        # 2. 上位単語の絞り込みとベクトル化（出現回数行列の作成）
        top_words = df_result['語句'].head(top_n).tolist()
        vectorizer = CountVectorizer(vocabulary=top_words)
        X = vectorizer.fit_transform(docs_words).toarray()
        
        # 単語×文書の行列に転置
        X_T = X.T
        valid_indices = X_T.sum(axis=1) > 0
        X_T_valid = X_T[valid_indices]
        valid_words = [top_words[i] for i, valid in enumerate(valid_indices) if valid]

        if len(valid_words) < n_clusters if plot_type == "散布図（K-means）" else len(valid_words) < 2:
            st.warning("有効な単語数が不足しています。抽出条件を見直してください。")
            return

        # 3. グラフの描画
        fig_cluster, ax_cluster = plt.subplots(figsize=(10, 6))

        if plot_type == "樹形図（階層型）":
            # 階層的クラスタリング（ウォード法）
            Z = linkage(X_T_valid, method='ward', metric='euclidean')
            dendrogram(Z, labels=valid_words, orientation='right', ax=ax_cluster, leaf_font_size=12)
            ax_cluster.set_title(f"単語の樹形図（{unit_option} / 上位{top_n}語）", fontsize=14)
            ax_cluster.set_xlabel("距離（非類似度）")
            
        else:
            # 非階層的クラスタリング（K-means法）と主成分分析（PCA）
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_T_valid)
            
            # 多次元のデータを2次元に圧縮（マッピング）
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_T_valid)
            
            # 散布図の描画
            scatter = ax_cluster.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap='tab10', alpha=0.7, s=150)
            
            # 単語ラベルの付与
            for i, word in enumerate(valid_words):
                ax_cluster.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=11, alpha=0.9, 
                                    xytext=(5, 5), textcoords='offset points')
                
            ax_cluster.set_title(f"単語のクラスター散布図（PCA + K-means / {n_clusters}グループ）", fontsize=14)
            ax_cluster.set_xlabel("第1主成分 (PC1)")
            ax_cluster.set_ylabel("第2主成分 (PC2)")
            ax_cluster.grid(True, linestyle='--', alpha=0.5)

        st.pyplot(fig_cluster)
        
        # 4. 画像保存機能
        buf_cluster = io.BytesIO()
        fig_cluster.savefig(buf_cluster, format="png", dpi=300, bbox_inches='tight')
        st.download_button("🖼️ グラフをPNGで保存", data=buf_cluster.getvalue(), file_name=f"cluster_{'dendrogram' if plot_type == '樹形図（階層型）' else 'scatter'}.png", mime="image/png")

def draw_kwic(text, df_result):
    st.write("### 🔍 文脈抽出（KWIC）")
    st.write("特定の単語が、テキストの中でどのような文脈（前後関係）で使われているかを確認できます。")
    top_words = df_result['語句'].head(100).tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        selected_word = st.selectbox("頻出語から選択してください（トップ100）:", options=["(直接入力する)"] + top_words)
    with col2:
        custom_word = st.text_input("自由に検索したい単語を入力（任意）:")

    target_word = custom_word if custom_word else (selected_word if selected_word != "(直接入力する)" else "")

    if target_word:
        sentences = [s.strip() + '。' for s in text.replace('\n', '。').split('。') if s.strip()]
        matched_sentences = [s for s in sentences if target_word in s]
        
        if matched_sentences:
            st.success(f"「**{target_word}**」を含む文が **{len(matched_sentences)}件** 見つかりました。")
            html_content = ""
            for i, sentence in enumerate(matched_sentences, 1):
                highlighted_text = sentence.replace(
                    target_word, 
                    f"<mark style='background-color: #ffeb3b; font-weight: bold; color: black; padding: 0 4px; border-radius: 3px;'>{target_word}</mark>"
                )
                html_content += f"<div style='padding: 8px; border-bottom: 1px solid #ddd; line-height: 1.6;'><b>{i}.</b> {highlighted_text}</div>"
            
            st.markdown(
                f"<div style='max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #fafafa;'>{html_content}</div>", 
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
            
            df_kwic = pd.DataFrame({"No.": range(1, len(matched_sentences) + 1), "文脈": matched_sentences})
            csv = df_kwic.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 抽出結果をCSVでダウンロード", data=csv, file_name=f"kwic_{target_word}.csv", mime="text/csv")
        else:
            st.warning(f"「{target_word}」を含む文は見つかりませんでした。")
    else:
        st.info("👆 確認したい単語を選択するか、入力してください。")

def draw_crosstab_and_ca(df, text_col, meta_cols, target_pos, synonym_dict, stopwords):
    st.write("### 🔀 属性別のクロス集計・コレスポンデンス分析")
    
    if not meta_cols:
        st.info("ファイル読み込み時に属性データ（メタデータ）の列が選択されていません。")
        return
        
    col1, col2 = st.columns(2)
    with col1:
        selected_meta = st.selectbox("比較したい属性を選んでください:", meta_cols)
    with col2:
        top_n = st.slider("集計に含める上位単語数:", 10, 100, 30, step=10, key="crosstab_slider")
    
    with st.spinner("クロス集計表とマップを作成中..."):
        t = Tokenizer()
        all_rows_words = []
        
        for idx, row in df.iterrows():
            text = str(row[text_col])
            if pd.isna(text) or text.strip() == "":
                all_rows_words.append([])
                continue
                
            words = []
            for token in t.tokenize(text):
                pos = token.part_of_speech.split(',')[0]
                base_form = token.base_form if token.base_form != '*' else token.surface
                
                if pos in target_pos:
                    base_form = synonym_dict.get(base_form, base_form)
                    if base_form not in stopwords:
                        words.append(base_form)
            all_rows_words.append(words)
            
        all_words_flat = [w for sublist in all_rows_words for w in sublist]
        if not all_words_flat:
            st.warning("単語が抽出できませんでした。")
            return
            
        top_words = [w for w, c in Counter(all_words_flat).most_common(top_n)]
        
        crosstab_data = []
        meta_values = df[selected_meta].dropna().unique()
        
        for word in top_words:
            row_data = {"単語": word}
            for meta_value in meta_values:
                indices = df[df[selected_meta] == meta_value].index
                meta_words = [w for i in indices if i < len(all_rows_words) for w in all_rows_words[i]]
                row_data[str(meta_value)] = meta_words.count(word)
            crosstab_data.append(row_data)
            
        df_crosstab = pd.DataFrame(crosstab_data).set_index("単語")
        
        st.write(f"#### 📌 「{selected_meta}」別の頻出単語 {top_n}語 の集計表")
        st.dataframe(df_crosstab, use_container_width=True)
        
        csv_cross = df_crosstab.to_csv().encode('utf-8-sig')
        st.download_button("📥 クロス集計表をCSVでダウンロード", csv_cross, f"crosstab_{selected_meta}.csv", "text/csv")
        
        st.markdown("---")
        st.write("#### 🗺️ コレスポンデンス分析（対応分析）マップ")
        st.write("属性（赤色の▲）と単語（青色の●）の位置関係をマップ化します。近くにある属性と単語ほど、関連性が強い（よく一緒に使われる）ことを示します。")

        try:
            import numpy as np
            from sklearn.decomposition import TruncatedSVD

            X = df_crosstab.values.astype(float)
            row_sums = X.sum(axis=1)
            col_sums = X.sum(axis=0)
            
            valid_rows = row_sums > 0
            valid_cols = col_sums > 0
            
            X = X[valid_rows][:, valid_cols]
            words_labels = df_crosstab.index[valid_rows].tolist()
            meta_labels = df_crosstab.columns[valid_cols].tolist()

            total = X.sum()
            P = X / total
            r = P.sum(axis=1)
            c = P.sum(axis=0)
            
            E = np.outer(r, c)
            Z = (P - E) / np.sqrt(E)
            
            svd = TruncatedSVD(n_components=2, random_state=42)
            svd.fit(Z)
            
            row_coords = svd.transform(Z) / np.sqrt(r[:, np.newaxis])
            col_coords = svd.components_.T * svd.singular_values_ / np.sqrt(c[:, np.newaxis])
            
            fig_ca, ax_ca = plt.subplots(figsize=(10, 8))
            
            ax_ca.scatter(row_coords[:, 0], row_coords[:, 1], c='#4A90E2', alpha=0.5, marker='o', s=50)
            for i, txt in enumerate(words_labels):
                ax_ca.annotate(txt, (row_coords[i, 0], row_coords[i, 1]), color='#333333', fontsize=11, ha='center', va='bottom')
                
            ax_ca.scatter(col_coords[:, 0], col_coords[:, 1], c='#E94A66', marker='^', s=200, edgecolors='white', linewidth=1.5, zorder=5)
            for i, txt in enumerate(meta_labels):
                ax_ca.annotate(txt, (col_coords[i, 0], col_coords[i, 1]), color='#E94A66', fontsize=15, fontweight='bold', ha='center', va='bottom')
                
            ax_ca.axhline(0, color='gray', linestyle='--', alpha=0.3)
            ax_ca.axvline(0, color='gray', linestyle='--', alpha=0.3)
            ax_ca.set_title(f"「{selected_meta}」と頻出単語の関連性マップ", fontsize=14)
            
            st.pyplot(fig_ca)
            
            buf_ca = io.BytesIO()
            fig_ca.savefig(buf_ca, format="png", dpi=300, bbox_inches='tight')
            st.download_button("🖼️ マップをPNGで保存", data=buf_ca.getvalue(), file_name=f"correspondence_map_{selected_meta}.png", mime="image/png")
            
        except Exception as e:
            st.error(f"コレスポンデンス分析の計算中にエラーが発生しました: {e}")

def analyze_metadata(df, meta_cols):
    st.write("### 👥 属性データの分布")
    st.write("選択された属性データ（年齢、性別、スコアなど）の偏りや傾向を確認します。")
    
    if not meta_cols:
        st.info("属性データが選択されていません。")
        return
        
    for col in meta_cols:
        st.markdown(f"#### 📌 {col} の分布")
        valid_data = df[col].dropna()
        if valid_data.empty:
            st.warning(f"「{col}」には有効なデータがありません。")
            continue
            
        col1, col2 = st.columns([1, 1.2])
        if pd.api.types.is_numeric_dtype(valid_data):
            with col1:
                st.write("**【記述統計】**")
                st.dataframe(valid_data.describe().to_frame(name="統計量"), use_container_width=True)
            with col2:
                st.write("**【ヒストグラム】**")
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(valid_data, bins=10, color='#4A90E2', edgecolor='black', alpha=0.7)
                ax.set_ylabel("人数 / 件数")
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                st.pyplot(fig)
        else:
            with col1:
                st.write("**【度数分布】**")
                st.dataframe(valid_data.value_counts().to_frame(name="件数"), use_container_width=True)
            with col2:
                st.write("**【棒グラフ】**")
                fig, ax = plt.subplots(figsize=(5, 3))
                value_counts_head = valid_data.value_counts().head(10) 
                ax.barh(value_counts_head.index[::-1], value_counts_head.values[::-1], color='#50E3C2')
                ax.set_xlabel("人数 / 件数")
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                st.pyplot(fig)
        st.markdown("---")

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
        st.dataframe(df_pos, use_container_width=True)
    with col_chart:
        fig = px.pie(df_pos, values='出現数', names='品詞', title='品詞別の出現数割合', hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# メイン画面のUIと処理
# ==========================================
st.set_page_config(page_title="計量テキスト分析ツール", layout="wide")
st.title("計量テキスト分析ツール")

st.markdown("""
    <style>
        /* タブのリスト部分を折り返し可能にする */
        div[data-testid="stTabs"] > div[role="tablist"] {
            flex-wrap: wrap;
            gap: 5px;
        }
        /* タブ自体の高さを調整して見やすくする */
        div[data-testid="stTabs"] [data-baseweb="tab"] {
            height: auto;
            padding-top: 8px;
            padding-bottom: 8px;
            margin-bottom: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# --- サイドバー：ページ切り替えナビゲーション ---
with st.sidebar:
    st.title("ナビゲーション")
    page_selection = st.radio("画面を選択してください", ["📊 分析ツール本体", "📖 使い方・機能紹介"])
    st.markdown("---")

# ==========================================
# 📖 使い方・機能紹介ページ
# ==========================================
if page_selection == "📖 使い方・機能紹介":
    st.title("📖 使い方・機能紹介")
    st.markdown("""
    ### このアプリについて
    このツールは、テキストデータを簡単に計量テキスト分析するためのアプリです。
    
    ### 分析の進め方
    1. **ファイルの読み込み**: 左側のメニューから、分析したいファイル（txt, csv, pdfなど）をアップロードします。
    2. **品詞の選択**: 抽出したい品詞（名詞、動詞など）を選びます。
    3. **分析実行**: 条件を設定すると、自動的に抽出とグラフ化が行われます。
    
    ### 各機能の紹介
    * **共起ネットワーク**: 一緒に使われやすい単語同士を線で結んだ図です。
    * **感情分析**: 文脈がポジティブかネガティブかをAIが判定します。
    * *(※ここに操作方法やライセンス情報などを自由に追記してください)*
    """)

# ==========================================
# 📊 分析ツール本体（これまでのコード）
# ==========================================
elif page_selection == "📊 分析ツール本体":
    # 既存のCSS設定
    st.markdown("""
                Ver.20260324
                         """) 
    # （※これ以降に、既存の with st.sidebar: やアップロード処理、分析処理をすべて入れ込みます。インデントに注意してください）


with st.sidebar:
    # --- サイドバーに終了ボタンを追加 ---
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 アプリを終了する"):
        st.session_state['confirm_exit'] = True

    # 終了確認メッセージ
    if st.session_state.get('confirm_exit', False):
        st.sidebar.warning("⚠️ この分析結果をこのまま閉じても良いですか？保存していないデータは消去されます。")
        col_yes, col_no = st.sidebar.columns(2)
    
        if col_yes.button("はい、終了します"):
            st.sidebar.success("アプリを終了しています... このタブを閉じてください。")
            # 自分自身のプロセス（コマンドプロンプト含む）を強制終了する
            os.kill(os.getpid(), signal.SIGTERM)
        
        if col_no.button("キャンセル"):
            st.session_state['confirm_exit'] = False
            st.rerun()

    st.header("1. 抽出する品詞の選択")
    target_pos = st.multiselect("集計対象とする品詞を選んでください", ["名詞", "動詞", "形容詞", "副詞"], default=["名詞", "動詞", "形容詞"])
    
    st.markdown("---")
    st.header("2. 複合語（1語として扱う語）の定義")
    option = st.radio(
        "処理方法の選択",
        ("4. 連続する名詞を自動結合する（ルールベース）", "1. 画面上で直接、語句定義を入力する", "2. ユーザーが作成した定義ファイルを読み込む", "3. 生成AI用のプロンプトを作成する")
    )
    
    custom_words_text = ""
    custom_dict_file = None
    if option == "1. 画面上で直接、語句定義を入力する":
        custom_words_text = st.text_area("1語として抽出したい単語を入力 (改行区切り)", "社会福祉\n防災教育")
        st.download_button("📥 入力した語句定義をファイルとして保存", data=custom_words_text, file_name="custom_words.txt")
    elif option == "2. ユーザーが作成した定義ファイルを読み込む":
        custom_dict_file = st.file_uploader("抽出語の定義ファイルを選択 (.txt または .csv)", type=['txt', 'csv'], key="extract_file")
        
    st.markdown("---")
    st.header("3. 除外設定（ストップワード）")
    st.write("集計結果から除外したい単語を定義します。")
    
    default_stopwords = "する\nある\nいる\nなる\nできる\nこれ\nそれ\nあれ"
    stop_words_text = st.text_area("除外する単語を入力 (改行区切り)", default_stopwords)
    stop_words_file = st.file_uploader("除外語の追加ファイルを選択 (.txt または .csv)", type=['txt', 'csv'], key="stop_file")

    st.markdown("---")
    st.header("4. プロジェクトの復元（読み込み）")
    st.write("過去に保存した分析データを読み込みます。")
    uploaded_project = st.file_uploader("📂 プロジェクトファイル（.pkl）を選択", type=['pkl'])

uploaded_file = st.file_uploader("新規に分析するテキストファイルをアップロード (txt, csv, xlsx, docx, pdf)", type=['txt', 'csv', 'xlsx', 'docx', 'pdf'])
synonym_file = st.file_uploader("同義語・ゆらぎ統一辞書をアップロード（任意, .csv）", type=['csv'])

synonym_dict = load_synonym_dict(synonym_file)

# --- プロジェクトの読み込み処理 ---
if uploaded_project is not None:
    try:
        project_data = pickle.load(uploaded_project)
        for key, value in project_data.items():
            st.session_state[key] = value
        st.sidebar.success("✅ プロジェクトを復元しました！")
    except Exception as e:
        st.error(f"プロジェクトファイルの読み込みに失敗しました: {e}")

# --- 新規ファイルの読み込み処理 ---
elif uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    # 変更点：違うファイルがアップロードされた時だけ、初期化処理を行う
    if st.session_state.get('current_uploaded_file') != uploaded_file.name:
        st.session_state['current_uploaded_file'] = uploaded_file.name
        st.session_state['file_name'] = uploaded_file.name
        st.session_state['text_ready'] = False
        
        # CSV/Excel以外（txt, pdf, docx等）は、アップロード直後にテキストを抽出
        if file_ext not in ['csv', 'xlsx']:
            text = extract_text(uploaded_file)
            st.session_state['text_ready'] = True
            st.session_state['extracted_text'] = text
            st.session_state['df_meta'] = None
            st.session_state['meta_cols'] = []
            st.session_state['text_col'] = None

    # CSV/Excelの場合の設定UI
    if file_ext in ['csv', 'xlsx']:
        if file_ext == 'csv':
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)
            
        st.write("### ⚙️ 読み込み設定（表データ）")
        
        # UIが再実行されても選択状態を保持できるよう、適宜keyを付与
        text_col = st.selectbox("📝 分析するテキスト（自由記述など）の列を選んでください", df_input.columns)
        meta_cols = st.multiselect("👤 属性データ（年代・性別・スコアなど）の列を選んでください（任意）", [col for col in df_input.columns if col != text_col])
        
        if st.button("この設定でテキストを抽出して分析を開始する"):
            text = "\n".join(df_input[text_col].dropna().astype(str).tolist())
            st.session_state['df_meta'] = df_input[[text_col] + meta_cols].copy()
            st.session_state['meta_cols'] = meta_cols
            st.session_state['text_col'] = text_col
            st.session_state['text_ready'] = True
            st.session_state['extracted_text'] = text
    else:
        text = extract_text(uploaded_file)
        st.session_state['text_ready'] = True
        st.session_state['extracted_text'] = text
        st.session_state['df_meta'] = None
        st.session_state['meta_cols'] = []
        st.session_state['text_col'] = None

# --- 分析処理と画面表示 ---
if st.session_state.get('text_ready', False):
    text = st.session_state.get('extracted_text', "")
    current_filename = st.session_state.get('file_name', '復元されたデータ')

    with st.sidebar:
        st.markdown("---")
        st.header("💾 現在のプロジェクトを保存")
        st.write("抽出したテキストや属性データをファイルに保存します。")
        project_data = {
            'text_ready': True,
            'extracted_text': text,
            'df_meta': st.session_state.get('df_meta'),
            'meta_cols': st.session_state.get('meta_cols'),
            'text_col': st.session_state.get('text_col'),
            'file_name': current_filename
        }
        st.download_button(
            label="📦 プロジェクトを保存（.pkl）",
            data=pickle.dumps(project_data),
            file_name=f"project_{current_filename}.pkl",
            mime="application/octet-stream"
        )

    if text.strip() == "":
        st.warning("テキストが抽出できませんでした。")
    else:
        if option == "3. 生成AI用のプロンプトを作成する":
            st.write("### 🤖 ChatGPT等への入力用プロンプト")
            sample_text = text[:2000] + ("...\n(以下略)" if len(text) > 2000 else "")
            prompt = f"以下のテキストデータから、計量テキスト分析において「1語として扱うべき複合名詞や専門用語」を抽出してください。\n出力形式は、抽出した単語のみを改行で区切ったプレーンテキストにしてください。\n\n【テキストデータ】\n{sample_text}\n"
            st.code(prompt, language="text")
        else:
            if not target_pos:
                st.warning("左側のサイドバーで、抽出する品詞を1つ以上選択してください。")
            else:
                st.info(f"「{current_filename}」の解析を実行中...")
                
                with st.spinner("⏳ テキストを読み込み、形態素解析を行っています..."):
                    custom_dict_content = custom_dict_file.read().decode('utf-8') if custom_dict_file else None
                    stop_words_content = stop_words_file.read().decode('utf-8') if stop_words_file else None
                    
                    df_result, sentences_words = analyze_text(text, option, custom_words_text, custom_dict_content, stop_words_text, stop_words_content, target_pos, synonym_dict)
                
                if df_result.empty:
                    st.warning("抽出された単語がありません。")
                else:
                    with st.spinner("📊 グラフやネットワーク図を生成しています..."):
                        
                        # ==========================================
                        # 1. 基本項目の表示
                        # ==========================================
                        st.markdown("---")
                        st.header("1. 基本項目")
                        tab_meta, tab_cross, tab0, tab1, tab2 = st.tabs([
                            "👥 属性データ", "🔀 クロス集計", "📊 記述統計", "📋 データ表", "📈 出現頻度"
                        ])

                        with tab_meta:
                            if st.session_state.get('df_meta') is not None and st.session_state.get('meta_cols'):
                                analyze_metadata(st.session_state['df_meta'], st.session_state['meta_cols'])
                            else:
                                st.info("ExcelやCSVファイルから属性データが読み込まれた場合、ここに分布が表示されます。")

                        with tab_cross:
                            if st.session_state.get('df_meta') is not None and st.session_state.get('meta_cols'):
                                stopwords = set([w.strip() for w in stop_words_text.replace('\n', ',').split(',') if w.strip()])
                                if stop_words_content:
                                    stopwords.update([w.strip() for w in stop_words_content.replace('\n', ',').split(',') if w.strip()])
                                draw_crosstab_and_ca(st.session_state['df_meta'], st.session_state['text_col'], st.session_state['meta_cols'], target_pos, synonym_dict, stopwords)
                            else:
                                st.info("属性データを含むファイル（Excel/CSV）を読み込むと利用できます。")

                        with tab0:
                            draw_descriptive_stats(text)

                        with tab1:
                            st.write("### 抽出された単語のデータ")
                            st.dataframe(df_result.head(50))
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                df_result.to_excel(writer, index=False, sheet_name='頻出語集計')
                            st.download_button("📊 Excelファイルをダウンロード", data=buffer.getvalue(), file_name="word_frequency.xlsx")

                        with tab2:
                            draw_frequency_chart(df_result)


                        # ==========================================
                        # 2. 応用項目の表示
                        # ==========================================
                        st.markdown("---")
                        st.header("2. 応用項目")
                        tab8, tab3, tab4, tab5, tab6, tab7, tab_cluster = st.tabs([
                            "🔗 N-gram", "☁️ ワードクラウド", "🕸️ 共起ネットワーク", "🌟 TF-IDF", "😊 感情分析", "🔍 KWIC", "🌳 クラスター分析"
                        ])

                        with tab8:
                            ngram_top_n = st.slider("表示するバイグラム数", min_value=10, max_value=100, value=20, step=10, key="ngram_slider")
                            draw_ngram(sentences_words, ngram_top_n)
                                
                        with tab3:
                            draw_wordcloud(df_result)
                            
                        with tab4:
                            draw_cooccurrence_network(df_result, sentences_words)
                        
                        with tab5:
                            draw_tfidf_chart(sentences_words)
                        
                        with tab6:
                            # ── 既存：テキスト全体の感情分析 ──
                            draw_sentiment_analysis(text)

                            # ── 追加：ケース別感情分析（属性データ読込済みの場合のみ表示）──
                            st.markdown("---")
                            if st.session_state.get("df_meta") is not None:
                                draw_sentiment_by_case(
                                    st.session_state["df_meta"],
                                    st.session_state["text_col"],
                                    st.session_state.get("meta_cols", [])
                                )
                            else:
                                st.info("💡 ケース別感情分析は、属性データ付きCSV/Excelを読み込むと利用できます。")
                            
                        with tab7:
                            draw_kwic(text, df_result)
                        
                        with tab_cluster:
                            # stopwordsの設定（他のタブと同じように取得）
                            stopwords = set([w.strip() for w in stop_words_text.replace('\n', ',').split(',') if w.strip()])
                            if stop_words_content:
                                stopwords.update([w.strip() for w in stop_words_content.replace('\n', ',').split(',') if w.strip()])
                            
                            draw_cluster_analysis(text, df_result, target_pos, synonym_dict, stopwords)    
                        
                        # ==========================================
                        # 3. AI分析（ローカルLLM連携）
                        # ==========================================
                        st.markdown("---")
                        st.header("3. AI分析（ローカルLLM）")
                        
                        # ★タブを2つに増やします
                        tab_ai, tab_after_coding = st.tabs(["🤖 テキスト要約・分析", "✨ AIアフターコーディング（辞書作成）"])
                        
                        # --- 既存の要約機能 ---
                        with tab_ai:
                            st.write("### LM Studioを使ったローカルAI要約")
                            st.write("※あらかじめLM Studioで Local Server (ポート1234) を起動しておいてください。")
                            
                            ai_prompt = st.text_area(
                                "AIへの指示（プロンプト）:", 
                                value="以下のテキストを読み込み、主要なトピックを3つに分けて要約してください。", 
                                height=100
                            )
                            
                            if st.button("AIで要約を実行する"):
                                with st.spinner("AIが考え中...（ローカルマシンの性能により時間がかかります）"):
                                    try:
                                        from openai import OpenAI
                                        
                                        # LM Studioのローカルサーバーへ接続
                                        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
                                        
                                        # テキストが長すぎる場合は先頭を切り取る（メモリ上限対策）
                                        max_length = 3000 
                                        input_text = text[:max_length] + ("\n...(以下省略)" if len(text) > max_length else "")
                                        
                                        response = client.chat.completions.create(
                                            model="local-model",
                                            messages=[
                                                {"role": "system", "content": "あなたは優秀なデータアナリストです。"},
                                                {"role": "user", "content": f"{ai_prompt}\n\n【テキストデータ】\n{input_text}"}
                                            ],
                                            temperature=0.7,
                                        )
                                        
                                        # 結果を取得し、セッションステート（保管庫）に保存する
                                        st.session_state['summary_result'] = response.choices[0].message.content
                                        st.success("出力完了！")
                                        
                                    except Exception as e:
                                        st.error(f"AIとの通信に失敗しました。LM Studioでサーバーが起動しているか確認してください。\nエラー詳細: {e}")

                            # 保管庫にデータがあれば表示＆ダウンロード可能にする
                            if 'summary_result' in st.session_state:
                                st.markdown(f"> {st.session_state['summary_result']}")
                                
                                # ダウンロード用のテキストデータ
                                summary_text_for_dl = st.session_state['summary_result'].encode('utf-8-sig')
                                st.download_button(
                                    label="📥 要約結果をテキストで保存", 
                                    data=summary_text_for_dl, 
                                    file_name="ai_summary_result.txt", 
                                    mime="text/plain"
                                )

                        # --- 新規追加：AIアフターコーディング機能 ---
                        with tab_after_coding:
                            st.write("### 抽出語の自動グルーピング（辞書作成支援）")
                            st.write("頻出単語の上位リストをAIに読み込ませ、同義語や関連語をまとめるための「ゆらぎ統一辞書」のベースを自動作成します。")
                            
                            # ★変更点：ユーザーが抽出する単語数を指定できる入力ボックスを追加
                            top_n = st.number_input(
                                "AIに渡す上位単語の数（10〜500語）:", 
                                min_value=10, 
                                max_value=500, 
                                value=100, 
                                step=10
                            )
                            
                            # 分析結果からユーザーが指定した上位N語を抽出してカンマ区切りの文字列にする
                            top_words_list = df_result.head(top_n)["語句"].tolist() if "語句" in df_result.columns else df_result.head(top_n).iloc[:, 0].tolist()
                            words_text = ", ".join(top_words_list)
                            
                            st.info(f"**AIに渡す対象単語（出現頻度上位{top_n}語）:**\n{words_text}")
                            
                            # アフターコーディング用の固定プロンプト
                            coding_prompt = """
あなたは優秀なデータアナリストです。以下の「対象単語リスト」の中に含まれる単語から、表記揺れや同義語を見つけ出し、名寄せ（統一）のための辞書を作成してください。

【厳守する出力ルール】
1. 必ず「元の単語,統一後の代表語」の【2列のみ】のCSV形式で出力してください。
2. 1行につきカンマは1つだけです。3つ以上の単語をカンマで繋いではいけません。
3. 複数の単語を同じ代表語に統一したい場合は、必ず行を分けてください。
4. ヘッダー（見出し行）や説明文は一切出力しないでください。
5. 統一の必要がない単語は出力しないでください。

【良い出力例】（このように必ず行を分けて2列で出力する）
いける,行く
行ける,行く
通う,行く
自動車,車
クルマ,車

【悪い出力例】（このように1行に3つ以上並べるのは絶対にNG）
行く,いける,行ける,通う
車,自動車,クルマ
"""

                            
                            if st.button("AIで辞書を作成する"):
                                with st.spinner("AIが単語を分類中...（しばらくお待ちください）"):
                                    try:
                                        from openai import OpenAI
                                        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
                                        
                                        response = client.chat.completions.create(
                                            model="local-model",
                                            messages=[
                                                {"role": "system", "content": "あなたは優秀なデータアナリストです。指示されたフォーマットのみを出力します。"},
                                                {"role": "user", "content": f"{coding_prompt}\n\n【対象単語リスト】\n{words_text}"}
                                            ],
                                            temperature=0.3, # 創造性より正確性を重視して温度を下げる
                                        )
                                        
                                        st.session_state['after_coding_result'] = response.choices[0].message.content
                                        st.success("辞書の作成が完了しました！")
                                        
                                    except Exception as e:
                                        st.error(f"AIとの通信に失敗しました。エラー詳細: {e}")

                            # 結果の表示とダウンロードボタン
                            if 'after_coding_result' in st.session_state:
                                st.write("#### AIの作成結果")
                                # AIの出力をそのまま表示（コードブロックでCSVっぽく見せる）
                                st.code(st.session_state['after_coding_result'], language="csv")
                                
                                # Excelでも文字化けしないように utf-8-sig でエンコード
                                csv_data = st.session_state['after_coding_result'].encode('utf-8-sig')
                                st.download_button(
                                    label="📥 ゆらぎ統一辞書（CSV）としてダウンロード", 
                                    data=csv_data, 
                                    file_name="ai_synonym_dict.csv", 
                                    mime="text/csv"
                                )
                                st.caption("※ダウンロードしたCSVファイルは、左側のメニュー「同義語・ゆらぎ統一辞書をアップロード」からそのまま読み込ませて分析に利用できます。必要に応じてExcel等で開いて微調整してください。")
