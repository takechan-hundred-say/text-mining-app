import io
import streamlit as st
import pandas as pd
from janome.tokenizer import Tokenizer
from collections import Counter, defaultdict
from datetime import datetime


def draw_sankey_diagram(stage_config, stage_texts, df_heatmap):
    import plotly.graph_objects as go
    st.write("### 📊 サンキーダイアグラム（カテゴリの流れ）")
    stages = list(df_heatmap.columns)
    categories = list(df_heatmap.index)
    node_labels = []; node_colors = []
    color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA15E', '#BC6C25', '#8E7DBE', '#A8DADC', '#457B9D']
    for stage in stages:
        for cat in categories:
            node_labels.append(f"{cat}<br>({stage})"); color_idx = categories.index(cat) % len(color_palette); node_colors.append(color_palette[color_idx])
    source, target, value = [], [], []
    for i in range(len(stages) - 1):
        current_stage, next_stage = stages[i], stages[i+1]
        for cat in categories:
            current_freq = df_heatmap.loc[cat, current_stage]; next_freq = df_heatmap.loc[cat, next_stage]
            if current_freq > 0 and next_freq > 0:
                source.append(node_labels.index(f"{cat}<br>({current_stage})")); target.append(node_labels.index(f"{cat}<br>({next_stage})")); value.append(max(current_freq, next_freq))
    fig = go.Figure(data=[go.Sankey(node=dict(label=node_labels, color=node_colors, pad=15, line=dict(color='white', width=2)), link=dict(source=source, target=target, value=value, color=['rgba(200,200,200,0.4)']*len(source)))])
    fig.update_layout(title="段階別カテゴリの流れ", font=dict(size=10), height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True); return fig


def create_heatmap_data(stage_config, stage_texts, categories):
    stages = [config[1]["name"] for config in sorted(stage_config.items(), key=lambda x: x[0])]
    heatmap_data = {stage: {} for stage in stages}
    tokenizer = Tokenizer()
    for stage_name, text in stage_texts.items():
        if not text.strip():
            for cat in categories: heatmap_data[stage_name][cat] = 0; continue
        analyzed_words = []
        for token in tokenizer.tokenize(text):
            pos = token.part_of_speech.split(',')[0]; base_form = token.base_form if token.base_form != '*' else token.surface
            if pos == '名詞': analyzed_words.append(base_form)
        category_counts = {cat: 0 for cat in categories}
        for word in analyzed_words:
            if word in categories: category_counts[word] += 1
        for cat in categories: heatmap_data[stage_name][cat] = category_counts[cat]
    df_heatmap = pd.DataFrame(heatmap_data).fillna(0).astype(int)
    return df_heatmap


def process_flow_ui():
    st.write("### 🔄 プロセスフロー分析")
    st.info("📌 段階数と各段階のテキストを設定し、カテゴリを定義してサンキーダイアグラム＆バブルチャートを生成します。")
    st.markdown("#### ステップ1：段階数を選択")
    num_stages = st.slider("段階数を選択", min_value=2, max_value=10, value=3, step=1, key="pf_num_stages")
    st.markdown("#### ステップ2：各段階の設定")
    if st.session_state.get('df_meta') is not None:
        available_cols = st.session_state['df_meta'].columns.tolist(); has_meta = True
    else:
        available_cols = []; has_meta = False
    stage_config = {}; stage_texts = {}; col_stages = st.columns(num_stages)
    for i in range(num_stages):
        with col_stages[i]:
            st.subheader(f"段階{i+1}")
            stage_name = st.text_input(f"段階{i+1}の名前", value=["初期","中盤","終盤"][i] if i<3 else f"段階{i+1}", key=f"pf_stage_name_{i}")
            if has_meta and available_cols:
                selected_col = st.selectbox(f"データ列を選択", available_cols, key=f"pf_stage_col_{i}")
                stage_config[i] = {"name": stage_name, "column": selected_col}
                combined_text = " ".join(st.session_state['df_meta'][selected_col].dropna().astype(str).tolist())
                stage_texts[stage_name] = combined_text; st.caption(f"✓ {len(combined_text.split())} 語を検出")
            else:
                text_input = st.text_area(f"{stage_name}のテキスト", height=100, key=f"pf_stage_text_{i}")
                stage_config[i] = {"name": stage_name, "column": None}; stage_texts[stage_name] = text_input
    st.markdown("#### ステップ3：カテゴリ抽出方法")
    category_method = st.radio("カテゴリの定義方法を選択", ("手動入力","形態素解析で自動抽出","AI提案で確認"), key="pf_category_method")
    categories = []
    if 'pf_auto_extracted_categories' not in st.session_state: st.session_state['pf_auto_extracted_categories'] = []
    if category_method == "手動入力":
        categories_text = st.text_input("例：課題, 不安, 期待, 成果, 自信", key="pf_categories_manual")
        categories = [cat.strip() for cat in categories_text.split(',') if cat.strip()]
    elif category_method == "形態素解析で自動抽出":
        st.info("💡 各段階のテキストから形態素解析で頻出語TOPを自動抽出します")
        top_n_extract = st.slider("各段階から抽出する上位語数", min_value=3, max_value=10, value=5, step=1, key="pf_auto_extract_top_n")
        if st.button("自動抽出を実行", key="pf_auto_extract"):
            try:
                all_extracted_words = set()
                for stage_name, text in stage_texts.items():
                    if not text.strip(): continue
                    tokenizer = Tokenizer(); words = []
                    for token in tokenizer.tokenize(text):
                        pos = token.part_of_speech.split(',')[0]; base_form = token.base_form if token.base_form != '*' else token.surface
                        if pos in ['名詞','動詞','形容詞'] and len(base_form)>1: words.append(base_form)
                    word_freq = Counter(words); top_words = [word for word,count in word_freq.most_common(top_n_extract)]
                    all_extracted_words.update(top_words); st.caption(f"✓ [{stage_name}]から {len(top_words)} 語を抽出: {', '.join(top_words)}")
                extracted_list = sorted(list(all_extracted_words))
                st.session_state['pf_auto_extracted_categories'] = extracted_list
                st.success(f"✅ {len(extracted_list)} 個のカテゴリを自動抽出しました"); st.write(f"**抽出されたカテゴリ：** {', '.join(extracted_list)}")
            except Exception as e: st.error(f"❌ 自動抽出中にエラー: {e}")
        categories = st.session_state.get('pf_auto_extracted_categories', [])
    else:
        st.info("💡 LLMが特徴的なカテゴリを提案します（実装予定）")
    st.markdown("#### ステップ4：可視化設定")
    col_color, col_norm = st.columns(2)
    with col_color: color_mode = st.radio("色分けモード", ("カテゴリ別固定色","グレースケール（モノクロ対応）"), key="pf_color_mode")
    with col_norm: normalization_mode = st.radio("正規化方式", ("段階ごと独立（推奨）","全体最大値基準"), key="pf_normalization")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1: generate_sankey = st.button("🎨 サンキーを生成", key="pf_gen_sankey")
    with col_btn2: generate_bubble = st.button("🔥 バブルチャートを生成", key="pf_gen_bubble")
    with col_btn3: st.write("")
    return {"num_stages": num_stages, "stage_config": stage_config, "stage_texts": stage_texts, "categories": categories, "color_mode": color_mode, "normalization_mode": normalization_mode, "generate_sankey": generate_sankey, "generate_bubble": generate_bubble}


def draw_bubble_heatmap(df_heatmap, categories, color_mode, normalization_mode):
    import plotly.graph_objects as go; import numpy as np
    st.write("### 🔥 バブルチャート型ヒートマップ")
    if normalization_mode == "段階ごと独立（推奨）":
        df_normalized = df_heatmap.copy()
        for col in df_normalized.columns:
            col_max = df_normalized[col].max()
            if col_max > 0: df_normalized[col] = (df_normalized[col] / col_max) * 100
            else: df_normalized[col] = 0
    else:
        max_global = df_heatmap.values.max()
        if max_global > 0: df_normalized = (df_heatmap / max_global) * 100
        else: df_normalized = df_heatmap.copy()
    df_normalized = df_normalized.fillna(0).astype(float)
    if color_mode == "カテゴリ別固定色":
        color_palette = {"課題":"#FF6B6B","不安":"#4ECDC4","期待":"#45B7D1","成果":"#96CEB4","自信":"#FFEAA7"}
        colors = [color_palette.get(cat,"#999999") for cat in categories]
    else: colors = []
    fig = go.Figure()
    for cat_idx, category in enumerate(df_normalized.index):
        x_coords = list(df_normalized.columns); y_coords = [category] * len(df_normalized.columns)
        bubble_sizes = []
        for stage in df_normalized.columns:
            value = df_normalized.loc[category, stage]
            if pd.isna(value) or np.isnan(value): bubble_sizes.append(0)
            else: bubble_sizes.append(max(0, float(value)*0.8))
        if color_mode == "グレースケール（モノクロ対応）":
            bubble_colors = []
            for freq in bubble_sizes:
                if freq==0: bubble_colors.append('rgb(200,200,200)')
                else: gray_rgb = int(255*(1-freq/100)); bubble_colors.append(f'rgb({gray_rgb},{gray_rgb},{gray_rgb})')
        else: bubble_colors = [colors[cat_idx]] * len(x_coords)
        safe_sizes = [max(1,s*1.5) if s>0 else 1 for s in bubble_sizes]
        fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='markers', marker=dict(size=safe_sizes, color=bubble_colors, opacity=0.7, line=dict(width=2,color='white')), name=category, text=[f"{category}<br>出現率: {freq:.1f}%" for freq in bubble_sizes], hovertemplate='<b>%{text}</b><extra></extra>'))
    fig.update_layout(title="段階別カテゴリ出現頻度（バブルチャート）", xaxis_title="段階", yaxis_title="カテゴリ", height=500, hovermode='closest', showlegend=True)
    st.plotly_chart(fig, use_container_width=True); return fig, df_heatmap, df_normalized


def count_words_with_morphology(text, synonym_dict=None, stopwords=None, target_pos=None, analyzer_choice="Janome（標準・推奨）"):
    if synonym_dict is None: synonym_dict = {}
    if stopwords is None: stopwords = set()
    if target_pos is None: target_pos = ["名詞","動詞","形容詞"]
    if not text or not text.strip(): return Counter()
    tokenizer = Tokenizer(); words = []
    for token in tokenizer.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        if pos not in target_pos: continue
        base_form = token.base_form if token.base_form != '*' else token.surface
        base_form = synonym_dict.get(base_form, base_form)
        if base_form in stopwords: continue
        if len(base_form) < 2: continue
        words.append(base_form)
    return Counter(words)


def split_text_into_stages(text, num_stages):
    if not text or not text.strip(): return [""]*num_stages
    sentences = text.replace('\n','。').split('。')
    sentences = [s.strip()+'。' for s in sentences if s.strip()]
    if not sentences: return [""]*num_stages
    stage_size = len(sentences)//num_stages; stages_text = []
    for i in range(num_stages):
        start_idx = i*stage_size; end_idx = (i+1)*stage_size if i<num_stages-1 else len(sentences)
        stages_text.append(' '.join(sentences[start_idx:end_idx]))
    return stages_text


def draw_word_transition_line(df_transition, stage_names):
    import plotly.express as px
    df_long = df_transition.melt(id_vars="語句", value_vars=stage_names, var_name="段階", value_name="頻度")
    fig = px.line(df_long, x="段階", y="頻度", color="語句", markers=True, title="語句の段階別推移", labels={"段階":"分析段階","頻度":"出現回数","語句":"単語"}, hover_data={"段階":True,"頻度":True,"語句":True})
    fig.update_layout(height=500, hovermode='x unified', template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)


def draw_word_transition_bubble(df_transition, stage_names):
    import plotly.graph_objects as go
    colors = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7','#DDA15E','#BC6C25','#8E7DBE','#A8DADC','#457B9D']
    fig = go.Figure()
    for idx, (_, row) in enumerate(df_transition.iterrows()):
        word = row['語句']; x_data = stage_names; y_data = [word]*len(stage_names)
        sizes = [row[stage]*10 for stage in stage_names]; color = colors[idx%len(colors)]
        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', marker=dict(size=sizes, color=color, opacity=0.6, line=dict(width=2,color='white')), name=word, text=[f"{word}<br>出現数: {row[stage]}" for stage in stage_names], hovertemplate='<b>%{text}</b><extra></extra>'))
    fig.update_layout(title="語句推移バブルチャート", xaxis_title="段階", yaxis_title="語句", height=500, hovermode='closest', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def label_word_state(freq, high_threshold=3):
    if freq == 0: return "非出現"
    elif freq < high_threshold: return "低頻度"
    else: return "高頻度"


def build_word_transition_by_stages(text, num_stages, top_n_words, synonym_dict=None, stopwords=None, target_pos=None, stage_names_custom=None):
    if synonym_dict is None: synonym_dict = {}
    if stopwords is None: stopwords = set()
    if target_pos is None: target_pos = ["名詞"]
    stages_text_list = split_text_into_stages(text, num_stages)
    if stage_names_custom is None:
        stage_names = [f"段階{i+1}" for i in range(num_stages)]
    else:
        stage_names = stage_names_custom[:num_stages]
    stage_counts = {}
    for stage_name, stage_text in zip(stage_names, stages_text_list):
        stage_counts[stage_name] = count_words_with_morphology(stage_text, synonym_dict=synonym_dict, stopwords=stopwords, target_pos=target_pos)
    total_counter = Counter()
    for counts in stage_counts.values(): total_counter.update(counts)
    top_words = [w for w, _ in total_counter.most_common(top_n_words)]
    df_transition = pd.DataFrame({"語句": top_words})
    for stage_name in stage_names:
        df_transition[stage_name] = df_transition["語句"].apply(lambda w: stage_counts[stage_name].get(w, 0))
    return df_transition, stage_names


def draw_word_transition_sankey(df):
    import plotly.graph_objects as go
    stages = ["初期", "中盤", "終盤"]
    links = []
    for _, row in df.iterrows():
        for i in range(len(stages) - 1):
            source = f"{stages[i]} : {label_word_state(row[stages[i]])}"
            target = f"{stages[i+1]} : {label_word_state(row[stages[i+1]])}"
            links.append((source, target))
    link_counts = Counter(links)
    nodes = list(set([s for s, _ in link_counts] + [t for _, t in link_counts]))
    node_index = {node: i for i, node in enumerate(nodes)}
    source_idx, target_idx, values = [], [], []
    for (s, t), v in link_counts.items():
        source_idx.append(node_index[s])
        target_idx.append(node_index[t])
        values.append(v)
    fig = go.Figure(go.Sankey(node=dict(label=nodes, pad=20, thickness=15), link=dict(source=source_idx, target=target_idx, value=values)))
    fig.update_layout(title="語句推移ケースフロー（サンキーダイアグラム）", font=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)


def export_flow_analysis_to_excel(df_transition, stage_names, analysis_params):
    from datetime import datetime
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_transition.to_excel(writer, sheet_name='語句×段階', index=False)
        params_data = {"パラメータ": ["分析日時", "段階数", "上位語句数", "段階名", "テキスト総字数"], "値": [analysis_params.get("分析日時", datetime.now()), analysis_params.get("段階数", len(stage_names)), analysis_params.get("上位語句数", len(df_transition)), " → ".join(stage_names), analysis_params.get("テキスト総字数", 0)]}
        pd.DataFrame(params_data).to_excel(writer, sheet_name='分析パラメータ', index=False)
        stats_data = {"語句": df_transition["語句"], "合計出現数": df_transition[stage_names].sum(axis=1), "平均出現数": df_transition[stage_names].mean(axis=1).round(1), "最高出現数": df_transition[stage_names].max(axis=1), "最低出現数": df_transition[stage_names].min(axis=1)}
        pd.DataFrame(stats_data).to_excel(writer, sheet_name='統計情報', index=False)
    buffer.seek(0)
    return buffer.getvalue()
