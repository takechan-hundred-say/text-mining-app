import streamlit as st
import pandas as pd
import pickle
import signal
import os
import sys
import io
from janome.tokenizer import Tokenizer
from collections import Counter
from datetime import datetime

from app.config import get_project_root, find_mecab_path, find_r_path
from app.utils.text_io import load_synonym_dict, extract_text, create_zip_data
from app.utils.tokenize import create_user_dict_file, analyze_text
from app.analysis.stats import draw_descriptive_stats
from app.analysis.ngram_tfidf import draw_ngram, draw_tfidf_chart
from app.analysis.wordcloud import draw_wordcloud
from app.analysis.network import draw_cooccurrence_network
from app.analysis.sentiment import draw_sentiment_analysis, draw_sentiment_by_case
from app.analysis.cluster_kwic import draw_cluster_analysis, draw_kwic
from app.analysis.crosstab_ca import draw_crosstab_and_ca, draw_ca_with_r, analyze_metadata
from app.analysis.process_flow import (
    draw_sankey_diagram, create_heatmap_data, process_flow_ui, draw_bubble_heatmap,
    count_words_with_morphology, split_text_into_stages, build_word_transition_by_stages,
    draw_word_transition_line, draw_word_transition_bubble, draw_word_transition_sankey,
    label_word_state, export_flow_analysis_to_excel
)
from app.r_bridge.r_runner import run_r_script, draw_frequency_chart_r

st.set_page_config(page_title="計量テキスト分析ツール", layout="wide")

if 'MECAB_PATH' not in st.session_state:
    st.session_state.MECAB_PATH = find_mecab_path()
if 'R_PATH' not in st.session_state:
    st.session_state.R_PATH = find_r_path()

st.title("計量テキスト分析ツール")

st.markdown("""
    <style>
        div[data-testid="stTabs"] > div[role="tablist"] {
            flex-wrap: wrap;
            gap: 5px;
        }
        div[data-testid="stTabs"] [data-baseweb="tab"] {
            height: auto;
            padding-top: 8px;
            padding-bottom: 8px;
            margin-bottom: 5px;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("ナビゲーション")
    page_selection = st.radio("画面を選択してください", ["📊 分析ツール本体", "📖 使い方・機能紹介"])
    st.markdown("---")

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
    """)

elif page_selection == "📊 分析ツール本体":
    st.markdown("""Ver.1.0(2026-06-25)""")

    with st.sidebar:
        with st.expander("⚙️ 環境確認", expanded=False):
            st.write("**インストール状況**")
            if st.session_state.MECAB_PATH:
                st.success(f"✅ MeCab: {st.session_state.MECAB_PATH}")
            else:
                st.warning("⚠️ MeCab: 見つかりません")
            if st.session_state.R_PATH:
                st.success(f"✅ R: {st.session_state.R_PATH}")
            else:
                st.warning("⚠️ R: 見つかりません")

        st.markdown("---")

        st.header("0. 形態素解析エンジン")
        analyzer_choice = st.radio(
            "使用するエンジンを選択",
            ("Janome（標準・推奨）", "MeCab + UniDic（高精度）"),
            help="MeCabはインストール済みの場合のみ選択できます。"
        )

        if analyzer_choice == "MeCab + UniDic（高精度）":
            if st.session_state.MECAB_PATH:
                try:
                    os.add_dll_directory(st.session_state.MECAB_PATH)
                    import MeCab
                    st.success("✅ MeCab 接続確認済み")
                except Exception as e:
                    st.error(f"❌ MeCab が使用できません: {e}")
                    analyzer_choice = "Janome（標準・推奨）"
            else:
                st.error("❌ MeCabがインストールされていません")
                st.info("https://taku910.github.io/mecab/#download")
                analyzer_choice = "Janome（標準・推奨）"

        st.markdown("---")
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

        st.markdown("---")

        if st.sidebar.button("🚪 アプリを終了する"):
            st.session_state['confirm_exit'] = True

        if st.session_state.get('confirm_exit', False):
            st.sidebar.warning("⚠️ この分析結果をこのまま閉じても良いですか？保存していないデータは消去されます。")
            col_yes, col_no = st.sidebar.columns(2)
            if col_yes.button("はい、終了します"):
                st.sidebar.success("アプリを終了しています... このタブを閉じてください。")
                os.kill(os.getpid(), signal.SIGTERM)
            if col_no.button("キャンセル"):
                st.session_state['confirm_exit'] = False
                st.rerun()

    uploaded_file = st.file_uploader("新規に分析するテキストファイルをアップロード (txt, csv, xlsx, docx, pdf)", type=['txt', 'csv', 'xlsx', 'docx', 'pdf'])
    synonym_file = st.file_uploader("同義語・ゆらぎ統一辞書をアップロード（任意, .csv）", type=['csv'])

    synonym_dict = load_synonym_dict(synonym_file)

    if uploaded_project is not None:
        try:
            project_data = pickle.load(uploaded_project)
            for key, value in project_data.items():
                st.session_state[key] = value
            st.sidebar.success("✅ プロジェクトを復元しました！")
        except Exception as e:
            st.error(f"プロジェクトファイルの読み込みに失敗しました: {e}")

    elif uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()

        if st.session_state.get('current_uploaded_file') != uploaded_file.name:
            st.session_state['current_uploaded_file'] = uploaded_file.name
            st.session_state['file_name'] = uploaded_file.name
            st.session_state['text_ready'] = False

            if file_ext not in ['csv', 'xlsx']:
                text = extract_text(uploaded_file)
                st.session_state['text_ready'] = True
                st.session_state['extracted_text'] = text
                st.session_state['df_meta'] = None
                st.session_state['meta_cols'] = []
                st.session_state['text_col'] = None

        if file_ext in ['csv', 'xlsx']:
            if file_ext == 'csv':
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)

            st.write("### ⚙️ 読み込み設定（表データ）")
            st.info("📋 CSVファイルを読み込みました。以下で分析に用いる列を選択してください。")

            col1, col2 = st.columns(2)
            with col1:
                text_col = st.selectbox("📝 **分析するテキスト（自由記述など）の列を選んでください**", df_input.columns, key="text_col_select")
            with col2:
                meta_cols = st.multiselect("👤 **属性データ（年代・性別など）の列を選んでください（任意）**", [col for col in df_input.columns if col != text_col], key="meta_cols_select")

            if st.button("▶ この設定で分析を開始する", key="start_analysis_csv"):
                if text_col:
                    text = "\n".join(df_input[text_col].dropna().astype(str).tolist())
                    if meta_cols:
                        st.session_state['df_meta'] = df_input[[text_col] + meta_cols].copy()
                        st.session_state['meta_cols'] = meta_cols
                    else:
                        st.session_state['df_meta'] = df_input[[text_col]].copy()
                        st.session_state['meta_cols'] = []
                    st.session_state['text_col'] = text_col
                    st.session_state['text_ready'] = True
                    st.session_state['extracted_text'] = text
                    st.success(f"✅ 設定完了！テキスト列: **{text_col}**、属性列: **{meta_cols if meta_cols else '(なし)'}**")
                    st.rerun()
                else:
                    st.error("❌ テキスト列を選択してください")
        else:
            if file_ext not in ['csv', 'xlsx']:
                text = extract_text(uploaded_file)
                st.session_state['text_ready'] = True
                st.session_state['extracted_text'] = text
                st.session_state['df_meta'] = None
                st.session_state['meta_cols'] = []
                st.session_state['text_col'] = None

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

                        df_result, sentences_words = analyze_text(
                            text, option, custom_words_text, custom_dict_content,
                            stop_words_text, stop_words_content, target_pos, synonym_dict,
                            analyzer_choice
                        )

                    if df_result.empty:
                        st.warning("抽出された単語がありません。")
                    else:
                        with st.spinner("📊 グラフやネットワーク図を生成しています..."):
                            st.markdown("---")
                            st.header("1. 基本項目")
                            tab0, tab1, tab2, tab_meta, tab_cross = st.tabs([
                                "📊 記述統計", "📋 データ表", "📈 出現頻度", "👥 属性データ", "🔀 クロス集計"
                            ])

                            with tab0:
                                draw_descriptive_stats(text)

                            with tab1:
                                st.markdown("### 📋 抽出された単語のデータ")
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    display_rows = st.slider("表示する語句の数を選択", min_value=50, max_value=200, value=50, step=50, key="display_rows_slider")
                                st.dataframe(df_result.head(display_rows), use_container_width=True)
                                st.markdown("---")
                                st.markdown("#### 📥 全データをダウンロード")
                                download_format = st.radio("ダウンロード形式を選択", ("Excel形式 (.xlsx)", "CSV形式 (.csv)"), horizontal=True, key="download_format_radio")
                                if download_format == "Excel形式 (.xlsx)":
                                    buffer = io.BytesIO()
                                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                        df_result.to_excel(writer, index=False, sheet_name='頻出語集計')
                                    st.download_button("📊 全データをExcelでダウンロード", data=buffer.getvalue(), file_name=f"word_frequency_all_{current_filename}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_excel_btn")
                                    st.info("💡 **全ての語句が含まれています。** 画面表示の上限に制限されません。")
                                else:
                                    csv_data = df_result.to_csv(index=False, encoding='utf-8-sig')
                                    st.download_button("📥 全データをCSVでダウンロード", data=csv_data, file_name=f"word_frequency_all_{current_filename}.csv", mime="text/csv", key="download_csv_btn")
                                    st.info("💡 **全ての語句が含まれています。** 日本語も正しく保存されます。")
                                st.markdown("---")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("総語句数", len(df_result))
                                with col2:
                                    st.metric("表示中の語句数", len(df_result.head(display_rows)))
                                with col3:
                                    st.metric("最頻出語の出現回数", df_result.iloc[0, 1] if len(df_result) > 0 else 0)

                            with tab2:
                                draw_frequency_chart_r(df_result, top_n=30)

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

                            st.markdown("---")
                            st.header("2. 応用項目")
                            tab8, tab3, tab4, tab5, tab6, tab7, tab_cluster, tab_pf = st.tabs([
                                "🔗 N-gram", "☁️ ワードクラウド", "🕸️ 共起ネットワーク",
                                "🌟 TF-IDF", "😊 感情分析", "🔍 KWIC", "🌳 クラスター分析", "🔄 プロセスフロー"
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
                                draw_sentiment_analysis(text)
                                st.markdown("---")
                                if st.session_state.get("df_meta") is not None:
                                    draw_sentiment_by_case(st.session_state["df_meta"], st.session_state["text_col"], st.session_state.get("meta_cols", []))
                                else:
                                    st.info("💡 ケース別感情分析は、属性データ付きCSV/Excelを読み込むと利用できます。")

                            with tab7:
                                draw_kwic(text, df_result, synonym_dict=synonym_dict)

                            with tab_cluster:
                                stopwords = set([w.strip() for w in stop_words_text.replace('\n', ',').split(',') if w.strip()])
                                if stop_words_content:
                                    stopwords.update([w.strip() for w in stop_words_content.replace('\n', ',').split(',') if w.strip()])
                                draw_cluster_analysis(text, df_result, target_pos, synonym_dict, stopwords)

                            EMOTION_COGNITIVE_WORDS = [
                                "怖い", "こわい", "不安", "楽しい", "嬉しい", "つらい", "辛い",
                                "悲しい", "寂しい", "安心", "嫌", "きつい",
                                "わかる", "分かる", "学ぶ", "学び", "気づく", "考える",
                                "感じる", "実感", "理解"
                            ]

                            with tab_pf:
                                st.subheader("🔄 語句頻出推移分析")
                                st.write("段階ごとのテキストから、語句の出現推移を分析します。")

                                st.markdown("#### ⚙️ ステップ1：段階数を設定")
                                num_stages = st.slider("分析する段階数を選択", min_value=2, max_value=10, value=3, step=1, key="flow_num_stages_v2")
                                st.info(f"💡 元のテキストを{num_stages}段階に自動分割します")

                                st.markdown("#### ⚙️ ステップ2：段階名を設定（任意）")
                                use_custom_names = st.checkbox("段階名をカスタマイズする（チェックなしはデフォルト名を使用）", value=False, key="flow_custom_names")
                                default_stage_names = ["初期段階", "中盤段階", "終盤段階", "発展段階", "成果段階"]

                                if use_custom_names:
                                    stage_names = []
                                    st.write("**各段階の名前を入力してください：**")
                                    col_names = st.columns(num_stages)
                                    for i in range(num_stages):
                                        with col_names[i]:
                                            custom_name = st.text_input(f"段階{i+1}の名前", value=default_stage_names[i] if i < len(default_stage_names) else f"段階{i+1}", key=f"flow_custom_name_{i}")
                                            stage_names.append(custom_name)
                                else:
                                    stage_names = default_stage_names[:num_stages]
                                    st.info(f"✅ デフォルト段階名を使用：{' → '.join(stage_names)}")

                                st.markdown("#### ⚙️ ステップ3：対象語句を選択")
                                word_selection_method = st.radio("対象語句の選択方法を選んでください", ("①出現頻度TOP N から選択", "②任意の語句を入力"), key="flow_word_method")

                                selected_words = []
                                if word_selection_method == "①出現頻度TOP N から選択":
                                    top_n_range = st.slider("表示する上位語句数（5～50語）", min_value=5, max_value=50, value=10, step=1, key="flow_top_n_range")
                                    top_words = df_result['語句'].head(top_n_range).tolist()
                                    st.write(f"**上位{top_n_range}語から推移を分析したい語句を選択してください：**")
                                    selected_words = st.multiselect("語句を選択（複数選択可能）", top_words, default=top_words[:5], key="flow_word_multiselect")
                                    st.caption(f"✅ {len(selected_words)}個の語句を選択しました")
                                else:
                                    st.write("**分析したい語句を入力してください（改行区切り）：**")
                                    words_input_text = st.text_area("例：\n生活\n保護\n介護\n困る", height=100, key="flow_word_input")
                                    selected_words = [w.strip() for w in words_input_text.split('\n') if w.strip()]
                                    st.caption(f"✅ {len(selected_words)}個の語句を入力しました")

                                st.markdown("#### ⚙️ ステップ4：分析を実行")
                                if st.button("▶ 分析を実行", key="flow_execute_improved"):
                                    if not selected_words:
                                        st.error("❌ 対象語句を1つ以上選択してください")
                                    else:
                                        with st.spinner("⏳ テキストを自動分割し、語句をカウント中..."):
                                            stages_text_list = split_text_into_stages(text, num_stages)
                                            stage_counts = {}
                                            for stage_name, stage_text in zip(stage_names, stages_text_list):
                                                word_counter = count_words_with_morphology(
                                                    stage_text,
                                                    synonym_dict=synonym_dict,
                                                    stopwords=set([w.strip() for w in stop_words_text.replace('\n', ',').split(',') if w.strip()]),
                                                    target_pos=target_pos
                                                )
                                                stage_counts[stage_name] = word_counter

                                            df_crosstab = pd.DataFrame({"語句": selected_words})
                                            for stage_name in stage_names:
                                                df_crosstab[stage_name] = df_crosstab["語句"].apply(lambda w: stage_counts[stage_name].get(w, 0))

                                            st.success("✅ 分析完了！")

                                            st.markdown("---")
                                            st.header("📊 分析結果")

                                            tab_table, tab_chart_line, tab_chart_bubble, tab_test, tab_excel = st.tabs([
                                                "📋 クロス集計表", "📈 折れ線グラフ", "🔵 バブルチャート", "🔬 有意差検定", "📥 Excel出力"
                                            ])

                                            with tab_table:
                                                st.write("### クロス集計表（語句 × 段階）")
                                                st.dataframe(df_crosstab, use_container_width=True)
                                                st.info(f"💡 {len(selected_words)}個の語句 × {num_stages}段階 = {len(selected_words) * num_stages}セルのデータ")

                                            with tab_chart_line:
                                                st.write("### 📈 折れ線グラフ")
                                                draw_word_transition_line(df_crosstab, stage_names)

                                            with tab_chart_bubble:
                                                st.write("### 🔵 バブルチャート")
                                                draw_word_transition_bubble(df_crosstab, stage_names)

                                            with tab_test:
                                                st.write("### 🔬 有意差検定（カイ二乗検定）")
                                                try:
                                                    from scipy.stats import chi2_contingency
                                                    contingency_table = df_crosstab.set_index('語句').values
                                                    chi2, p_value, dof, expected_freq = chi2_contingency(contingency_table)
                                                    st.markdown(f"""
                            ### カイ二乗検定の結果
                            | 項目 | 値 |
                            |---|---|
                            | **検定手法** | カイ二乗検定 |
                            | **χ² 統計量** | {chi2:.3f} |
                            | **p 値** | {p_value:.4f} |
                            | **自由度** | {dof} |
                            | **有意差判定** | {"✅ あり（p < 0.05）" if p_value < 0.05 else "❌ なし（p ≥ 0.05）"} |
                            """)
                                                    if p_value < 0.05:
                                                        st.success(f"✅ 段階間に **統計的に有意な差** があります。\n\nつまり、選択した語句の出現パターンは、段階によって **有意に異なっています**。")
                                                    else:
                                                        st.info(f"❌ 段階間に有意な差は認められません。\n\nつまり、選択した語句の出現パターンは、段階によって **大きな違いがない**ようです。")
                                                    st.session_state['flow_test_result'] = {'chi2': chi2, 'p_value': p_value, 'dof': dof}
                                                except Exception as e:
                                                    st.error(f"検定中にエラーが発生しました: {e}")

                                            with tab_excel:
                                                st.write("### 📥 Excelで一括ダウンロード")
                                                buffer = io.BytesIO()
                                                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                                    df_crosstab.to_excel(writer, sheet_name='クロス集計', index=False)
                                                    params = {"項目": ["分析日時", "段階数", "対象語句数", "段階名", "テキスト総字数"], "値": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), num_stages, len(selected_words), " → ".join(stage_names), len(text)]}
                                                    pd.DataFrame(params).to_excel(writer, sheet_name='分析パラメータ', index=False)
                                                    if 'flow_test_result' in st.session_state:
                                                        test_result = st.session_state['flow_test_result']
                                                        test_df = pd.DataFrame({"検定手法": ["カイ二乗検定"], "χ²統計量": [test_result['chi2']], "p値": [test_result['p_value']], "自由度": [test_result['dof']], "有意差": ["あり" if test_result['p_value'] < 0.05 else "なし"]})
                                                        test_df.to_excel(writer, sheet_name='検定結果', index=False)
                                                buffer.seek(0)
                                                st.download_button("📊 結果をExcelでダウンロード（3シート）", data=buffer.getvalue(), file_name=f"flow_analysis_{stage_names[0]}_to_{stage_names[-1]}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_flow_excel_final")
                                                st.info("✅ **3つのシートが含まれています：**\n\n1. **クロス集計**：語句 × 段階の出現度数\n2. **分析パラメータ**：いつ、どのように分析したか\n3. **検定結果**：有意差検定の統計量とp値")

                        st.markdown("---")
                        st.header("3. AI分析（ローカルLLM）")
                        tab_ai, tab_after_coding = st.tabs(["🤖 テキスト要約・分析", "✨ AIアフターコーディング（辞書作成）"])

                        with tab_ai:
                            st.write("### LM Studioを使ったローカルAI要約")
                            st.write("※あらかじめLM Studioで Local Server (ポート1234) を起動しておいてください。")

                            ai_prompt = st.text_area("AIへの指示（プロンプト）:", value="以下のテキストを読み込み、主要なトピックを3つに分けて要約してください。", height=100)

                            tab_api, tab_file = st.tabs(["🚀 APIで直接実行", "📁 ファイル経由で実行（推奨: 文字数制限なし）"])

                            with tab_api:
                                api_text_len = len(text)
                                st.caption(f"テキスト長: {api_text_len}文字")
                                if st.button("AIで要約を実行する", key="btn_ai_api"):
                                    with st.spinner("AIが考え中...（ローカルマシンの性能により時間がかかります）"):
                                        try:
                                            from openai import OpenAI
                                            client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
                                            response = client.chat.completions.create(
                                                model="local-model",
                                                messages=[
                                                    {"role": "system", "content": "あなたは優秀なデータアナリストです。"},
                                                    {"role": "user", "content": f"{ai_prompt}\n\n【テキストデータ】\n{text}"}
                                                ],
                                                temperature=0.7,
                                            )
                                            st.session_state['summary_result'] = response.choices[0].message.content
                                            st.success("出力完了！")
                                        except Exception as e:
                                            err_msg = str(e)
                                            st.error(f"AIとの通信に失敗しました。\n{err_msg}")
                                            context_keywords = ["context length", "context window", "too many tokens", "token limit", "maximum length", "too long"]
                                            if any(kw in err_msg.lower() for kw in context_keywords):
                                                st.warning(f"💡 **テキストが長すぎるため、APIでの直接処理ができません。** 下の「📁 ファイル経由で実行」タブを開いて、テキストファイルをLM StudioのGUIに読み込ませてください（文字数制限なし）。")

                                if 'summary_result' in st.session_state:
                                    st.markdown(f"> {st.session_state['summary_result']}")
                                    summary_text_for_dl = st.session_state['summary_result'].encode('utf-8-sig')
                                    st.download_button("📥 要約結果をテキストで保存", data=summary_text_for_dl, file_name="ai_summary_result.txt", mime="text/plain")

                            with tab_file:
                                text_len = len(text)
                                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
                                os.makedirs(output_dir, exist_ok=True)
                                text_file_path = os.path.join(output_dir, f"analysis_text_{ts}.txt")
                                with open(text_file_path, "w", encoding="utf-8") as f:
                                    f.write(text)
                                with open(text_file_path, "rb") as f:
                                    file_bytes = f.read()
                                st.download_button("📥 分析対象テキストをファイルとして保存", data=file_bytes, file_name=f"analysis_text_{ts}.txt", mime="text/plain")
                                st.info(f"✅ テキストを保存しました（{text_len}文字）")
                                st.markdown(f"""
**📋 分析手順（ファイル経由）**

1. LM Studio を起動し、左側の **💬 Chat** タブを開く
2. 上部の **📎 ファイル添付** ボタンをクリックし、上記の保存ボタンからダウンロードしたファイル（または `{os.path.relpath(text_file_path)}`）を選択
3. チャット画面で以下の**プロンプト**を入力して実行

```text
{ai_prompt}
```
                                """)

                        with tab_after_coding:
                            st.write("### 抽出語の自動グルーピング（辞書作成支援）")
                            st.write("頻出単語の上位リストをAIに読み込ませ、同義語や関連語をまとめるための「ゆらぎ統一辞書」のベースを自動作成します。")
                            top_n = st.number_input("AIに渡す上位単語の数（10〜500語）:", min_value=10, max_value=500, value=100, step=10)
                            top_words_list = df_result.head(top_n)["語句"].tolist() if "語句" in df_result.columns else df_result.head(top_n).iloc[:, 0].tolist()
                            words_text = ", ".join(top_words_list)
                            st.info(f"**AIに渡す対象単語（出現頻度上位{top_n}語）:**\n{words_text}")

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
                                            temperature=0.3,
                                        )
                                        st.session_state['after_coding_result'] = response.choices[0].message.content
                                        st.success("辞書の作成が完了しました！")
                                    except Exception as e:
                                        st.error(f"AIとの通信に失敗しました。エラー詳細: {e}")

                            if 'after_coding_result' in st.session_state:
                                st.write("#### AIの作成結果")
                                st.code(st.session_state['after_coding_result'], language="csv")
                                csv_data = st.session_state['after_coding_result'].encode('utf-8-sig')
                                st.download_button("📥 ゆらぎ統一辞書（CSV）としてダウンロード", data=csv_data, file_name="ai_synonym_dict.csv", mime="text/csv")
                                st.caption("※ダウンロードしたCSVファイルは、左側のメニュー「同義語・ゆらぎ統一辞書をアップロード」からそのまま読み込ませて分析に利用できます。必要に応じてExcel等で開いて微調整してください。")
