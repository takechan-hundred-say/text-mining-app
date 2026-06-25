import os
import io
import unicodedata
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from janome.tokenizer import Tokenizer as JanomeTokenizer


def draw_sentiment_analysis(text):
    st.write("### 感情分析（ネガポジ判定）")
    st.write("東北大学評価極性辞書（名詞編）を使用して、文章中の各文の感情をポジティブ／ネガティブに判定します。")
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
    sentences = [s.strip() + "。" for s in text.replace("\n", "。").split("。") if s.strip()]
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
            "文": sentence, "判定": label, "平均スコア": round(avg_score, 3),
            "ポジティブ語数": pos_count, "ネガティブ語数": neg_count, "推移スコア": avg_score
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
    sizes = [pos_count_total, neg_count_total, neu_count_total]
    colors = ['#ff9999', '#66b3ff', '#d3d3d3']
    if sum(sizes) > 0:
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        col_chart, col_table = st.columns([1, 1])
        with col_chart:
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300)
            st.download_button("🖼️ 円グラフを保存", data=buf.getvalue(), file_name="sentiment_pie.png", mime="image/png")
        with col_table:
            st.dataframe(df_sentiment[["文", "判定", "平均スコア", "ポジティブ語数", "ネガティブ語数"]], height=300)
    else:
        st.info("判定できる文がありませんでした。")
    st.markdown("---")
    st.write("#### 📈 文章の展開に伴う感情の推移")
    st.write("横軸が文章の進行（最初から最後）、縦軸が感情スコアを示します。")
    fig_line, ax_line = plt.subplots(figsize=(10, 4))
    ax_line.plot(df_sentiment.index + 1, df_sentiment["推移スコア"], marker='o', linestyle='-', color='#9b59b6')
    ax_line.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_line.set_xlabel("文の順番")
    ax_line.set_ylabel("感情スコア (←ネガティブ / ポジティブ→)")
    ax_line.spines['right'].set_visible(False)
    ax_line.spines['top'].set_visible(False)
    st.pyplot(fig_line)
    plt.close(fig_line)


def draw_sentiment_by_case(df_meta, text_col, meta_cols):
    st.write("### 📋 ケース別 感情分析")
    all_cols = list(df_meta.columns)
    text_col_options = [text_col] + [c for c in all_cols if c != text_col]
    selected_text_col = st.selectbox(
        "📝 感情分析する質問列（テキスト列）を選んでください",
        text_col_options, index=0, key="case_sentiment_text_col"
    )
    st.caption(f"現在選択中: **{selected_text_col}**　（他の自由記述列がある場合は切り替えてください）")
    method = st.radio(
        "感情分析の方式を選択してください",
        ["asari（AIモデル）", "東北大学評価極性辞書（名詞編）"],
        horizontal=True, key="case_sentiment_method"
    )
    group_col = None
    if meta_cols:
        group_col_input = st.selectbox(
            "📌 グループ化する属性列を選んでください",
            ["（グループ化しない）"] + meta_cols, key="case_sentiment_group"
        )
        if group_col_input != "（グループ化しない）":
            group_col = group_col_input
    else:
        st.info("属性列が読み込まれていません。属性列を含むCSV/Excelを読み込むとグループ集計が利用できます。")
    if not st.button("▶ ケース別感情分析を実行", key="run_case_sentiment"):
        return
    with st.spinner("ケースごとに感情分析中..."):
        results = []
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
                sentences = [s.strip() + "。" for s in case_text.replace("\n", "。").split("。") if s.strip()]
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
                avg_pos = sum(pos_scores) / len(pos_scores)
                avg_neg = sum(neg_scores) / len(neg_scores)
                if abs(avg_trend) < 0.2:
                    overall = "😐 ニュートラル"
                elif avg_trend > 0:
                    overall = "😊 ポジティブ"
                else:
                    overall = "😞 ネガティブ"
                record = {
                    "ケースNo": idx + 1, "テキスト（先頭50文字）": case_text[:50],
                    "総合判定": overall, "平均ポジティブ率": round(avg_pos, 3),
                    "平均ネガティブ率": round(avg_neg, 3), "平均推移スコア": round(avg_trend, 3),
                }
                for col in meta_cols:
                    record[col] = row[col]
                results.append(record)
            score_col = "平均推移スコア"
        else:
            dic_path = os.path.join("dic", "pn.csv.m3.120408.trim")
            if not os.path.exists(dic_path):
                st.error(f"辞書ファイルが見つかりません: {dic_path}"); st.caption("dicフォルダにpn.csv.m3.120408.trimを配置してください。"); return
            df_pn = pd.read_csv(dic_path, sep="\t", names=["term", "sentiment", "category"])
            df_pn["term"] = [unicodedata.normalize("NFKC", t) for t in df_pn["term"]]
            pn_dict = {}
            for _, r in df_pn.iterrows():
                s = r["sentiment"]
                if s == "p": pn_dict[r["term"]] = 1.0
                elif s == "n": pn_dict[r["term"]] = -1.0
                elif s == "e": pn_dict[r["term"]] = 0.0
            t = JanomeTokenizer()
            for idx, row in df_meta.iterrows():
                case_text = str(row[selected_text_col]) if pd.notna(row[selected_text_col]) else ""
                if not case_text.strip():
                    continue
                tokens = list(t.tokenize(case_text))
                scores = [pn_dict[unicodedata.normalize("NFKC", tk.base_form)]
                          for tk in tokens if unicodedata.normalize("NFKC", tk.base_form) in pn_dict]
                if not scores:
                    continue
                avg_score = sum(scores) / len(scores)
                pos_count = scores.count(1.0); neg_count = scores.count(-1.0); total = len(scores)
                if abs(avg_score) < 0.05: overall = "😐 ニュートラル"
                elif avg_score > 0: overall = "😊 ポジティブ"
                else: overall = "😞 ネガティブ"
                record = {"ケースNo": idx + 1, "テキスト（先頭50文字）": case_text[:50],
                          "総合判定": overall, "平均スコア": round(avg_score, 3),
                          "ポジティブ語数": pos_count, "ネガティブ語数": neg_count, "判定語数合計": total}
                for col in meta_cols:
                    record[col] = row[col]
                results.append(record)
            score_col = "平均スコア"
    if not results:
        st.warning("判定できるケースがありませんでした。テキスト列や辞書ファイルを確認してください。"); return
    df_result_case = pd.DataFrame(results)
    st.markdown("---"); st.write(f"#### 📋 ケース別 感情スコア一覧（{len(df_result_case)}件）"); st.dataframe(df_result_case, width='stretch')
    df_group_stats = None
    if group_col and group_col in df_result_case.columns:
        st.markdown("---"); st.write(f"#### 📊 「{group_col}」別 感情スコアの記述統計")
        df_group_stats = df_result_case.groupby(group_col)[score_col].describe().round(3).reset_index()
        df_group_stats.columns = [group_col, "件数", "平均値", "標準偏差", "最小値", "25%ile", "中央値", "75%ile", "最大値"]
        st.dataframe(df_group_stats, width='stretch')
        st.write(f"#### 📊 「{group_col}」別 平均感情スコア（確認用）")
        df_bar = df_result_case.groupby(group_col)[score_col].mean().reset_index()
        df_bar.columns = [group_col, "平均感情スコア"]
        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        colors = ["#6699ff" if v >= 0 else "#ff6666" for v in df_bar["平均感情スコア"]]
        ax_bar.bar(df_bar[group_col].astype(str), df_bar["平均感情スコア"], color=colors)
        ax_bar.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax_bar.set_xlabel(group_col); ax_bar.set_ylabel("平均感情スコア")
        ax_bar.set_title(f"{group_col} 別 平均感情スコア（確認用）")
        ax_bar.spines["right"].set_visible(False); ax_bar.spines["top"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_bar); plt.close(fig_bar)
    from scipy import stats
    if group_col and group_col in df_result_case.columns:
        st.markdown("---"); st.write(f"#### 🔬 統計的検定：「{group_col}」別 感情スコアの比較")
        test_type = st.radio("検定方式", ["ノンパラメトリック検定（推奨）", "パラメトリック検定"], index=0, horizontal=True, key="test_type_select")
        groups = df_result_case.groupby(group_col)[score_col].apply(list)
        group_names = list(groups.index); group_values = list(groups.values); n_groups = len(group_names)
        df_test_result = None
        if n_groups < 2:
            st.info("グループが1つのみのため、検定はできません。")
        elif test_type == "ノンパラメトリック検定（推奨）":
            if n_groups == 2:
                st.write("##### 📌 Mann-Whitney U 検定（2グループ・ノンパラメトリック）")
                g1, g2 = group_values[0], group_values[1]
                u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                import numpy as np
                n_total = len(g1) + len(g2); z_val = stats.norm.ppf(1 - u_p / 2); effect_r = z_val / np.sqrt(n_total)
                u_label = "✅ 有意差あり" if u_p < 0.05 else "❌ 有意差なし"
                st.markdown(f"U統計量: {round(u_stat,3)} / p値: **{round(u_p,4)}** / 効果量r: {round(effect_r,3)} / 判定: {u_label}")
                if u_p < 0.05: st.success(f"「{group_names[0]}」と「{group_names[1]}」の感情スコアには統計的に有意な差があります（p={round(u_p,4)}）。")
                else: st.info(f"「{group_names[0]}」と「{group_names[1]}」の感情スコアに有意差は認められませんでした（p={round(u_p,4)}）。")
                df_test_result = pd.DataFrame([{"検定手法": "Mann-Whitney U検定", "グループ1": group_names[0], "グループ2": group_names[1], "U統計量": round(u_stat,3), "p値": round(u_p,4), "効果量r": round(effect_r,3), "有意差": "あり" if u_p<0.05 else "なし"}])
            else:
                st.write(f"##### 📌 Kruskal-Wallis 検定（{n_groups}グループ）")
                h_stat, kw_p = stats.kruskal(*group_values)
                st.markdown(f"H統計量: {round(h_stat,3)} / p値: **{round(kw_p,4)}** / 判定: {'✅ 有意差あり' if kw_p<0.05 else '❌ 有意差なし'}")
                if kw_p < 0.05:
                    st.success(f"グループ間に有意差あり（p={round(kw_p,4)}）")
                    from itertools import combinations
                    pair_results = []; pairs = list(combinations(range(n_groups), 2)); n_pairs = len(pairs)
                    for i, j in pairs:
                        _, raw_p = stats.mannwhitneyu(group_values[i], group_values[j], alternative="two-sided")
                        adjusted_p = min(raw_p * n_pairs, 1.0)
                        pair_results.append({"グループ1": group_names[i], "グループ2": group_names[j], "p値（補正前）": round(raw_p,4), "p値（Bonferroni補正後）": round(adjusted_p,4), "有意差": "あり" if adjusted_p<0.05 else "なし"})
                    st.dataframe(pd.DataFrame(pair_results), width='stretch')
                    df_test_result = pd.concat([pd.DataFrame([{"検定手法": "Kruskal-Wallis", "H統計量": round(h_stat,3), "p値": round(kw_p,4), "有意差": "あり"}]), pd.DataFrame(pair_results)], ignore_index=True)
                else:
                    st.info(f"有意差なし（p={round(kw_p,4)}）")
                    df_test_result = pd.DataFrame([{"検定手法": "Kruskal-Wallis", "H統計量": round(h_stat,3), "p値": round(kw_p,4), "有意差": "なし"}])
        else:
            if n_groups == 2:
                g1, g2 = group_values[0], group_values[1]; lev_stat, lev_p = stats.levene(g1, g2)
                t_stat, t_p = stats.ttest_ind(g1, g2, equal_var=lev_p>=0.05)
                st.markdown(f"t={round(t_stat,3)} / p={round(t_p,4)} / {'✅ 有意' if t_p<0.05 else '❌ なし'}")
                df_test_result = pd.DataFrame([{"検定手法": "t検定", "t統計量": round(t_stat,3), "p値": round(t_p,4), "有意差": "あり" if t_p<0.05 else "なし"}])
            else:
                f_stat, anova_p = stats.f_oneway(*group_values)
                st.markdown(f"F={round(f_stat,3)} / p={round(anova_p,4)} / {'✅ 有意' if anova_p<0.05 else '❌ なし'}")
                if anova_p<0.05:
                    try:
                        from statsmodels.stats.multicomp import pairwise_tukeyhsd
                        tukey_result = pairwise_tukeyhsd(df_result_case[score_col].values, df_result_case[group_col].values, alpha=0.05)
                        st.dataframe(pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0]), width='stretch')
                    except ImportError: st.warning("Tukey HSDにはstatsmodelsが必要です")
                df_test_result = pd.DataFrame([{"検定手法": "ANOVA", "F統計量": round(f_stat,3), "p値": round(anova_p,4), "有意差": "あり" if anova_p<0.05 else "なし"}])
    else:
        df_test_result = None
    st.markdown("---"); st.write("#### 📥 Excelで一括ダウンロード")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_result_case.to_excel(writer, index=False, sheet_name="ケース別感情分析")
        if df_group_stats is not None:
            df_group_stats.to_excel(writer, index=False, sheet_name="グループ別集計")
        else:
            summary_df = df_result_case[[score_col]].describe().round(3).reset_index()
            summary_df.columns = ["統計量", "値"]; summary_df.to_excel(writer, index=False, sheet_name="全体サマリー")
        if df_test_result is not None:
            df_test_result.to_excel(writer, index=False, sheet_name="統計的検定結果")
    st.download_button("📥 結果をExcelでダウンロード", data=buffer.getvalue(), file_name=f"case_sentiment_{selected_text_col}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_case_sentiment_excel")
