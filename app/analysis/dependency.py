import re
import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def draw_dependency_analysis(text):
    st.subheader("🔗 係り受け解析")
    st.write("文節・句単位の係り受け関係（どの単語がどの単語に係るか）を解析・可視化します。")

    st.info(
        "💡 **GINZA** (spaCy ベース) を使用します。インストール: `pip install ginza ja_ginza`"
    )

    max_sentences = st.slider("解析する文数（先頭から）", min_value=1, max_value=50, value=5, key="dep_max_sent")

    if not st.button("▶ 係り受け解析を実行", key="run_dep"):
        st.info("👆 実行ボタンを押してください。")
        return

    sentences = [s.strip() + "。" for s in text.replace("\n", "。").split("。") if s.strip()]
    target_sentences = sentences[:max_sentences]

    with st.spinner("GINZA で係り受け解析を実行中..."):
        result = _parse_with_ginza(target_sentences)

    if result is None:
        st.error("GINZA の解析に失敗しました。")
        return

    relations_list, tree_text_list = result

    st.success(f"✅ {len(target_sentences)} 文の係り受け解析が完了しました。")

    tab_tree, tab_table, tab_graph = st.tabs([
        "🌲 係り受けツリー", "📋 関係一覧", "🕸️ グラフ可視化"
    ])

    with tab_tree:
        for i, tree_text in enumerate(tree_text_list):
            with st.expander(f"文{i+1}: {target_sentences[i][:60]}..." if len(target_sentences[i]) > 60 else f"文{i+1}: {target_sentences[i]}", expanded=(i == 0)):
                st.text(tree_text)

    with tab_table:
        _show_relations_table(relations_list)

    with tab_graph:
        _draw_dependency_graph(relations_list, target_sentences)


def _parse_with_cabocha(sentences):
    try:
        import cabocha
        parser = cabocha.Parser()
    except ImportError:
        st.error("CaboCha がインストールされていません。")
        st.info("`pip install cabocha` を実行し、CaboCha本体もインストールしてください。")
        return None
    except Exception as e:
        st.error(f"CaboCha の初期化に失敗しました: {e}")
        return None

    all_relations = []
    tree_texts = []

    for sentence in sentences:
        if not sentence.strip():
            continue
        try:
            tree = parser.parse(sentence)
        except Exception as e:
            st.warning(f"CaboCha 解析エラー: {e}")
            tree_texts.append(f"[解析エラー: {e}]")
            continue

        sent_relations = []
        for i in range(tree.chunk_size()):
            chunk = tree.chunk(i)
            link = chunk.link
            score = chunk.score

            chunk_tokens = []
            for j in range(chunk.token_pos, chunk.token_pos + chunk.token_size):
                token = tree.token(j)
                chunk_tokens.append(token.surface)
            chunk_text = "".join(chunk_tokens)

            if link == -1:
                head_text = "ROOT"
            else:
                link_chunk = tree.chunk(link)
                link_tokens = []
                for j in range(link_chunk.token_pos, link_chunk.token_pos + link_chunk.token_size):
                    token = tree.token(j)
                    link_tokens.append(token.surface)
                head_text = "".join(link_tokens)

            sent_relations.append({
                "dependent": chunk_text,
                "head": head_text,
                "score": score
            })
            all_relations.append({
                "文番号": len(tree_texts) + 1,
                "文": sentence[:80],
                "係り元（従属部）": chunk_text,
                "係り先（主部）": head_text,
                "スコア": round(score, 3)
            })

        tree_str_lines = [f"[文{len(tree_texts)+1}] {sentence}"]
        for r in sent_relations:
            arrow = " ← " if r["head"] != "ROOT" else " [ROOT]"
            tree_str_lines.append(f"  {r['dependent']}{arrow}{r['head']}  (score: {r['score']:.3f})")
        tree_texts.append("\n".join(tree_str_lines))

    return all_relations, tree_texts


def _parse_with_knp(sentences):
    try:
        from pyknp import KNP
        knp = KNP()
    except ImportError:
        st.error("pyknp がインストールされていません。")
        st.info("`pip install pyknp` を実行し、JUMAN++ と KNP も別途インストールしてください。")
        return None
    except Exception as e:
        st.error(f"KNP の初期化に失敗しました: {e}")
        return None

    all_relations = []
    tree_texts = []

    for sentence in sentences:
        if not sentence.strip():
            continue
        try:
            result = knp.parse(sentence)
        except Exception as e:
            st.warning(f"KNP 解析エラー: {e}")
            tree_texts.append(f"[解析エラー: {e}]")
            continue

        sent_relations = []
        for bnst in result.bnst_list():
            dependent_text = bnst.genkei if bnst.genkei else bnst.midasi
            parent_id = bnst.parent_id

            if parent_id == -1:
                head_text = "ROOT"
            else:
                parent_bnst = result.bnst_list()[parent_id]
                head_text = parent_bnst.genkei if parent_bnst.genkei else parent_bnst.midasi

            dpndtype = bnst.dpndtype if bnst.dpndtype else ""

            sent_relations.append({
                "dependent": dependent_text,
                "head": head_text,
                "type": dpndtype
            })
            all_relations.append({
                "文番号": len(tree_texts) + 1,
                "文": sentence[:80],
                "係り元（従属部）": dependent_text,
                "係り先（主部）": head_text,
                "関係タイプ": dpndtype
            })

        tree_str_lines = [f"[文{len(tree_texts)+1}] {sentence}"]
        for r in sent_relations:
            arrow = " ← " if r["head"] != "ROOT" else " [ROOT]"
            type_str = f" [{r['type']}]" if r["type"] else ""
            tree_str_lines.append(f"  {r['dependent']}{arrow}{r['head']}{type_str}")
        tree_texts.append("\n".join(tree_str_lines))

    return all_relations, tree_texts


def _parse_with_ginza(sentences):
    try:
        import spacy
        try:
            nlp = spacy.load("ja_ginza_electra")
        except OSError:
            try:
                nlp = spacy.load("ja_ginza")
            except OSError:
                st.error("GINZA モデルが見つかりません。")
                st.info("`pip install ginza ja_ginza_electra` または `python -m spacy download ja_ginza` を実行してください。")
                return None
    except ImportError:
        st.error("spacy / ginza がインストールされていません。")
        st.info("`pip install ginza ja_ginza_electra` を実行してください。")
        return None

    all_relations = []
    tree_texts = []

    text = "".join(sentences)
    try:
        doc = nlp(text)
    except Exception as e:
        st.error(f"GINZA 解析エラー: {e}")
        return None

    sent_idx = 0
    sent_boundaries = []
    for sent in doc.sents:
        sent_boundaries.append((sent.start, sent.end))

    for sent_start, sent_end in sent_boundaries:
        sent_idx += 1
        sent_tokens = list(doc[sent_start:sent_end])
        sent_text = doc[sent_start:sent_end].text

        sent_relations = []
        token_map = {}
        for i, token in enumerate(sent_tokens):
            token_map[id(token)] = i

        for token in sent_tokens:
            if token.head == token:
                head_text = "ROOT"
                head_idx = -1
            else:
                head_idx = token_map.get(id(token.head), -1)
                if head_idx >= 0 and head_idx < len(sent_tokens):
                    head_text = sent_tokens[head_idx].text
                else:
                    head_text = token.head.text

            sent_relations.append({
                "dependent": token.text,
                "head": head_text,
                "type": token.dep_,
                "head_idx": head_idx
            })
            all_relations.append({
                "文番号": sent_idx,
                "文": sent_text[:80],
                "係り元（従属部）": token.text,
                "品詞": token.pos_,
                "係り先（主部）": head_text,
                "関係ラベル": token.dep_
            })

        tree_str_lines = [f"[文{sent_idx}] {sent_text}"]
        for r in sent_relations:
            arrow = " ← " if r["head"] != "ROOT" else " [ROOT]"
            type_str = f" [{r['type']}]" if r["type"] else ""
            tree_str_lines.append(f"  {r['dependent']}{arrow}{r['head']}{type_str}")
        tree_texts.append("\n".join(tree_str_lines))

    return all_relations, tree_texts


def _show_relations_table(relations_list):
    if not relations_list:
        st.info("表示できる関係がありません。")
        return
    df = pd.DataFrame(relations_list)
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 関係一覧をCSVでダウンロード", data=csv,
                       file_name="dependency_relations.csv", mime="text/csv",
                       key="dl_dep_csv")


def _draw_dependency_graph(relations_list, sentences):
    if not relations_list:
        st.info("グラフを描画できるデータがありません。")
        return

    G = nx.DiGraph()

    for r in relations_list:
        dep = r.get("係り元（従属部）", r.get("dependent", ""))
        head = r.get("係り先（主部）", r.get("head", ""))
        label = r.get("関係ラベル", r.get("関係タイプ", r.get("type", "")))
        sent_no = r.get("文番号", 1)

        if not dep or not head:
            continue
        if head == "ROOT":
            G.add_node(dep, sentence=sent_no)
            G.add_node(f"ROOT_{sent_no}", label="ROOT")
            G.add_edge(dep, f"ROOT_{sent_no}", label=label or "")
        else:
            G.add_node(dep, sentence=sent_no)
            G.add_node(head, sentence=sent_no)
            G.add_edge(dep, head, label=label or "")

    if len(G.nodes) == 0:
        st.info("グラフを描画できるデータがありません。")
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(G.nodes) * 0.3)))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    node_labels = {n: n.replace("ROOT_", "ROOT(") + ")" if n.startswith("ROOT_") else n for n in G.nodes()}
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True) if d.get("label")}

    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='#ADD8E6',
                           edgecolors='#333', linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9,
                            font_family='Meiryo', ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->',
                           arrowsize=15, edge_color='#888', width=1.2, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7,
                                 font_family='Meiryo', ax=ax)

    ax.set_title("係り受け構造グラフ", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    st.download_button("🖼️ グラフをPNGで保存", data=buf.getvalue(),
                       file_name="dependency_graph.png", mime="image/png",
                       key="dl_dep_graph")
    plt.close(fig)
