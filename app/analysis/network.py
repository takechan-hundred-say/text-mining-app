import os
import io
import streamlit as st
import networkx as nx
from collections import defaultdict
from janome.tokenizer import Tokenizer
from pyvis.network import Network
import streamlit.components.v1 as components
import matplotlib.pyplot as plt


def draw_cooccurrence_network(df_result, sentences_words):
    st.write("### 🕸️ 共起ネットワーク")
    st.write("単語同士が一緒に出現する傾向を可視化します。線が太いほど、強い共起関係があります。")
    col1, col2 = st.columns(2)
    with col1:
        num_words = st.slider("表示する単語数", min_value=10, max_value=100, value=30, step=5, key="cooc_words")
    with col2:
        min_edge_weight = st.slider("共起回数の下限", min_value=1, max_value=10, value=2, step=1, key="cooc_edges")
    top_words = df_result['語句'].head(num_words).tolist()
    edge_counts = defaultdict(int)
    for sentence_words in sentences_words:
        for i in range(len(sentence_words) - 1):
            for j in range(i + 1, len(sentence_words)):
                w1, w2 = sorted([sentence_words[i], sentence_words[j]])
                edge_counts[(w1, w2)] += 1
    G = nx.Graph()
    for (w1, w2), weight in edge_counts.items():
        if weight >= min_edge_weight:
            if w1 in top_words and w2 in top_words:
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
        fig_net, ax_net = plt.subplots(figsize=(14, 10))
        pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors_for_static, node_size=1500, alpha=0.9, ax=ax_net, edgecolors='white', linewidths=2)
        edge_weights = [G[u][v]['weight'] * 0.8 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='#A0C4FF', alpha=0.6, ax=ax_net)
        nx.draw_networkx_labels(G, pos, font_family='Meiryo', font_size=9, ax=ax_net, font_weight='bold')
        ax_net.set_title(f"共起ネットワーク図（表示単語数: {num_words}個）", fontsize=16, fontweight='bold', pad=20)
        ax_net.axis('off')
        plt.tight_layout()
        st.markdown("---")
        st.markdown("#### 📥 共起ネットワーク図をダウンロード")
        col1, col2 = st.columns(2)
        with col1:
            buf_png = io.BytesIO()
            fig_net.savefig(buf_png, format="png", dpi=300, bbox_inches='tight')
            buf_png.seek(0)
            st.download_button(label="🖼️ PNG（高画質）でダウンロード", data=buf_png.getvalue(), file_name="cooccurrence_network.png", mime="image/png", key="download_network_png")
        with col2:
            buf_svg = io.BytesIO()
            fig_net.savefig(buf_svg, format="svg", bbox_inches='tight')
            buf_svg.seek(0)
            st.download_button(label="📊 SVG（拡大対応）でダウンロード", data=buf_svg.getvalue(), file_name="cooccurrence_network.svg", mime="image/svg+xml", key="download_network_svg")
        st.markdown("#### 🌐 HTML版（動く図）")
        with open(f"{path}/network.html", 'rb') as html_file:
            st.download_button(label="📱 HTML（ブラウザで開ける動く図）でダウンロード", data=html_file, file_name="cooccurrence_network_interactive.html", mime="text/html", key="download_network_html")
        plt.close(fig_net)
    else:
        st.warning("指定された条件（単語数・共起回数）では、つながりが見つかりませんでした。スライダーの数値を小さく調整してみてください。")
