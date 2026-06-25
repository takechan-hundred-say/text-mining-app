import io
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from app.config import DEFAULT_FONT_PATH


def draw_wordcloud(df_result):
    import numpy as np
    st.write("### ワードクラウド（出現頻度が高いほど大きく表示）")
    wc_text = " ".join([" ".join([word] * count) for word, count in zip(df_result['語句'], df_result['頻度'])])
    font_path = DEFAULT_FONT_PATH
    width, height = 800, 400
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height / 2, width / 2
    radius_y, radius_x = height / 2, width / 2
    mask = ((x - center_x) / radius_x)**2 + ((y - center_y) / radius_y)**2 > 1
    mask = 255 * mask.astype(int)
    wc = WordCloud(
        width=width, height=height, background_color='white',
        font_path=font_path, colormap='viridis', collocations=False,
        prefer_horizontal=1.0, mask=mask
    ).generate(wc_text)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis("off")
    plt.tight_layout()
    st.pyplot(fig_wc)
    buf_wc = io.BytesIO()
    fig_wc.savefig(buf_wc, format="png", dpi=300)
    st.download_button("🖼️ ワードクラウドをPNGで保存", data=buf_wc.getvalue(), file_name="wordcloud.png", mime="image/png")
