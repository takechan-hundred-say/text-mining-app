import io
import zipfile
import pandas as pd
import docx
import pypdf
import streamlit as st


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
