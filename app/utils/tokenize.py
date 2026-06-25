import os
import csv
import tempfile
import streamlit as st
import pandas as pd
from janome.tokenizer import Tokenizer
from collections import Counter
import sys


def create_user_dict_file(word_list):
    temp_file = tempfile.NamedTemporaryFile(
        mode='w', delete=False, encoding='utf-8', suffix='.csv', newline=''
    )
    writer = csv.writer(temp_file)
    for word in word_list:
        word = word.strip()
        if word:
            row = [word, -1, -1, -5000, '名詞', 'カスタム', '*', '*', '*', '*', word, '*', '*']
            writer.writerow(row)
    temp_file.close()
    return temp_file.name


@st.cache_data
def analyze_text(text, option, custom_words_text, custom_dict_content,
                 stop_words_text, stop_words_content, target_pos, synonym_dict,
                 analyzer_choice):
    USE_MECAB = (analyzer_choice == "MeCab + UniDic（高精度）")
    stopwords = set([w.strip() for w in stop_words_text.replace('\n', ',').split(',') if w.strip()])
    if stop_words_content is not None:
        file_stopwords = [w.strip() for w in stop_words_content.replace('\n', ',').split(',') if w.strip()]
        stopwords.update(file_stopwords)
    words = []
    if option == "1. 画面上で直接、語句定義を入力する" and custom_words_text:
        words = [w.strip() for w in custom_words_text.replace('\n', ',').split(',') if w.strip()]
    elif option == "2. ユーザーが作成した定義ファイルを読み込む" and custom_dict_content:
        words = [w.strip() for w in custom_dict_content.replace('\n', ',').split(',') if w.strip()]
    temp_dict_path = None
    if USE_MECAB:
        import MeCab
        if words:
            temp_dict_path = create_user_dict_file(words)
            try:
                mecab_tagger = MeCab.Tagger(f'-u "{temp_dict_path}"')
            except Exception as e:
                st.warning(f"MeCab辞書の読み込みエラー: {e}。標準辞書で継続します。")
                mecab_tagger = MeCab.Tagger()
        else:
            mecab_tagger = MeCab.Tagger()
    else:
        if words:
            temp_dict_path = create_user_dict_file(words)
            tokenizer = Tokenizer(udic=temp_dict_path, udic_enc='utf8', udic_type='ipadic')
        else:
            tokenizer = Tokenizer()
    sentences = text.replace('。', '。\n').split('\n')
    all_words_list = []
    sentences_words = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        current_sentence_words = []
        if USE_MECAB:
            compound_map = {}
            processed_sentence = sentence
            for i, compound in enumerate(sorted(words, key=len, reverse=True)):
                if compound in processed_sentence:
                    placeholder = f"CMPD{i}X"
                    compound_map[placeholder] = compound
                    processed_sentence = processed_sentence.replace(compound, placeholder)
            node = mecab_tagger.parseToNode(processed_sentence)
            while node:
                feature = node.feature.split(',')
                surface = node.surface
                if surface in compound_map:
                    word = compound_map[surface]
                    word = synonym_dict.get(word, word)
                    if word not in stopwords:
                        current_sentence_words.append(word)
                        all_words_list.append((word, '名詞'))
                    node = node.next
                    continue
                if len(feature) >= 8:
                    pos = feature[0]
                    pos_detail = feature[1]
                    base_form = feature[7] if feature[7] != '*' else surface
                    if pos_detail in ['非自立', '接尾', '数', '接続助詞']:
                        node = node.next
                        continue
                    pos_map = {'名詞': '名詞', '動詞': '動詞', '形容詞': '形容詞', '副詞': '副詞'}
                    mapped_pos = pos_map.get(pos)
                    if mapped_pos and mapped_pos in target_pos:
                        word = base_form
                        if word not in stopwords and len(word) > 1:
                            word = synonym_dict.get(word, word)
                            current_sentence_words.append(word)
                            all_words_list.append((word, mapped_pos))
                node = node.next
        else:
            for token in tokenizer.tokenize(sentence):
                pos_info = token.part_of_speech.split(',')
                pos = pos_info[0]
                pos_detail = pos_info[1] if len(pos_info) > 1 else ''
                base_form = token.base_form if token.base_form != '*' else token.surface
                if pos_detail in ['非自立', '接尾', '数']:
                    continue
                if pos in target_pos:
                    word = base_form
                    if word not in stopwords and len(word) > 1:
                        word = synonym_dict.get(word, word)
                        all_words_list.append((word, pos))
                        current_sentence_words.append(word)
        if current_sentence_words:
            sentences_words.append(current_sentence_words)
    all_words_counter = Counter(all_words_list)
    df_result_df = pd.DataFrame([
        {'語句': word, '品詞': pos, '頻度': count}
        for (word, pos), count in all_words_counter.items()
    ]).sort_values('頻度', ascending=False).reset_index(drop=True)
    if temp_dict_path and os.path.exists(temp_dict_path):
        os.remove(temp_dict_path)
    return df_result_df, sentences_words
