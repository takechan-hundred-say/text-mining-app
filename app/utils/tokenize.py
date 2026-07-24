import os
import csv
import tempfile
import streamlit as st
import pandas as pd
from janome.tokenizer import Tokenizer as JanomeTokenizer
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


def parse_compound_words(option, custom_words_text, custom_dict_content):
    words = []
    if option == "1. 画面上で直接、語句定義を入力する" and custom_words_text:
        words = [w.strip() for w in custom_words_text.replace('\n', ',').split(',') if w.strip()]
    elif option == "2. ユーザーが作成した定義ファイルを読み込む" and custom_dict_content:
        words = [w.strip() for w in custom_dict_content.replace('\n', ',').split(',') if w.strip()]
    return words

_parse_compound_words = parse_compound_words


def _build_stopwords(stop_words_text, stop_words_content):
    stopwords = set([w.strip() for w in stop_words_text.replace('\n', ',').split(',') if w.strip()])
    if stop_words_content is not None:
        file_stopwords = [w.strip() for w in stop_words_content.replace('\n', ',').split(',') if w.strip()]
        stopwords.update(file_stopwords)
    return stopwords


def _to_alpha(n):
    letters = []
    while True:
        letters.append(chr(65 + n % 26))
        n //= 26
        if n == 0:
            break
    return "".join(reversed(letters))

def _preprocess_compounds(sentence, compound_words):
    if not compound_words:
        return sentence, {}
    compound_map = {}
    processed = sentence
    for i, compound in enumerate(sorted(compound_words, key=len, reverse=True)):
        if compound in processed:
            placeholder = f"CMPD{_to_alpha(i)}"
            compound_map[placeholder] = compound
            processed = processed.replace(compound, placeholder)
    return processed, compound_map


def _tokenize_janome(sentence, tokenizer, target_pos, stopwords, synonym_dict, compound_map):
    current_words = []
    for token in tokenizer.tokenize(sentence):
        surface = token.surface
        if surface in compound_map:
            word = compound_map[surface]
            word = synonym_dict.get(word, word)
            if word not in stopwords:
                current_words.append((word, '名詞'))
            continue
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
                current_words.append((word, pos))
    return current_words


def _tokenize_mecab(sentence, tagger, target_pos, stopwords, synonym_dict, compound_map):
    current_words = []
    node = tagger.parseToNode(sentence)
    while node:
        feature = node.feature.split(',')
        surface = node.surface
        if surface in compound_map:
            word = compound_map[surface]
            word = synonym_dict.get(word, word)
            if word not in stopwords:
                current_words.append((word, '名詞'))
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
                    current_words.append((word, mapped_pos))
        node = node.next
    return current_words


def _tokenize_jumanpp(sentence, jumanpp, target_pos, stopwords, synonym_dict, compound_map):
    current_words = []
    try:
        result = jumanpp.analysis(sentence)
        for mrph in result.mrph_list():
            surface = mrph.midasi
            if surface in compound_map:
                word = compound_map[surface]
                word = synonym_dict.get(word, word)
                if word not in stopwords:
                    current_words.append((word, '名詞'))
                continue
            pos = mrph.hinsi
            genkei = mrph.genkei if mrph.genkei != '*' else mrph.midasi
            if pos in target_pos:
                word = genkei
                if word not in stopwords and len(word) > 1:
                    word = synonym_dict.get(word, word)
                    current_words.append((word, pos))
    except Exception as e:
        st.error(f"JUMAN++ 解析エラー: {e}")
    return current_words


def _tokenize_sudachi(sentence, sudachi_tok, split_mode, target_pos, stopwords, synonym_dict, compound_map):
    current_words = []
    try:
        tokens = sudachi_tok.tokenize(sentence, split_mode)
        for token in tokens:
            surface = token.surface()
            if surface in compound_map:
                word = compound_map[surface]
                word = synonym_dict.get(word, word)
                if word not in stopwords:
                    current_words.append((word, '名詞'))
                continue
            pos = token.part_of_speech()[0]
            word = token.dictionary_form() if token.dictionary_form() != '*' else token.surface()
            if pos in target_pos:
                if word not in stopwords and len(word) > 1:
                    word = synonym_dict.get(word, word)
                    current_words.append((word, pos))
    except Exception as e:
        st.error(f"Sudachi 解析エラー: {e}")
    return current_words


@st.cache_data
def analyze_text(text, option, custom_words_text, custom_dict_content,
                 stop_words_text, stop_words_content, target_pos, synonym_dict,
                 analyzer_choice):
    stopwords = _build_stopwords(stop_words_text, stop_words_content)
    compound_words = _parse_compound_words(option, custom_words_text, custom_dict_content)

    USE_MECAB = (analyzer_choice == "MeCab + UniDic（高精度）")
    USE_JUMANPP = (analyzer_choice == "JUMAN++")
    USE_SUDACHI = (analyzer_choice == "Sudachi")

    mecab_tagger = None
    tokenizer = None
    jumanpp = None
    sudachi_tok = None
    split_mode = None
    temp_dict_path = None

    if USE_MECAB:
        import MeCab
        if compound_words:
            temp_dict_path = create_user_dict_file(compound_words)
            try:
                mecab_tagger = MeCab.Tagger(f'-u "{temp_dict_path}"')
            except Exception as e:
                st.warning(f"MeCab辞書の読み込みエラー: {e}。標準辞書で継続します。")
                mecab_tagger = MeCab.Tagger()
        else:
            mecab_tagger = MeCab.Tagger()
    elif USE_JUMANPP:
        try:
            from pyknp.juman.juman import Juman
            jumanpp = Juman(jumanpp=True, multithreading=True)
        except Exception as e:
            st.error(f"JUMAN++ の初期化に失敗しました: {e}")
            st.info("JUMAN++ を使用するには、別途 JUMAN++ のインストールが必要です。")
            return pd.DataFrame(), []
    elif USE_SUDACHI:
        try:
            from sudachipy import dictionary
            from sudachipy import tokenizer as sudachitokenizer
            sudachi_tok = dictionary.Dictionary().create()
            split_mode = sudachitokenizer.Tokenizer.SplitMode.C
        except Exception as e:
            st.error(f"Sudachi の初期化に失敗しました: {e}")
            st.info("Sudachi を使用するには `pip install sudachipy sudachidict_core` が必要です。")
            return pd.DataFrame(), []
    else:
        if compound_words:
            temp_dict_path = create_user_dict_file(compound_words)
            tokenizer = JanomeTokenizer(udic=temp_dict_path, udic_enc='utf8', udic_type='ipadic')
        else:
            tokenizer = JanomeTokenizer()

    sentences = text.replace('。', '。\n').split('\n')
    all_words_list = []
    sentences_words = []

    for sentence in sentences:
        if not sentence.strip():
            continue

        processed, compound_map = _preprocess_compounds(sentence, compound_words)
        current_sentence_words = []

        if USE_MECAB:
            results = _tokenize_mecab(processed, mecab_tagger, target_pos, stopwords, synonym_dict, compound_map)
        elif USE_JUMANPP:
            results = _tokenize_jumanpp(processed, jumanpp, target_pos, stopwords, synonym_dict, compound_map)
        elif USE_SUDACHI:
            results = _tokenize_sudachi(processed, sudachi_tok, split_mode, target_pos, stopwords, synonym_dict, compound_map)
        else:
            results = _tokenize_janome(processed, tokenizer, target_pos, stopwords, synonym_dict, compound_map)

        for word, pos in results:
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
