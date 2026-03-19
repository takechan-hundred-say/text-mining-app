# text-mining-app
StreamlitとローカルLLMを使ったテキストマイニングアプリ

計量テキスト分析ツール (Quantitative Text Analysis Tool)
誰でも簡単にテキストデータを分析・可視化できる、Python (Streamlit) ベースのローカルアプリケーションです。

🌟 主な機能
多様なファイル形式に対応: TXT, CSV, DOCX, PDFからテキストを自動抽出
形態素解析と頻出語抽出: Janomeを用いた高精度な日本語の単語分割
共起ネットワーク図: 単語同士のつながりを可視化
感情分析: ポジティブ/ネガティブのスコア化と、文章の進行に伴う感情推移のグラフ化
AI要約機能: ローカルLLM（LM Studio等）と連携し、セキュアな環境で文章を要約・ダウンロード可能
🚀 インストールと起動方法
前提条件
Python 3.9 以上
セットアップ
リポジトリをクローンまたはダウンロードします。

必要なライブラリをインストールします。

pip install -r requirements.txt  
アプリを起動します。 bash

streamlit run app.py

📄 ライセンス 本プロジェクトは MIT License のもとで公開されています。 使用しているサードパーティ製ライブラリのライセンスについては ThirdPartyNotices.txt をご参照ください。

✍️ 開発者 坂本　毅啓（Takeharu Sakamoto）　北九州市立大学(The University of Kitakyushu)
