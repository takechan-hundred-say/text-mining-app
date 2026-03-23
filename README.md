# text-mining-app
StreamlitとローカルLLMを使ったテキストマイニングアプリ

# 計量テキスト分析アプリ
誰でも簡単にテキストデータを分析・可視化できる、Python (Streamlit) ベースのローカルアプリケーションです。

## 🌟 主な機能
* 多様なファイル形式に対応: TXT, CSV, DOCX, PDFからテキストを自動抽出
* 形態素解析と頻出語抽出: Janomeを用いた高精度な日本語の単語分割
* 共起ネットワーク図: 単語同士のつながりを可視化
* 感情分析: ポジティブ/ネガティブのスコア化と、文章の進行に伴う感情推移のグラフ化
* AI要約機能: ローカルLLM（LM Studio等）と連携し、セキュアな環境で文章を要約・ダウンロード可能
* クラスター分析：語句、文章、段落、ファイル全体単位で分類が可能です。（K平均法、デンドログラム、散布図）

## 🚀 インストールと起動方法
前提条件
  Python 3.9 以上

### 本体のセットアップ
#### ①セットアップファイルからインストールする場合
* リリースからセットアップファイルをダウンロードします。
  > https://github.com/takechan-hundred-say/text-mining-app/releases/download/v0.4.0/Text_mining_mysetup_ver.0.4.exe
* setupのexeファイルを実行して、インストールを行ってください。
* 最初の起動には少し時間がかかりますので、ご注意ください。
#### ②リポジトリからインストールする場合
* リポジトリをクローンまたはダウンロードします。
* 必要なライブラリをインストールします。
  * pip install -r requirements.txt  
* アプリを起動します。
  * bash streamlit run app.py

### LMStudioのセットアップ
* ローカルでのAI要約を実施するには、LMStudioが必要です。
* LMStudioをダウンロードして、インストールしてください。
  * おすすめは「google/gemma-3-12b」（gemma-3-12b-it-Q4_K_M.gguf,　8.15GB）です。（これより小さいモデルでの検証はしていません。）
  * ローカルサーバーとしてRunningさせて、アドレスは「http://127.0.0.1:1234」として下さい。

## 📄 ライセンス 
本プロジェクトは MIT License のもとで公開されています。 使用しているサードパーティ製ライブラリのライセンスについては ThirdPartyNotices.txt をご参照ください。

## 更新履歴
* 2026.03.23　クラスター分析を追加しました。それに伴いThirdpartyNotices.txtとrequirements.txtを更新しました。

## ✍️ 開発者
- **氏名**: 坂本　毅啓
- **所属**: 北九州市立大学　地域創生学群
- **連絡先**: Researchmap https://researchmap.jp/s-takeharu

## 📝 引用について
研究や論文で本アプリを使用された場合は、以下のように引用していただけると幸いです。
> 坂本毅啓 (2026)「計量テキスト分析ツール (text-mining-app)」（GitHub repository: https://github.com/takechan-hundred-say/text-mining-app.git、閲覧日）。
* 2026年秋頃に、アプリ開発及び評価に関する論文を発表する予定です。
