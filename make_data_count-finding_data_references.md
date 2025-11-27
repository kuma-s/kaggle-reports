# はじめに

Kaggleで開催された **「Make Data Count - Finding Data References」** コンペに参加しました。  
コンペの概要と上位解法、私の取り組みについて紹介させていただきます。

この記事の内容は、資料やメモを用意して、AIエージェントを使って元になるドラフトを作成してもらい、その後手直しして書いています。  

また、私の理解として記述をしているため（結構AIに書いて貰ってますが...）、一部に誤りや解釈の相違が含まれる可能性があります。あらかじめご了承ください。

興味ある方は、是非、実際にkaggleコンペのwebページを見ていただくと面白いかと思います。

<br>

## 目次

- [コンペ歴](#コンペ歴)
- [コンペ概要](#コンペ概要)
  - [コンペの目的](#コンペの目的)
    - [コンペティションの背景](#コンペティションの背景)
  - [データセット](#データセット)
    - [論文とデータセットの識別子](#論文とデータセットの識別子)
    - [提供されるファイル](#提供されるファイル)
    - [提出形式](#提出形式)
  - [評価指標](#評価指標)
- [上位解法](#上位解法)
  - [上位チームが採用していた効果的なアプローチ](#上位チームが採用していた効果的なアプローチ)
    - [1. 2段階パイプライン](#1-2段階パイプライン)
    - [2. DOIとAccession IDの戦略分離](#2-doiとaccession-idの戦略分離)
    - [3. 外部データソースの活用](#3-外部データソースの活用)
    - [4. テキストの前処理](#4-テキストの前処理)
    - [5. ヒューリスティックルールの追加](#5-ヒューリスティックルールの追加)
    - [6. LLMの量子化と高速化](#6-llmの量子化と高速化)
- [取り組んだ内容](#取り組んだ内容)
  - [解法](#解法)
    - [1. PDFとXMLをパース](#1-pdfとxmlをパース)
    - [2. パースしたデータから、データ引用箇所を抽出](#2-パースしたデータからデータ引用箇所を抽出)
    - [3. 抽出したAccession IDをPrimaryとSecondaryに分類](#3-抽出したaccession-idをprimaryとsecondaryに分類)
    - [4. 抽出したDOIをデータ引用かそれ以外かを分類](#4-抽出したdoiをデータ引用かそれ以外かを分類)
    - [5. データ引用と判断したIDをPrimaryとSecondaryに分類](#5-データ引用と判断したidをprimaryとsecondaryに分類)
    - [6. 特定のプレフィックスのDOIを取り除く](#6-特定のプレフィックスのdoiを取り除く)
  - [うまく行かなかったこと](#うまく行かなかったこと)
    - [PDFパーサーに高性能なライブラリを使う](#pdfパーサーに高性能なライブラリを使う)
    - [ディスカッションから重要な部分を見逃していた](#ディスカッションから重要な部分を見逃していた)
    - [正規表現で大きく順位を落とす](#正規表現で大きく順位を落とす)
  - [使用したツールなど](#使用したツールなど)
    - [情報収集](#情報収集)
      - [Kaggleデータのスクレイピング](#kaggleデータのスクレイピング)
      - [Google AI Studio](#google-ai-studio)
    - [モデル学習](#モデル学習)
      - [Google Colaboratory](#google-colaboratory)
    - [利用した技術](#利用した技術)
      - [モデル学習に利用](#モデル学習に利用)
        - [PEFT（Parameter-Efficient Fine-Tuning）ライブラリ](#peftparameter-efficient-fine-tuningライブラリ)
        - [LoRA（Low-Rank Adaptation）](#loralow-rank-adaptation)
        - [QLoRA（Quantized LoRA）](#qlorquantized-lora)
        - [PEFTでQLoRAを利用して学習するサンプル](#peftでqlorを利用して学習するサンプル)
      - [推論に利用](#推論に利用)
        - [vLLM（Very Large Language Model Serving）ライブラリ](#vllmvery-large-language-model-servingライブラリ)
        - [vLLMの実行サンプル](#vllmの実行サンプル)
  - [結果](#結果)
    - [最終スコア](#最終スコア)
    - [原因](#原因)
- [おわりに](#おわりに)
- [おまけ](#おまけ)

<br>

# コンペ歴
今まで細々とKaggleのコンペに参加していましたが（参加したり、しばらく何もしなかったり、何もしなかったり、、、、💤）、
機械学習などを業務で触れることはほぼありませんでした。

なので、業務でどのようにAIを組み込んだり、利用していくかってのは、あまり経験がないんですが、少しでも何か参考になることがあればいいなと思い共有させていただきます。

<br>

# コンペ概要

<https://www.kaggle.com/competitions/make-data-count-finding-data-references>

## コンペの目的

本コンペティションは、科学論文における **研究データへの言及（データ引用）** を特定し、その引用が「**プライマリ（Primary）**」か「**セカンダリ（Secondary）**」かを分類する高性能なモデルを開発することを目的としています。

- **Primary** - その研究のために新たに生成された生データまたは処理済みデータ
- **Secondary** - 既存の記録や公開データから再利用された生データまたは処理済みデータ

この取り組みは、科学データの価値を正しく評価し、オープンな科学データの再利用を促進することを目指す **Make Data Count (MDC)** というグローバルなイニシアチブを支援するものです。

### コンペティションの背景

科学的発見や革新の基盤となる科学データは、その重要性にもかかわらず、その価値が正しく評価されていません。研究の約86%のデータが「未引用」の状態にあり、データ引用は形式も多様で自動検出が困難な状況です。

著者は論文中でデータに言及する際、以下のような様々な方法を取ります：

- メソッドセクションでデータの詳細な説明を提供
- 他の箇所で間接的に言及
- 参考文献リストに正式な引用を記載
- データが「公開されている（publicly available）」と示唆
- データが「他から取得した（obtained from）」と言及

このような多様な言及方法により、プログラムによる自動的な特定が非常に困難となっています。

## データセット

参加者には、科学論文のPDFファイルとXMLファイルが提供され（約75%の論文にXMLファイルが存在）、これらのテキストからデータ参照を識別子（DOIやAccession ID）によって抽出し、その種類を分類します。

### 論文とデータセットの識別子

各オブジェクト（論文とデータセット）には、一意で永続的な識別子があります：

1. **DOI（Digital Object Identifier）**  
論文、書籍、データセット、図表、ソフトウェアなど  
デジタル出版物”を永続的に識別するための国際標準ID
   - **形式**: `https://doi.org/[prefix]/[suffix]`
   - **例**: `https://doi.org/10.1371/journal.pone.0303785`

2. **Accession ID**  
生命科学データベースに登録されたデータ（配列、遺伝子、構造、画像など）
   - データリポジトリごとに形式が異なる
   - **例**: `GSE12345` (Gene Expression Omnibus)、`PDB1Y2T` (Protein Data Bank)

**DOI** にはデータ以外も含まれていますが、**Accession ID** はデータのみになります。

### 提供されるファイル

- **train/{PDF,XML}** - 訓練用の論文（PDFとXML形式）
- **test/{PDF,XML}** - テスト用の論文（約2,600記事）
- **train_labels.csv** - 訓練用ラベル
  - `article_id` - 論文のDOI
  - `dataset_id` - データセット識別子
  - `type` - 引用タイプ（Primary/Secondary）
- **sample_submission.csv** - サンプル提出ファイル

### 提出形式

予測結果は、論文ID (`article_id`)、データセットID (`dataset_id`)、引用の種類 (`type`) の一意な組み合わせで提出します。データ参照を含まない論文は提出に含めず、含めた場合は偽陽性としてペナルティが課されます。

## 評価指標

評価は **F1スコア** で行われます。F1スコアは、情報検索の分野で一般的に用いられる指標で、精度（Precision）と再現率（Recall）を調和平均したものです。

$$
F_1 = 2\frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$


$$
\text{precision} = \frac{\text{tp}}{\text{tp}+\text{fp}}, \quad \text{recall} = \frac{\text{tp}}{\text{tp} +\text{fn}}
$$

F1指標は、再現率と精度を等しく重視し、両方を同時に最大化することを目指します。したがって、どちらか一方で極端に良い性能を発揮するよりも、両方で適度に良い性能を発揮する方が好まれます。

<br>

# 上位解法

上位チームの多くは、**ハイブリッドアプローチ**を採用していました。

1. Data Citation CorpusやEurope PMCといった外部データソースを活用してデータ引用候補を抽出し
2. その後LLMや勾配ブースティングモデルで分類する

という2段階の手法です。

特に、DOIとAccessionIDで異なる戦略を使い分けることが成功の鍵となっていました。

また、1位のチームの解法はDOIの分類にCatBoostを利用しており、LLMでもDeep Learningでもないモデルを利用して1位になっているのはすごいと思いましたし、道具は使う人によって、ほんとに効果的になるんだなと思いました。

| 順位      | 候補抽出 | 分類手法 |
|---------| --- | --- |
| **1位**  | ・Data Citation Corpus v4.1（DataCite）<br>・EUPMC Text-Mined Terms + Corpus (eupmc) | ・DOI：CatBoost（6-fold）<br>・Accession ID：Qwen2.5-Coder-32B（AWQ、vLLM） |
| **2位** | ・Data Citation Corpus v2<br>・DataCite Public Data File 2024<br>・EUPMC Text-Mined Terms | ・DOI：MedGemma-4B（LoRA）<br>・Accession ID：BiomedBERT |
| **3位** | ・DataCite Annual Dump 2024<br>・EUPMC Mined Terms Corpus<br>・Crossref 2025 Dump | ・DeBERTa-v3-Large（6-fold） |
| **4位** | ・Data Citation Corpus v3.0<br>・PMC TextMinedTerms<br>・Regex fallback + LLMフィルタリング | ・Qwen2.5ファミリー（14B/32B/72B）<br>・Tool-calling agent |
| **5位** | ・Data Citation Corpus v4.1<br>・EUPMC Original Mapping | ・Qwen2.5-32B-Instruct-AWQ<br>・Qwen3-32B-AWQ |

<br>

## 上位チームが採用していた効果的なアプローチ

### 1. 2段階パイプライン

すべての上位チームが採用した基本的なアプローチ：

1. **候補抽出（Stage 1）**: 論文テキストからデータ引用の可能性がある箇所を網羅的に抽出
   - Data Citation CorpusやEurope PMCなどの外部データベースを活用
   - 正規表現（Regex）も補完的に使用
   
2. **分類（Stage 2）**: 抽出された候補を、文脈情報に基づいて「Primary」か「Secondary」に分類
   - Large Language Model（LLM）や勾配ブースティングモデルを使用

### 2. DOIとAccession IDの戦略分離

データセットの種類によって異なるアプローチが効果的でした：

**DOIの分類アプローチ：**
- 論文とデータセットのメタデータ（タイトル、著者、発行年）を比較
- その類似度を特徴量としてCatBoostなどの機械学習モデルで分類
- 1位チーム：タイトル類似度と著者類似度が最重要特徴量

**Accession IDの分類アプローチ：**
- LLM（特にQwenモデル）を用いて、ID周辺の文脈を読み取らせて分類
- プロンプトエンジニアリングが精度に大きく影響
- 0-shotの単純なプロンプトが効果的

### 3. 外部データソースの活用

すべての上位チームが以下のデータソースを利用：

- **Data Citation Corpus** (v2, v3, v4.1など)
- **Europe PMC Text-Mined Terms**
- **DataCite Public Data File**
- **Crossref** (論文メタデータ)


### 4. テキストの前処理

- 論文末尾の「References」セクションは論文への引用が多くノイズ源となるため削除
- PDFをテキストに変換する際、空白文字やUnicodeの正規化が重要
- XMLとPDFの両方を活用（候補がPDFで見つからない場合はXMLも検索）

### 5. ヒューリスティックルールの追加

分類前後に適用されたルール：

- `SAMN`（BioSample）と`EMDB`のAccession IDはPrimary
- `dryad`リポジトリのDOIはPrimary
- 論文とデータセットのタイトル・著者が類似していればPrimary
- 多数の論文で引用されているデータセットはSecondary
- DOIが見つかった論文ではAccession IDを除外（一部例外あり）

### 6. LLMの量子化と高速化

推論時間の制約（9時間以内）に対応するため：

- **AWQ量子化**を使用してモデルサイズを削減
- **vLLM**を使用して推論を高速化
- **カスケード推論**：小さなモデルで全体を処理 → 不確実なサンプルのみ大きなモデルで再処理



<br>

# 取り組んだ内容

僕のアプローチを紹介します。上位チームのような洗練された手法ではありません😅  
多くは、パブリックノートブックやディスカッションを参考にし、そこに色々な手法を試して、効果があったものを追加していきました。

## 解法

### 1. PDFとXMLをパース
PDFパーサーにはpymupdfを、XMLパーサーにはlxmlを利用。  

<br>

### 2. パースしたデータから、データ引用箇所を抽出  
正規表現を使用して、データIDを抽出しました。  
データID箇所の抽出には以下の3パターンで実施しています。  
「**参考文献からデータ系のDOIのみを抽出**」の追加が今回の取り組みで一番スコアが伸びました。

   - 本文からDOIを抽出  
本文から「10.1234/abcd」のように「10.」で始まり、数字が続き、「/」区切りで、続く箇所を抽出します。


   - 参考文献からデータ系のDOIのみを抽出  
参考文献箇所は多数のDOIが記載されていますので、論文系などのデータ系ではないDOIが多数記載されています。  
本文と同じように抽出すると、すごい数のIDを抽出してしまい、スコアが大幅に下がります。  
なので、 DataCite（研究データ向けDOI登録機関）が提供している REST API のエンドポイント（https://api.datacite.org/prefixes） を利用して、
あらかじめデータ系と思われるプレフィックスを取得しておいて、それを含むDOIのみ抽出しました。


   - 全文からAccession IDを抽出  
Accession IDはデータ系のIDのため、全文から抽出しました。

<br>

### 3. 抽出したAccession IDをPrimaryとSecondaryに分類  
抽出したAccession IDは全てデータ引用とみなして、PrimaryかSecondaryかの分類をヒューリスティック（ルールベース）に実施しています。
（特定の種類のIDはPrimary、ある言葉がIDの前後に含まれているとPrimaryのような）

LLMで分類しようと試みたのですが、スコアが向上しなく、、諦めてパブリックノートを参考にしてルールベースにしました。。

<br>

### 4. 抽出したDOIをデータ引用かそれ以外かを分類  
ファインチューニングした **Qwen2.5 32b-instruct-awq** モデルを利用して、データ引用かそれ以外かの分類用のプロンプトを用意して、データ引用でないとされたIDを除外しています。

<br>

### 5. データ引用と判断したIDをPrimaryとSecondaryに分類  
4でデータ引用と判断したIDを、**Qwen2.5 72b-instruct-awq** モデルを利用して、分類用のプロンプトを作成して分類しています。

なぜ4より大規模なモデルを利用しているかといいますと、72bモデルは、精度は高かったのですが、GPUメモリを多く使用するため、
KaggleのGPUメモリでは、Out of Memoryしてしまいます。なので、CPUメモリにオフロードする設定を使いましたが、そのため速度がかなり遅かったです。  

そのため事前に件数を絞った状態のレコードに利用しないと制限時間内に終わりませんでした。  
大きなデータを事前に小さめのモデルで絞り、そこから大きめのモデルでさらに分類することを**カスケード推論**というらしいです。

<br>

### 6. 特定のプレフィックスのDOIを取り除く  
5で分類したIDから、データ系のDOIでないと思われるプレフィックスのIDを取り除く

<br>

## うまく行かなかったこと

### PDFパーサーに高性能なライブラリを使う 
MarkerというPDFパーサーが、より扱いやすい形でPDFをパースしてくれますが、  
処理時間が長く、制限時間内に処理が終わらないので使えませんでした。

<br>

### ディスカッションから重要な部分を見逃していた
うまく行かなかったというか、ちゃんと見てなかったというか、、  
データ引用ペアは、以下のソースに基づいて構築されていると、主催者によって[言及されていた](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/584337#3224495)ようで、上位陣もデータ引用の抽出に利用していました。（DataCiteは、そうとは知らずAPIのみ利用していました）

- **EUPMC（Europe PMC）**
- **DataCite**

<br>

### 正規表現で大きく順位を落とす
うまく行かなかったというか、最終結果でやらかしたことに気づいたことですが、、  
Accession IDの抽出のために追加した正規表現で大きく最終結果の順位を落としていました。。

最終結果まではパブリックリーダーボードで結果が表示されるのですが、
最終結果はプライベートリーダーボードで結果が表示されます。  
使用されるテストデータ（非公開）も完全に別れているため、パブリックリーダーボードのデータに有効な手法が、プライベートリーダーボードのデータにも有効とは限りません。

そのためロバスト（堅牢）性の高い手法を取る必要があるということで、
僕はそれができていなかったということです😭

<br>

## 使用したツールなど

コンペティションに取り組むにあたって、以下のツールを活用しました。

### 情報収集

---

#### Kaggleデータのスクレイピング
- コンペティション概要、データセット説明、パブリックノートブック、ディスカッションをスクレイピング
- 収集した情報をMarkdown形式に変換

<br>

#### Google AI Studio
- 収集した情報をコンテキストとして渡し、以下について質問：
  - コンペティションの説明
  - よく使われている手法
  - 参考になる過去のコンペ
  - その他わからないことや気になることなど相談相手のような感じ
  

- 選択理由：コンテキストのtoken制限が100万と非常に大きい（前までモデルによっては200万と記載されているモデルもありましたが、今は書かれていないので、わかりません）
  - ChatGPTのデスクトップアプリやClaudeのWeb UIではもっと制限が小さい
  - スクレイピングで大量のテキストを抽出したため、Google AI Studio以外では対応困難

<br>

### モデル学習

---

#### Google Colaboratory

- 32Bモデルなどの学習には、Kaggle NotebookではGPUメモリ不足になるため、Google Colabを利用
- 課金が必要だが、手軽に使える
- Colab Pro+プランはバックグランドでの実行が可能なので、実行中にPCを起動したままにしておく必要がない
- google driveをmountして使える
- ブラウザ上でterminalを起動して操作することも可能です

課金してハイスペックなGPUで高めのスペックのモデルを学習でもさせないと、自分の実力ではメダルを取れないと判断しました。（取れませんでしたが😂）  
Colab Pro+プランで、1か月あたり ￥5,767で600コンピューティングが付与され、  
A100GPUを搭載した以下の高性能なスペックのサーバーを1時間あたり5.37コンピューティング（約52円）ほどで利用できます。  
GPUの利用料としてはかなり安いと思います。
 
```
RAM: 83.5 GB  
GPU RAM: 40.0 GB  
ディスク: 235.7 GB  
```
無料でもGPU（T4GPU 16GB GPUメモリ）は使えますので、LLMを試してみるであったり、ちょっとした利用に手軽でおすすめです。

<br>

### 利用した技術

---

#### モデル学習に利用
- **PEFT（Parameter-Efficient Fine-Tuning）ライブラリ**  
LLMを全パラメータ更新せずに、必要な部分だけ学習して性能を引き出す機能を複数持っています。
大きなモデルの全パラメータ（数百億～数千億）を更新するとコストもGPUも重すぎるため、
工夫して「ごく一部だけ」更新する方法を持っています。
以下のLoRAやQLoRAもPEFTで実施できます。

- **LoRA（Low-Rank Adaptation）**  
大規模言語モデルのパラメータは数億から数十億のパラメータがありますので、フルチューニングすると、とてつもないメモリが必要になります。
それを **ごく少ない追加パラメータだけ学習してモデルを特定タスクに適応させる（Fine-tuning）** ための方法です。<br><br>
通常の重み行列 W（d × k）を全部更新する代わりに、<br><br>
W’ = W + ΔW<br><br>
とし、 この ΔW を低ランク近似した行列で表し、元のパラメータにadapterとして足すことでパラメータを更新します。
例えば、4096×4096（= 約1677万）のパラメータがあるとします。
これをrank=4で（4096, 4）と（4, 4096）の行列を学習させます。この二つの行列の積をとると同じ4096×4096で同じサイズの行列ができます。
しかし、パラメータは16,384 x 2（=約3.2万）になるので、約500分の1のパラメータの学習のみに削減することができます。


- **QLoRA（Quantized LoRA）**  
QLoRA（量子化＋LoRA）だとさらに効率化され、<br>
LoRA に 4bit量子化（NF4）を組み合わせたもので、<br>
・ 元モデル：4bit に圧縮（VRAM 約1/4）<br>
・ 学習：LoRA 部分のみ 16bit  <br>
することにより、メモリの使用を抑えることができます。

今回は利用しませんでしたが、**Unsloth**というライブラリも存在します。こちらの方がより早く、メモリ効率もいいそうですが、対応していないモデルもあるそうです。
PEFTがHugging Face公式なのもあり、Transformers互換のモデルになんでも対応しているそうで、Unslothはサードパーティということでした。

中にはハイスペックで超高価なGPUをいくつも使ってフルチューニングしているセレブもいます。

- **PEFTでQLoRAを利用して学習するサンプル** 
```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import torch

# モデルとトークナイザーの読み込み
model_name = "Qwen/Qwen2.5-32B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # パディングトークンの設定

# 4bit量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 量子化モデルの読み込み
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# QLoRA用にモデルを準備
model = prepare_model_for_kbit_training(model)

# LoRA設定
lora_config = LoraConfig(
    r=16,  # LoRAのランク（低いほどパラメータが少ない）
    lora_alpha=32,  # LoRAのスケーリングパラメータ（rの倍に設定することが多い）
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 適用するレイヤー
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# データセットの準備（例）
def format_prompt(example):
    """プロンプトをフォーマットする関数"""
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    return {"text": prompt}

# データセットのロードとフォーマット
dataset = Dataset.from_dict({
    "instruction": [
        "以下のテキストからデータ引用を分類してください。\nテキスト: The dataset is available at https://doi.org/10.1234/data",
        "以下のテキストからデータ引用を分類してください。\nテキスト: We analyzed data from GSE12345",
        # ... 他のデータ
    ],
    "response": [
        "Primary",
        "Secondary",
        # ... 他のラベル
    ]
})
dataset = dataset.map(format_prompt)

# トレーニング引数の設定
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",  # 8bitオプティマイザー
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
)

# SFTTrainerの作成と学習実行
# SFTTrainerは、トークナイズとデータコレクションを自動で処理してくれます
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,  # LoRA設定を直接指定
    dataset_text_field="text",  # データセットのテキストフィールド名
    max_seq_length=512,  # 最大シーケンス長
    tokenizer=tokenizer,
)

trainer.train()

# 学習済みモデルの保存
trainer.save_model("./qwen2.5-32b-lora")
tokenizer.save_pretrained("./qwen2.5-32b-lora")
```

#### 推論に利用

- **vLLM（Very Large Language Model Serving）ライブラリ**  
大規模言語モデルを “高速・低メモリ・大スループット” で推論するためのサーバーエンジン です。
今の LLM 推論基盤としては 事実上の標準 になっており、量子化・高速化・LoRA マージに強いなどの特徴を持っています。

- **vLLMの実行サンプル**  
```python
from vllm import LLM, SamplingParams

# モデルの読み込み（AWQ量子化モデルなど）
model_path = "Qwen/Qwen2.5-32B-Instruct-AWQ"

# vLLMでモデルを初期化
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    max_model_len=4096,  # 最大コンテキスト長
    gpu_memory_utilization=0.9,  # GPUメモリ使用率
    tensor_parallel_size=1,  # テンソル並列化（複数GPUの場合は調整）
    dtype="auto"
)

# LoRAアダプターをロードする場合
# llm = LLM(
#     model=model_path,
#     enable_lora=True,
#     max_lora_rank=16,
#     max_loras=1,
# )
# llm.load_lora_weights("./qwen2.5-32b-lora")

# サンプリングパラメータの設定
sampling_params = SamplingParams(
    temperature=0.1,  # 生成のランダム性、0.1はより確定的な出力
    top_p=0.9,  # Nucleus Sampling、確率の累積が90%に入る範囲の単語だけを候補にする
    max_tokens=512,  # 最大生成トークン数、モデルが 512 トークン生成したら生成を強制終了
    stop=["###", "\n\n\n"]  # 停止トークン、特定の文字列が出現したら、そこで生成を止める
)

# プロンプトの準備
prompts = [
    "### Instruction:\nデータ引用の分類を行います。以下のDOIがデータ引用かどうか判定してください。\n\n### Input:\nDOI: 10.1234/example\n\n### Response:",
    "### Instruction:\nデータ引用の分類を行います。以下のDOIがデータ引用かどうか判定してください。\n\n### Input:\nDOI: 10.5678/sample\n\n### Response:",
]

# バッチ推論の実行
outputs = llm.generate(prompts, sampling_params)

# 結果の取得
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt {i+1}:")
    print(f"Generated: {generated_text}\n")
```

<br>

## 結果

### 最終スコア

- **Public Leaderboard**: 80位
- **Private Leaderboard**: 467位

最終順位が**387位下落**という、非常に残念な結果となりました。

### 原因
コンペ終了後は、過去に提出したNotebookのPrivate Leaderboardのスコアを確認できますので、 スコアの遍歴から、途中でガクンとスコアが落ちているのを発見しました。

**問題の特定**
- XML、PDFをテキスト化したデータからAccession IDを抽出するために追加した正規表現が原因
- Public Leaderboardでは少しスコアが良くなったが、Private Leaderboardでは大幅にスコアを悪化させていた

**正規表現を追加していなかったら**
- コンペ終了後もNotebookを提出してスコアを確認できるため、問題の正規表現を削除したバージョンを提出すると、85位相当のスコアとなりました
- コンペ終了の**1ヶ月近くも前**に追加した正規表現だったため、それ以降の取り組みは何だったのか。。

<br>

# おわりに
最終的な順位は残念な結果となりましたが、多くの学びがありました。  
次のコンペとして、[AI Mathematical Olympiad - Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/overview)というLLMで数学の問題を解くコンペに参加していますので、  
なんとか最後までやりきって、また取り組み内容を共有したいと思っています！

<br>

# おまけ
どうしてもメダルを取りたいということで、別のコンペにも参加しました。  
コンペに関する内容は割愛しますが、[Jigsaw - Agile Community Rules Classification](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)というコンペで、169位で銅メダルを取れました。  
ただ、あまり何もしてなくて、パブリックノートの解法を複数アンサンブルしただけの他力本願メダルでした😅


<br><br><br>