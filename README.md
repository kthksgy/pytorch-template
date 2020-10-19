# PyTorch Template
研究室向けに作ったPyTorchプロジェクトのテンプレートです。

必要に応じて項目を増やしたり減らしたりしてください。

## 必要環境
ここには実行に必要なライブラリ等を書きます。

```
python == 3.8.6
torch == 1.6.0
torchvision == 0.7.0
```

## 実行方法
ここには実行方法を書きましょう。

例えばこんな感じで書きます。

```bash
# ヘルプを表示
$ python main_classification.py --help
$ python main_dcgan.py --help

# 分類モデルを実行
$ python main_classification.py -b 1000 -e 10 --dataset mnist

# 生成モデルを実行
$ python main_dcgan.py -b 1000 -e 25 --dataset mnist
```

## 参考
参考にした文献等を此処にメモしておくと便利です。

もしくは、`refs`ディレクトリを作ってリンクを記録していくのもアリです。

### Webページ
#### 便利ツール
- [DeepL翻訳](https://www.deepl.com/ja/translator)
- [Langsmith Editor](https://editor.langsmith.co.jp/)
- [Write your best with Grammarly.](https://www.grammarly.com/)
- [Abbreviations and acronyms dictionary](https://www.acronymfinder.com/)
- [arXiv.org](https://arxiv.org/)

#### PythonやGitについて
- [PEP8](https://pep8-ja.readthedocs.io/ja/latest/)
- [TODO: 以外のアノテーションコメントをまとめた](https://qiita.com/taka-kawa/items/673716d77795c937d422)
- [Python命名規則一覧](https://qiita.com/naomi7325/items/4eb1d2a40277361e898b)
- [オープンソースライセンス、どれなら使っても良いの？？](https://qiita.com/fate_shelled/items/a928709d7610cee5aa66)

#### PyTorchやGANについて
- [GAN(Generative Adversarial Networks)を学習させる際の14のテクニック](https://qiita.com/underfitting/items/a0cbb035568dea33b2d7)
- [CUDA SEMANTICS - PyTorch](https://pytorch.org/docs/stable/notes/cuda.html)

#### Twitterアカウント
- [arXiv.org](https://twitter.com/arxiv)
- [arxiv](https://twitter.com/arxiv_org)

### リポジトリ
- [soumith/ganhacks - GitHub](https://github.com/soumith/ganhacks)
- [hindupuravinash/the-gan-zoo - GitHub](https://github.com/hindupuravinash/the-gan-zoo)

### 論文
ここには参考になった論文を貼ります。

Google Scholarから生成したCitationやBibtexのコードを貼っておくと良いかも。

#### [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)
GeneratorとDiscriminatorでそれぞれ異なる学習率を適用する。