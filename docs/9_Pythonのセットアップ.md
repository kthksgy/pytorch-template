# Pythonのセットアップ
リモート環境にユーザーのPython環境を構築しましょう。

バージョン管理ツールとして**pyenv**を用いて**PyTorch**プロジェクト向けの**Python**環境を構築します。

以降の操作は、リモートのターミナル(`ssh Host`で繋いだ後でも、VSCodeで接続して`New Terminal`しても良い)で行ってください。

## 前提
私が管理しているサーバーではpyenvはユーザーに対して標準でインストールしているので、pyenvがインストールされた状態からスタートします。

今回は2020年10月時点の最新安定バージョンで環境の構築を行います。

- Python 3.8.6
- PyTorch 1.6.0 (Linux/CUDA版)

## 手順
順序として、ユーザー個別のPython環境を構築し、更にプロジェクト個別のPython環境を構築して、最後にPyTorchを含む諸々のパッケージをプロジェクト個別のPython環境にインストールします。

こうする事で、OSのPython環境はおろか、ユーザー個別のPython環境を改変する事なく、完全にプロジェクト個別のPython環境を構築できます。

### (必要に応じて)pyenvの更新
pyenvの更新を行います。

```bash
# pyenvは~/.pyenvにインストールされています
$ cd ~/.pyenv
# pyenvをgitを使って更新します
$ git pull
# ホームディレクトリに帰ります
$ cd
```

### ユーザーのPython環境構築
まずは、ユーザーのPython環境を構築します。

```bash
# インストール可能なPython 3.8のバージョンを確認
$ pyenv install --list | grep 3.8
# Python 3.8.6をインストール
$ pyenv install 3.8.6
# Python 3.8.6をデフォルトのPython環境として指定
$ pyenv global 3.8.6
# Pythonのバージョンが正しく設定されたかの確認
$ python --version
```

### プロジェクトのPython環境構築
次に、Python標準の**venv**を用いてプロジェクトのPython環境を構築します。

venvは元となるPythonを単純に複製するので、venvで作成されるPython環境のバージョンはユーザーのPython環境に依存します。

```bash
# プロジェクトのディレクトリに移動
$ cd my_python_project
# venvを使って.venvディレクトリにPython環境を構築
$ python -m venv .venv
```

### 【重要】プロジェクトのPython環境の有効化
デフォルトではpyenvで作成したユーザーのPython環境を参照するので、この状態でパッケージのインストールを行うと、ユーザーのPython環境を改変してしまいます。

Pythonに関する諸々の操作を行う前に、プロジェクトのPython環境を有効化する必要があります。

この操作は、`source`コマンドを実行してから`deactivate`コマンドを実行するかログアウトするまで有効で、新規にSSH等でログインした際はその都度実行する必要があります。

```bash
# .venv/bin/activateでプロジェクトのPython環境を有効化
$ source .venv/bin/activate
# pythonコマンドが.venv内のPythonを参照するようになったかを確認
$ which python
$ which pip
```

### PyTorch等のパッケージのインストール
`pip`を使ってパッケージをインストールします。

必ず、実行前に**プロジェクトのPython環境が有効になっているかを確認**してください。

```bash
# (初回のみ)pipの更新
$ python -m pip install --upgrade pip
# (初回のみ)wheelのインストール
$ pip install --upgrade wheel
# PyTorch(とその追加パッケージのTorchVision)のインストール
$ pip install torch torchvision
```

以下のパッケージは、PyTorchに直接関係ありませんが便利なパッケージやプログラムが少し豪華になるパッケージです。

必要になったらインストールしてください。

```bash
# 自動コードレビューパッケージ(文法のチェックとかをやってくれる)
$ pip install flake8
# グラフ描画パッケージ
$ pip install matplotlib
# プログレスバー([|||..]: 60/100みたいなやつ)の表示パッケージ
$ pip install tqdm
# 画像処理パッケージ(OpenCV)
$ pip install opencv-python
```