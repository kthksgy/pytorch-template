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
