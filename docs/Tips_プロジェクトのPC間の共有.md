# プロジェクトのPC間の共有
今回はプロジェクトをPC間で共有する方法として、GitHubを使います。

「プロジェクトのバックアップ」でプロジェクトをGitHub上にプッシュして、それを別のPCでプルします。

```bash
# PC A
$ git status
$ git add some_files.txt
$ git commit
$ git push
```

プロジェクトをまだクローンしていない場合はクローンします。

```bash
# PC B
$ git clone https://github.com/ユーザー名/リポジトリ名.git
```

クローンしたプロジェクトがある場合は、そのディレクトリでプルします。

```bash
# PC B
$ cd リポジトリ名
$ git pull
```

## Pythonパッケージの共有
pipでインストールしたPythonパッケージは、以下のコマンドでバージョン付きで書き出す事が出来ます。

プロジェクトのPython環境が有効になっている事を確認してから行ってください。

```bash
$ pip freeze > requirements.txt
$ cat requirements.txt
```

`requirements.txt`を元にパッケージをインストールする時は以下のコマンドで行います。

異なるOS間ではエラーになる事が多いので、この方法は同じ環境でのみ行うようにしてください。

```bash
$ pip install -r requirements.txt
```
