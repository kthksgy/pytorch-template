# GitHubのリポジトリ作成
プロジェクトを保存する場所を確保するため、GitHub右上の「+」ボタンから`New repository`を選択します。

画面に従って必要事項を記入します。

## Owner
自分のユーザーを選びます。

## Repository name
プロジェクト名を入力します。意味の無い文字列(`project`や`a`等)は避けてください。

個人的には小文字のチェインケース(`chain-case`)で入力するのが好きです。

プロジェクトの内容を表した名前を考えるのが一番ですが、迷った人は卒業研究なので以下のようなプロジェクト名で良いと思います。

- `bachelor-thesis`: 学士論文
  - 修士に進んだ時 `master-thesis`: 修士論文
- `bachelor-study`: 学士の研究
  - 修士に進んだ時 `master-study`: 修士の研究
- `b4`: B4の集大成のプロジェクトを表す
  - 修士に進んだ時 `m2`

余談ですが、修士に進む人が「卒業」という単語を使うと修論の時に困ります。

- `graduate-study`, `graduation-research`: 卒業研究
- `graduation-thesis`: 卒業論文

## Description (optional)
リポジトリの説明を入力します。

簡潔に、リポジトリの内容が分かるようにしてください。

迷った人は、`卒業研究`や`卒業研究の深層学習プロジェクト`のような感じで良いと思います。

## Public or Private
Privateにチェックを入れてください。

## Initialize this repository with:
以下にチェックを入れてください。

- Add a README file
- Add .gitignore
- Choose a license

### Add a README file
チェックを入れると`README.md`ファイルを自動生成します。

### Add .gitignore
チェックを入れたら、`.gitignore template:`から`Python`を選んでください。

`.gitignore`は、バックアップを取らないファイルの拡張子やディレクトリを設定するファイルで、Pythonを選んでおくとPythonの実行時のキャッシュファイル等は自動的にバックアップに含めなくなって便利です。

これをベースに、自分でバックアップしないファイル(動作ログや実行結果等)を追記していきます。

### Choose a license
一般公開予定は無いですが、一応ライセンスファイルも含めておきましょう。

`License:`から`Apache License 2.0`を選んでください。

## Create repositoryを押す
ボタンを押すと、リポジトリが作成されます。

作成されたリポジトリのURLは、`https://github.com/ユーザー名/リポジトリ名`になります。
