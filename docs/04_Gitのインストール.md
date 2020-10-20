# Gitのインストール
バージョン管理システムGitを使って、プロジェクトのバックアップを行います。

## (Windowsのみ)Git for Windowsのインストール
この手順はWindowsのみ実行してください。

[Git for Windows 公式サイト](https://gitforwindows.org/)の*Download*ボタンからセットアップファイル`Git-2.28.0-64-bit.exe`をダウンロードして実行してください。

画面に従ってインストールします。*Choosing the default editor used by Git*以外は殆どデフォルトで構いません。

### Select Components
以下にチェックを入れて先に進んでください。

- Git LFS
- Associate .git* configuration files with the default text editor
- Associate .sh files to be run with Bash

### Choosing the default editor used by Git
Gitを使う上でテキストエディタが必要になるので、先ほどインストールしたVSCodeを選択します。

- Use Visual Studio Code as Git's default editor

### Adjusting your PATH environment
Git for Windowsでインストールされる諸々をコマンドプロンプト等から利用できるようにします。

(Recommended)としてデフォルトで選択されているので、そのまま進みます。

- Git from the command line and also from 3rd-party software

### Choosing HTTPS transport backend
HTTPS接続時に使用するライブラリの選択です。

デフォルトのまま進みます。

- Use the OpenSSL library

### Configuring the line ending conversions
WindowsとLinuxで改行コードが違うので、それをどのように扱うかの設定です。

デフォルトのまま進みます。

- Checkout Windows-style, commit Unix-style line endings

### Configuring the terminal emulator to use with Git Bash
ターミナルエミュレータの設定です。

デフォルトのまま進みます。

- Use MinTTY (the default terminal of MSYS2)

### Choose the default behavior of `git pull`
`git pull`コマンドの挙動の設定です。

デフォルトのまま進みます。

- Default (fast-forward or merge)

### Choose a credential helper
認証情報の保存のヘルパの選択です。

デフォルトのままでも良いですが、クロスプラットフォーム版を選択します。

- Git Credential Manager Core

### Configuring extra options
デフォルトのまま進みます。

### Configuring experimental options
デフォルトのまま進みます。

### 確認
`Git Bash`を起動して、Gitのインストールが完了したかを確認しましょう。

```bash
$ git --version
git version 2.28.0.windows.1
```

## (Macのみ)xcode-selectのインストール
この手順はMacのみ実行してください。

Macでは、Command Line ToolsをインストールするとGitも自動的にインストールされます。

以下のコマンドを実行すると、ポップアップでCommand Line Toolsをインストールするかを聞かれるのでインストールします。

```zsh
% xcode-select --install
```

### インストール出来ない場合
既にインストールされていると以下のようなエラーが出ます。

```zsh
% xcode-select --install
xcode-select: error: command line tools are already installed, use "Software Update" to install updates
```

また、OSのバージョンによってはインストールに失敗する場合もあるようです。

加えて、Appleのサーバーの状況によってインストールに失敗する場合もあります。

その場合、[More Downloads for Apple Developers](https://developer.apple.com/download/more/)からインストーラを手動でダウンロードして実行する必要があります。

### 確認
ターミナルを起動して、Gitのインストールが完了したかを確認しましょう。

```zsh
% git --version
git version 2.24.3 (Apple Git-128)
```

## Gitの設定
先ほどGitHubで取得したアカウントのユーザー名と専用のメールアドレスを設定します。

```bash
$ git config --global user.name "ユーザー名"
$ git config --global user.email "数列+ユーザー名@users.noreply.github.com"
```
