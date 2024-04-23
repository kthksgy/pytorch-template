※ 2024-04-23: GitホスティングサービスにSSHで接続する都合で、設定を追加しました。

# SSHの設定
SSHの設定は、Windowsは`C:\Users\ユーザー名\.ssh\config`、Macは`/Users/ユーザー名/.ssh/config`に記述します。

VSCode上からは、`Remote Explorer`(左メニュー)を選択し、リスト上部の`Containers`を`SSH Targets`に変更した状態で、リスト上にカーソルを移動すると`Configure`(歯車のボタン)を押すと開く事が出来ます。

以下のように、SSHの設定をサーバーごとに並べて記述します。

デフォルトや未設定の項目を書く必要はありません。

- `Host`: 設定名(サーバー名)を指定します。
- `HostName`: サーバーのIPを入力します。
- `Port`: ポート番号(デフォルト: `22`)を指定します。
- `User`: そのサーバーでのユーザー名を指定します。
- `IdentityFile`: 認証に使う秘密鍵のファイルを指定します。

```
Host server_one
  HostName 111.111.111.111
  User サーバー1でのユーザー名
Host server_two
  HostName 222.222.222.222
  Port 22222
  User サーバー2でのユーザー名
  IdentityFile ~/.ssh/id_rsa
```

## GitHubやGitLabのための設定項目を追加する

SSH経由でGitHubやGitLabを利用する(クローン、プッシュ、プルを行う)ために、少し順番が前後してしまいますが、先に設定を追加します。

以下の内容を設定ファイルに追加してください。

```
Host git-github.com
  HostName github.com
  User GitHubにおけるあなたのユーザー名

# GitLabも使う場合はこの設定もしておくと便利。
Host git-gitlab.com
  HostName gitlab.com
  User GitLabにおけるあなたのユーザー名

# ワイルドカードを用いると共通設定をまとめて行える。他にもカンマで対象を列挙する事も出来る。
# ここでは上記2つの設定が対象になるように、『`git-*`』で絞り込みをしている。
# 逆に言うと『`git-*`』で絞り込めるように、設定名の接頭辞として`git-`を用いている。
Host git-*
  # この秘密鍵ファイルは後の項目で作成する。
  IdentityFile ~/.ssh/id_ed25519
  TCPKeepAlive yes
  IdentitiesOnly yes
```

## 確認
ファイルを保存したら、ターミナルからSSH接続をテストします。

```bash
# ssh Hostで指定したHostに接続できる
$ ssh server_one
```
