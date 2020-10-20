# SSHの鍵生成とGitHubへの鍵登録
GitHubで作成したプライベートリポジトリにアクセスするために、アクセスしたいPCの鍵を登録する必要があります。

ローカルはターミナル(WindowsではGit Bash)を開いて以下の操作を、リモートはSSH接続(`ssh server_one`)してから、リモートのターミナル上で操作してください。

## 鍵生成
GitHubの認証に使う公開鍵と秘密鍵のペアを作成します。

RSA鍵ではなく現状最強と言われるEd25519鍵を作成します。

パスフレーズは、秘密鍵を使うのに必要となるパスワードなので、念のため設定しておきましょう。

```bash
$ ssh-keygen -t ed25519
Enter file in which to save the key (~/.ssh/id_ed25519): (何も入力せずEnter)
Enter passphrase (empty for no passphrase): パスフレーズを入力
Enter same passphrase again: 同じパスフレーズをもう一度入力
```

実行が終わると、ホームディレクトリ直下の`.ssh`ディレクトリ(`~/.ssh/`)に`id_ed25519`(秘密鍵)と`id_ed25519.pub`(公開鍵)が生成されます。

## GitHubへの鍵登録
1で行ったのと同じように、GitHubの画面右上のアイコンから`Settings`を開きます。

左カラムから今度は`SSH and GPG keys`を開きます。

`SSH keys`右の`New SSH key`ボタンを押して、`Title`と`Key`を入力します。

### Title
鍵の登録名を入力します。分かりやすい名前を設定しましょう。

`Server01, my_user@svr01`のように、鍵を生成したコンピュータやサーバーの名前とユーザー名にするのが良いと思います。

### Key
公開鍵の中身をコピーします。

**【重要】間違っても秘密鍵の中身をコピーしないようにしてください。**

```bash
$ cat ~/.ssh/id_ed25519.pub
# 出て来た内容を全部コピーしてKeyに貼り付け
```

### Add SSH Key
ボタンをクリックして鍵を登録します。
