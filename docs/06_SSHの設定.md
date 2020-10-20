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

## 確認
ファイルを保存したら、ターミナルからSSH接続をテストします。

```bash
# ssh Hostで指定したHostに接続できる
$ ssh server_one
```
