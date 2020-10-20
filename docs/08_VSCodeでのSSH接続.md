# VSCodeでのSSH接続
6でSSHの設定をしたので、`Remote Explorer`の`SSH Targets`のリストに先ほど書いた`Host`が表示されたはずです。

リストの`Host`名の上にカーソルを移動すると、`Connect to Host in New Window`ボタンが現れるので、それを押すと、VSCodeのウィンドウがもう1つ立ち上がり、SSH接続が行われます。

必要に応じて、パスワードの入力やSSH鍵のパスフレーズの入力が求められます。

## リモートのターミナルを開く
新しく立ち上がったVSCodeのウィンドウ(SSHでリモートに接続しているウィンドウ)で、画面上部のメニューから`Terminal -> New Terminal`すると、リモートのターミナルが開かれます。
