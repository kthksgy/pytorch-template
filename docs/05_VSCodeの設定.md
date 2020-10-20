# VSCodeの設定

## PythonのLinter(自動コードレビュー)の選択
コマンドパレットを開き(Windows: `Ctrl + Shift + P`, Mac: `Command + Shift + P`)、`Preferences: Open Settings (JSON)`を選択してください。

```json
{
}
```

とだけ書かれたファイルが開かれるので、`{`と`}`の間に`"python.linting.flake8Enabled": true`と`"python.linting.pylintEnabled": false`を追加します。

ファイル全体では以下のような形になります。辞書形式なので末尾のコンマを忘れずに。

```json
{
    "python.linting.flake8Enabled": true,
    "python.linting.pylintEnabled": false,
}
```

書き加えたら、保存(Windows: `Ctrl + S`, Mac: `Command + S`)してください。

## (Windowsのみ)デフォルトシェルの設定
Windowsのみ追加で、VSCodeがデフォルトシェルとしてGit Bashを開くように設定します。

先ほどと同じく`"terminal.integrated.shell.windows": "C:\\Program Files\\Git\\bin\\bash.exe"`を追加して保存します。

これで、VSCode上部のメニューから`Terminal -> New Terminal`した時に、コマンドプロンプトではなくGit Bashが開かれるようになりました。

JSONファイル全体ではこのようになります。

```json
{
    "terminal.integrated.shell.windows": "C:\\Program Files\\Git\\bin\\bash.exe",
    "python.linting.flake8Enabled": true,
    "python.linting.pylintEnabled": false,
}
```

## 拡張機能のインストール
VSCodeの画面左のボタンから、`Extensions`を開きます。

検索窓で検索して、以下の拡張機能をインストールしてください。

- `Python`
- `Remote Development`

Remote Developmentをインストールすると、画面左にRemote Developmentのボタン(`Remote Explorer`)が追加されます。
