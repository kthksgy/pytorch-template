# エディタのインストール
このドキュメントでは、Windows/MacでLinux(Ubuntu)サーバー上で開発するための環境を作ります。

バージョンは2020年10月時点の物なので、適宜読み替えてください。

既に実行した事のある手順は飛ばしてください。

## [Visual Studio Code](https://azure.microsoft.com/ja-jp/products/visual-studio-code/)のインストール

[Download Visual Studio Code](https://code.visualstudio.com/download)から、自分のOS用のファイルをダウンロードしてください。

### (Windowsのみ)Installerの選択とインストール
Windowsについてはダウンロードボタンを押すと自動で`User Installer`がダウンロードされてしまうので、ボタン下の`System Installer`から自分のアーキテクチャ(多くの場合64bit)用のインストーラをダウンロードしてください。

管理者権限が無い場合のみ、`User Installer`をダウンロードしてください。

ダウンロードが完了したら、インストールしてください。インストール時のオプションはデフォルトのままで構いません。

`サポートされているファイルの種類のエディターとして、Codeを登録する`にチェックを入れると便利かもしれません。

### (Macのみ)ファイルの移動
Macでは、ダウンロードした`Visual Studio Code.app`を`/Applications/`に移動するとインストールできます。

```zsh
% sudo mv ~/Downloads/Visual\ Studio\ Code.app /Applications/
```

移動したら、`Launchpad`からVisual Studio Codeを起動できる事を確認してください。
