# Python
このドキュメントには、Pythonでの開発のTipsを書きます。

基本的には、Flake8が自動でコードレビューをしてくれるので、文句を言われた箇所を修正していけばある程度の可読性は保てると思います。

## ディレクトリ構造
Pythonプロジェクトに限りませんが、ソースコードのファイルを階層的に配置するように心がけて、全てのファイルが1つのディレクトリに入っている状況にならないよう注意してください。

プロジェクト直下のディレクトリには実行可能ファイルとプロジェクトの`README.md`とライセンスファイルがあり、それ以外のファイルはサブディレクトリに入っている状態を目指してください。

とりあえずはこのリポジトリを参考にすると良いと思いますが、GitHubで他のPythonプロジェクトを見に行くと更に良いです。

## コメント
無意味なコメントをしないように気を付けましょう。

```python
# BAD
flag = 1  # flagに1を代入

# GOOD
flag = 1  # 動作フラグに1を設定

if flag == 1:
    do_some_operations()
```

## 変数名/関数名
変数名や関数名に意味を持たせましょう。

```python
# BAD
a = 50
b = 1.5
def func1(a1, a2):
    return a1 / a2 / a2

# GOOD
weight_kg = 50
height_m = 1.5
def calc_bmi(weight, height):
    return weight / height / height
```

## 命名規則
変数名や関数名に関する規則を命名規則と呼びます。

最低限、以下を守ればOKです。

```python
# 変数/関数
#   スネークケース(snake_case)
a, b = 1, 2

def add(a, b):
    return a + b

a_plus_b = add(a, b)

# 定数(プログラムの最初から最後まで値が変わらない変数)
#   アッパースネークケース(UPPER_SNAKE_CASE)
PROGRAM_NAME = 'My Application'

# クラス
#   アッパーキャメルケース(UpperCamelCase)
class MyClass:
    def __init__(self):
        pass
```

他にも、プライベート変数等の概念があります。

興味があれば[PEP8のコードスタイルガイド](https://pep8-ja.readthedocs.io/ja/latest/)を見たり、「Python 命名規則」で検索してみると良いと思います。

## プリントとログ
プリント(`print()`)とログ(`logger.*()`)は、表示する内容が「ユーザーに見せたい結果」か「プログラムの動作ログ」かで使い分けてください。

また、ログ出力のレベル(`.debug()`, `.info()`, ...)は必要に応じて使い分けてください。

使い分けについては、[Logging HOWTO](https://docs.python.org/ja/3.8/howto/logging.html)が参考になります。

```python
# やや過剰にデバッグ出力を入れています。
a = 1
b = 2
logger.info('変数aとbを定義しました。')

def add(a, b):
    logger.debug('a + bを計算します。')
    res = a + b
    logger.debug('a + bを計算しました。')
    return a + b

logger.debug('add()を定義しました。')

logger.debug('add()を実行します。')
a_plus_b = add(a, b)
logger.info('add()を実行しました。')

print(f'{a}と{b}を足した結果は{a_plus_b}です。')
```

具体例はサンプルコードを見てください。興味が出たら「Python ロギング」や「ロギング」で検索すると良いと思います。

## ドキュメント
プロジェクトの可読性を上げるためには、ドキュメントがあると良いです。

最低限、引数の型くらいは明記しておいて、余裕があればDocstringを記述しましょう。Docstringから自動でドキュメントを生成するアプリケーションもあり、後々便利です。

Docstringには厳密には色々スタイルがありますが、Google Styleを簡略化した形で書くのが個人的に楽で良いと思います。

```python
def calc_bmi(weight: float, height: float) -> float:
    '''
    BMIを計算します。
    Args:
        weight: 体重[kg]
        height: 身長[m]
    Returns:
        BMI
    '''
    return weight / height / height
```

型の明記についてはこのリポジトリのコードでも行っているので、参考にしてください。

興味があれば、「Python Docstring Google Style」とかで調べてください。
