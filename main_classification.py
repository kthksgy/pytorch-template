# 好みですがモジュールのアルファベット順に書きます
# 必要に応じて直接インポートを行います
# 標準モジュール
import argparse
import csv
# クラスや関数の直接インポートは、
# from a import Bで書きます
from datetime import datetime
# ログを取るためのモジュール
from logging import (
    getLogger, basicConfig,
    DEBUG, INFO, WARNING
)
from pathlib import Path
import random
from time import perf_counter
from tqdm import tqdm

# 追加パッケージ(pipでインストールしたパッケージ)
import numpy as np

# PyTorchパッケージとそのサブモジュール
import torch
# サブモジュールの直接インポートは、
# import a.b as Cで書きます
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

# 自作パッケージ
from models.classification32 import Classification32
from utils.device import AutoDevice
from utils.transforms.common import Normalize

# コマンドライン引数を取得するパーサー
#   プログラムに関するヘルプ等を一括で作ってくれて、
#   これだけでなんかプログラムっぽくなる
#   【重要】引数usage, add_helpを触ると
#    使用法やヘルプの自動生成が行われないので注意
parser = argparse.ArgumentParser(
    prog='PyTorchサンプルプログラム01',
    description='PyTorchを用いてMNIST/FashionMNISTの画像分類を行います。'
)

# 訓練に関するコマンドライン引数
parser.add_argument(
    '-b', '--batch-size', help='バッチサイズを指定します。',
    type=int, default=250, metavar='B'
)
parser.add_argument(
    '-e', '--epochs', help='エポック数を指定します。',
    type=int, default=25, metavar='E'
)
parser.add_argument(
    '--dataset', help='データセットを指定します。',
    type=str, default='mnist', choices=['mnist', 'fashion_mnist']
)
parser.add_argument(
    '--data-path', help='データセットのパスを指定します。',
    type=str, default='./data'
)
parser.add_argument(
    '--padding', help='画像をゼロパディングして32画素×32画素にします。',
    action='store_true'
)
parser.add_argument(
    '--seed', help='乱数生成器のシード値を指定します。',
    type=int, default=42
)
# モデルの保存に関するコマンドライン引数
parser.add_argument(
    '--save', help='訓練したモデルを保存します。',
    action='store_true'
)
parser.add_argument(
    '--save-interval', help='モデルの保存間隔をエポック数で指定します。',
    type=int, default=10,
)
parser.add_argument(
    '--load', help='訓練したモデルを読み込みます。',
    type=str, default=None
)
# テストに関するコマンドライン引数
parser.add_argument(
    '--test', help='訓練を行わずテストのみ行います。',
    action='store_true'
)
# PyTorchに関するコマンドライン引数
parser.add_argument(
    '--disable-cuda', '--cpu', help='GPU上で計算せず、全てCPU上で計算します。',
    action='store_true'
)
parser.add_argument(
    '--info', help='ログ表示レベルをINFOに設定し、詳細なログを表示します。',
    action='store_true'
)
parser.add_argument(
    '--debug', help='ログ表示レベルをDEBUGに設定し、より詳細なログを表示します。',
    action='store_true'
)
# parse_args()でコマンドライン引数を変数に格納する
args = parser.parse_args()

# 結果を出力するために起動日時を保持する
LAUNCH_DATETIME = datetime.now()

# ロギングの設定
#   NOTE: Python 3.9では少し設定が増えるのでバージョンに注意する。
#   https://docs.python.org/ja/3.8/library/logging.html#logging.basicConfig
#   format: ログの出力形式、以下の属性を指定してログの出力形式を指定する。
#     https://docs.python.org/ja/3.8/library/logging.html#logrecord-attributes
#   datefmt: 時間の出力形式、%(asctime)sにどのような形式で日時が入力されるかを指定する。
#     https://docs.python.org/ja/3.8/library/time.html#time.strftime
#   level: ログの出力レベル。どれくらい詳細なログを表示するかを設定する。
#     https://docs.python.org/ja/3.8/library/logging.html#levels
basicConfig(
    format='%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=DEBUG if args.debug else INFO if args.info else WARNING,
)
# ロガーの名前を指定してロガーを取得する
# NOTE: "生成"ではなく"取得"している。名前が被ると同一のロガーのインスタンスを得る。
# 以降はlogger.debug('Hello, Debug!')のような形でログを出力する
logger = getLogger('main')

# テスト時は何度エポックを回しても結果は同じなので、
# エポック数を1に強制する
if args.test:
    args.epochs = 1
    logger.info('テストモードで実行されているためエポック数を1に設定しました。')

# 結果を出力するためのディレクトリの作成
OUTPUT_DIR = Path(
    LAUNCH_DATETIME.strftime(
        f'./outputs/{args.dataset}/%Y%m%d%H%M%S'))
OUTPUT_DIR.mkdir(parents=True)
logger.info(f'結果出力用のディレクトリ({OUTPUT_DIR})を作成しました。')
if args.save:
    OUTPUT_MODEL_DIR = OUTPUT_DIR.joinpath('models')
    OUTPUT_MODEL_DIR.mkdir(parents=True)
    logger.info(f'モデル保存用のディレクトリ({OUTPUT_MODEL_DIR})を作成しました。')

# 【重要】乱数のシード値の設定(忘れると結果の再現性が無くなる)
random.seed(args.seed)  # Python標準のランダムシード
np.random.seed(args.seed)  # NumPyのランダムシード
torch.manual_seed(args.seed)  # PyTorchのランダムシード
# cuDNNは高速化に一部の処理が非決定的に行われる
# GPUで再現性を確保するために以下の2つを設定する
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.deterministic = True  # cuDNN(GPU)の動作を決定性に設定
torch.backends.cudnn.benchmark = False  # ベンチマークモードを切る
logger.info(f'ランダムシードを{args.seed}に設定しました。')

# デバイスについての補助クラスをインスタンス化
auto_device = AutoDevice(disable_cuda=args.disable_cuda)
# AutoDeviceを使って最適なデバイスを得る
device = auto_device()

# データセットのデータ(今回は画像)に行う処理のリスト
# 最後にtransforms.Compose(リスト)すると画像1枚1枚に対して
# 指定した処理が順番に行われる
tfs_list = []
# コマンドライン引数でパディングを指定していたらパディングを最初に行う
if args.padding:
    tfs_list.append(
        transforms.Pad(2, fill=0, padding_mode='constant')
    )

# 正規化を行う際の平均と標準偏差
MEAN_GRAYSCALE = (0.5,)
STD_GRAYSCALE = (0.5,)

# リスト同士の結合は.extend(リスト)
tfs_list.extend([
    # 画像をNumPy配列からテンソルに変換する
    # この時点で画素は0~1に正規化される
    transforms.ToTensor(),
    # 更に正規化を行う
    Normalize(MEAN_GRAYSCALE, STD_GRAYSCALE)
])
logger.info('画像のトランスフォームを定義しました。')


# データセットを読み込むための補助関数
# main.pyで定義した関数のtyping(型明記)やDocstringは最小限で良い
def load_dataset(name: str, transform=None):
    if isinstance(transform, (list, tuple)):
        transform = transforms.Compose(transform)
    if name == 'mnist':
        num_classes = 10
        trainset = dset.MNIST(
            root=args.data_path, download=True, train=True,
            transform=transform)
        testset = dset.MNIST(
            root=args.data_path, download=True, train=False,
            transform=transform)
    elif name == 'fashion_mnist':
        num_classes = 10
        trainset = dset.FashionMNIST(
            root=args.data_path, download=True, train=True,
            transform=transform)
        testset = dset.FashionMNIST(
            root=args.data_path, download=True, train=False,
            transform=transform)
    return trainset, testset, num_classes


# 補助関数でデータセットをロードする
trainset, testset, NUM_CLASSES = load_dataset(args.dataset, tfs_list)
# MNISTもFashionMNISTもグレイスケールなので最初の特徴数は1(RGBだとここが3)
NUM_FEATURES = 1
logger.info('データセットを読み込みました。')

# データローダーのインスタンス化
# これをイテレート(順に読み込む)する事でデータがバッチサイズごとに得られる
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True)

# テスト用のデータセットはシャッフルしない(しても意味が無いため)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False)

logger.info('データローダーを生成しました。')

# モデルの定義(今回は分類モデルなので入力特徴数とクラス数が必要)
model = Classification32(NUM_FEATURES, NUM_CLASSES)
logger.info('モデルを定義しました。')
# モデルをAutoDeviceによって自動選択したデバイスに転送
model = model.to(device)
logger.info(f'モデルを{device}に転送しました。')
# コマンドライン引数に指定されたファイルに保存された重み・バイアスをモデルに適用
if args.load is not None:
    model.load_state_dict(torch.load(args.load))
    logger.info(f'学習済みモデルを{args.load}から読み込みました。')

# オプティマイザ(最適化アルゴリズム)の定義
# SGDやRMSProp等色々あるがAdamが最も主流
optimizer = optim.Adam(model.parameters(), lr=0.01)
logger.info('オプティマイザを定義しました。')

# 損失関数の定義 ([英]Criterion: [日]基準)
# nn.CrossEntropyLoss()はソフトマックス関数と交差エントロピー計算が合わさった物
# そのため手動でモデルの出力にソフトマックス関数は適用しなくて良い
criterion = nn.CrossEntropyLoss()
logger.info('損失関数を定義しました。')

# 各エポックの学習の評価を保存するためにCSVファイルを作成する
f_results = open(
    OUTPUT_DIR.joinpath('results.csv'), mode='w', encoding='utf-8')
logger.info('結果出力用のファイルを開きました。')
# ファイルに書き込むためにはcsv.writer()を取得する必要がある
csv_writer = csv.writer(f_results, lineterminator='\n')
# ファイルにどのような項目を記録するかを自分で決める
result_items = [
    'Epoch', 'Train Loss Mean', 'Train Accuracy Mean', 'Train Elapsed Time',
    'Test Loss Mean', 'Test Accuracy Mean', 'Test Elapsed Time',
    'Saved File Name',
]
# ファイルの一行目に項目名を書く
# csv.writer()に.writerow()でリストを渡すと自動的にCSV形式で記述してくれる
csv_writer.writerow(result_items)
# バッファをフラッシュする
# ファイルに何か書き込む際には効率化のため、ある程度データを
# システムが自動的にバッファに溜めてから一度に書き込むため、すぐに結果を出力したい場合は
# 手動でフラッシュする必要がある
f_results.flush()

# それぞれの項目名が何番目の項目になったかを辞書に記録しておく
# こうするとCSVへの結果の出力が少し楽になると思う
csv_idx = {item: i for i, item in enumerate(result_items)}

# 訓練ループ中では何度も同じ処理を繰り返すのでログは最小限に留める
# どうしてもループ中にログ出力が必要な場合は.debug()で書くと良いかも
logger.info('訓練を開始します。')

# 訓練のループ
# 今回は訓練用の補助関数は定義していないのでループ内に訓練の処理を記述する
for epoch in range(args.epochs):
    # CSVファイルへの結果の初期化
    # ''(長さ0の文字列)で初期化する事で特定の項目を記録しないエポックが在った時に
    # 項目の列がズレないようにできる
    # このリストを.writerow()に渡してエポックの最後に書き込む
    results = ['' for _ in range(len(csv_idx))]
    # エポック数を記録する
    results[csv_idx['Epoch']] = f'{epoch + 1}'

    # コマンドライン引数でテストモードを指定していたら学習は行わない
    if not args.test:
        # 訓練モードと評価モードで動作の変わる層が存在するので必ず切り替える
        model.train()  # モデルを訓練モードに設定
        # tqdmで実行時に進捗を表示するプログレスバーを簡単に作れる
        pbar = tqdm(
            trainloader,
            desc=f'[{epoch + 1}/{args.epochs}] 訓練開始',
            total=len(trainset)//args.batch_size,
            leave=False)
        # エポックの最後に損失の平均と正解率の平均を求めるための変数を初期化
        losses, cnt_total, cnt_correct = [], 0, 0
        # 訓練時間を測定する
        begin_time = perf_counter()  # 時間計測開始
        for images, labels in pbar:
            # モデルと同じ場所に画像とラベルのテンソルを転送
            images = images.to(device)
            labels = labels.to(device)
            # 最初にモデルの勾配を初期化
            model.zero_grad()

            # 画像をモデルに入力して出力を得る
            outputs = model(images)
            # 損失(交差エントロピー)を計算する
            loss = criterion(outputs, labels)
            # 損失の誤差を逆伝播する
            loss.backward()
            # オプティマイザで逆伝播した勾配をモデルのパラメータに適用する
            optimizer.step()

            # 正解率の計算のための計算
            _, predicted = torch.max(outputs.data, 1)
            cnt_total += labels.size(0)
            cnt_correct += (predicted == labels).sum().item()

            # 平均損失を記録するためにこのバッチの損失をリストに記録
            losses.append(loss.item())
            # プログレスバーの損失表示を更新
            pbar.set_description_str(
                f'[{epoch+1}/{args.epochs}] 訓練中... '
                f'<損失: {losses[-1]:.016f}>')
        end_time = perf_counter()  # 時間計測終了
        pbar.close()

        # それぞれの項目について記録する
        train_loss_mean = np.mean(losses)
        results[csv_idx['Train Loss Mean']] = f'{train_loss_mean:.016f}'

        train_accuracy_mean = 100 * cnt_correct / cnt_total
        results[csv_idx['Train Accuracy Mean']] = f'{train_accuracy_mean:.03f}'

        train_elapsed_time = end_time - begin_time
        results[csv_idx['Train Elapsed Time']] = f'{train_elapsed_time:.07f}'

    # withを使って一時的に勾配計算をしないモードに移行
    # テスト時に勾配は必要無い(パラメータ更新を行わないため)ので、
    # 勾配計算を行わない事で計算資源を節約できる
    # テストの処理はパラメータ更新が無い以外は訓練の処理と同じ
    model.eval()  # モデルを評価モードに設定
    with torch.no_grad():
        pbar = tqdm(
            enumerate(testloader),
            desc=f'[{epoch + 1}/{args.epochs}] テスト開始',
            total=len(testset)//args.batch_size,
            leave=False)
        losses, cnt_total, cnt_correct = [], 0, 0
        begin_time = perf_counter()  # 時間計測開始
        for i, (images, labels) in pbar:
            images = images.to(auto_device())
            labels = labels.to(auto_device())
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            cnt_total += labels.size(0)
            cnt_correct += (predicted == labels).sum().item()
            pbar.set_description_str(
                f'[{epoch+1}/{args.epochs}] テスト中... '
                f'<損失: {losses[-1]:.016f}>')
        end_time = perf_counter()  # 時間計測終了
    test_loss_mean = np.mean(losses)
    results[csv_idx['Test Loss Mean']] = f'{test_loss_mean:.016f}'

    test_accuracy_mean = 100 * cnt_correct / cnt_total
    results[csv_idx['Test Accuracy Mean']] = f'{test_accuracy_mean:.03f}'

    test_elapsed_time = end_time - begin_time
    results[csv_idx['Test Elapsed Time']] = f'{test_elapsed_time:.07f}'

    # コマンドライン引数に応じて特定のタイミングでモデルの保存を行う
    if args.save and (
            (epoch + 1) % args.save_interval == 0
            or epoch + 1 == args.epochs):
        print(f'[{epoch+1}/{args.epochs}] モデルの保存中... ', end='')
        saved_file_name = OUTPUT_MODEL_DIR.joinpath(f'model_{epoch+1:06d}.pt')
        torch.save(model.state_dict(), saved_file_name)
        results[csv_idx['Saved File Name']] = saved_file_name
        print('<完了>')
    # 結果をファイルに書き込みフラッシュする
    csv_writer.writerow(results)
    f_results.flush()

    # 標準出力にも結果を表示
    # 文字列の後に文字列を続けて書くと自動的に結合される
    # 'あいう' 'えお' ⇒ 'あいうえお'
    if not args.test:
        print(
            f'[{epoch+1}/{args.epochs}] 訓練完了. '
            f'<訓練: (経過時間: {train_elapsed_time:.03f}[s/epoch]'
            f', 平均損失: {train_loss_mean:.05f}'
            f', 平均正解率: {train_accuracy_mean:.02f}[%])'
            f', テスト: (経過時間: {test_elapsed_time:.03f}[s/epoch]'
            f', 平均損失: {test_loss_mean:.05f}'
            f', 平均正解率: {test_accuracy_mean:.02f}[%])>')
    else:
        print(
            f'[{epoch+1}/{args.epochs}] テスト完了. '
            f'<テスト: (経過時間: {test_elapsed_time:.03f}[s/epoch]'
            f', 平均損失: {test_loss_mean:.05f}'
            f', 平均正解率: {test_accuracy_mean:.02f}[%])>')

logger.info('実行が終了しました。')
