# 標準パッケージ
import argparse
import csv
from datetime import datetime
from logging import (
    getLogger, basicConfig,
    DEBUG, INFO, WARNING
)
from importlib import import_module
from pathlib import Path
import random
from time import perf_counter
from tqdm import tqdm

# 追加パッケージ
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# PyTorchとそのサブパッケージ
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# 自作パッケージ
from models.dcgan32 import Generator, Discriminator
from utils.device import AutoDevice
from utils.transforms.common import Normalize

# コマンドライン引数を取得するパーサー
parser = argparse.ArgumentParser(
    prog='PyTorchサンプルプログラム02',
    description='PyTorchを用いてMNIST/FashionMNISTの画像生成を行います。'
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
    '--criterion', help='訓練時に使用する損失関数を指定します。',
    type=str, default='binary_cross_entropy',
    choices=['binary_cross_entropy', 'hinge']
)
parser.add_argument(
    '--seed', help='乱数生成器のシード値を指定します。',
    type=int, default=42
)
# 画像生成に関する引数
parser.add_argument(
    '-z', '--z-dim', help='潜在空間のサイズを指定します。',
    type=int, default=128, metavar='Z'
)
parser.add_argument(
    '--sample-interval', help='生成画像の保存間隔をエポック数で指定します。',
    type=int, default=10,
)
parser.add_argument(
    '--plot', help='Matplotlibでサンプルを表示します。',
    action='store_true'
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
    '--load', help='訓練したモデルのパスをGenerator、Discriminatorの順に指定します。',
    type=str, nargs=2, default=None
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
args = parser.parse_args()

# 結果を出力するために起動日時を保持する
LAUNCH_DATETIME = datetime.now()

# ロギングの設定
basicConfig(
    format='%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=DEBUG if args.debug else INFO if args.info else WARNING,
)

# ロガーの名前を指定してロガーを取得する
logger = getLogger('main')

# 結果を出力するためのディレクトリの作成
OUTPUT_DIR = Path(
    LAUNCH_DATETIME.strftime(
        f'./outputs/{args.dataset}/%Y%m%d%H%M%S'))
OUTPUT_DIR.mkdir(parents=True)
logger.info(f'結果出力用のディレクトリ({OUTPUT_DIR})を作成しました。')
OUTPUT_SAMPLE_DIR = OUTPUT_DIR.joinpath('samples')
OUTPUT_SAMPLE_DIR.mkdir(parents=True)
logger.info(f'画像出力用のディレクトリ({OUTPUT_SAMPLE_DIR})を作成しました。')
if args.save:
    OUTPUT_MODEL_DIR = OUTPUT_DIR.joinpath('models')
    OUTPUT_MODEL_DIR.mkdir(parents=True)
    logger.info(f'モデル保存用のディレクトリ({OUTPUT_MODEL_DIR})を作成しました。')

# 再現性の設定
random.seed(args.seed)  # Python標準のランダムシード
np.random.seed(args.seed)  # NumPyのランダムシード
torch.manual_seed(args.seed)  # PyTorchのランダムシード
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
logger.info(f'ランダムシードを{args.seed}に設定しました。')

# デバイスについての補助クラスをインスタンス化
auto_device = AutoDevice(disable_cuda=args.disable_cuda)
device = auto_device()

# データセットのデータに行う処理のリスト
tfs_list = []
tfs_list.append(
    transforms.Pad(2, fill=0, padding_mode='constant')
)

# 正規化を行う際の平均と標準偏差
MEAN_GRAYSCALE = (0.5,)
STD_GRAYSCALE = (0.5,)

# 正規化クラスのインスタンス
# 今回は生成画像の正規化を元に戻す必要があるので
# transforms.Normalizeを継承して作った自前のNormalizeを用いる
# これには.inverse()で逆変換を行うメソッドが追加されている
normalize = Normalize(MEAN_GRAYSCALE, STD_GRAYSCALE)

# リスト同士の結合は.extend(リスト)
tfs_list.extend([
    transforms.ToTensor(),
    normalize
])
logger.info('画像のトランスフォームを定義しました。')


# データセットを読み込むための補助関数
def load_dataset(name: str, transform=None):
    if isinstance(transform, (list, tuple)):
        transform = transforms.Compose(transform)
    if name == 'mnist':
        num_classes = 10
        trainset = dset.MNIST(
            root=args.data_path, download=True, train=True,
            transform=transform)
    elif name == 'fashion_mnist':
        num_classes = 10
        trainset = dset.FashionMNIST(
            root=args.data_path, download=True, train=True,
            transform=transform)
    return trainset, num_classes


# データセットのロード
trainset, NUM_CLASSES = load_dataset(args.dataset, tfs_list)
NUM_FEATURES = 1
logger.info('データセットを読み込みました。')

# データローダーのインスタンス化
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True)

logger.info('データローダーを生成しました。')

# モデルの定義
model_g = Generator(args.z_dim, NUM_FEATURES)
model_d = Discriminator(NUM_FEATURES)
logger.info('モデルを定義しました。')
model_g = model_g.to(device)
model_d = model_d.to(device)
logger.info(f'モデルを{device}に転送しました。')
# モデルのパラメータの読み込み
if args.load is not None:
    model_g.load_state_dict(torch.load(args.load[0]))
    logger.info(f'学習済みGeneratorを{args.load[0]}から読み込みました。')
    model_d.load_state_dict(torch.load(args.load[1]))
    logger.info(f'学習済みDiscriminatorを{args.load[1]}から読み込みました。')

# オプティマイザ(最適化アルゴリズム)の定義
# Adamを使う場合はbetasを[0.5, 0.999]に設定する
# 小さい画像ではGeneratorが強くなりがち、大きい画像ではDiscriminatorが強くなりがち
# なので、画像のサイズに応じて学習率を非対称にする
optimizer_g = torch.optim.Adam(
    model_g.parameters(), lr=0.0016 * args.batch_size / 1000,
    betas=[0.5, 0.999])
optimizer_d = torch.optim.Adam(
    model_d.parameters(), lr=0.0004 * args.batch_size / 1000,
    betas=[0.5, 0.999])
logger.info('オプティマイザを定義しました。')

# 損失関数の定義 ([英]Criterion: [日]基準)
# ここではコマンドライン引数に応じた動的インポートを行っている
criterion_module = import_module(f'criterions.{args.criterion}')
logger.info(f'損失関数モジュール{args.criterion}を動的インポートしました。')
criterion = criterion_module.Criterion(args.batch_size, device)
logger.info('損失関数を定義しました。')

# 生成画像の確認のための乱数のサンプル
# 固定したノイズを使用する事で生成画像がどのように変化したかを確認できる
sample_z = torch.randn(100, args.z_dim, device=device)
logger.info('サンプルとなるノイズを生成しました。')

# 結果を記録するCSVファイルのハンドラ
f_results = open(
    OUTPUT_DIR.joinpath('results.csv'), mode='w', encoding='utf-8')
logger.info('結果出力用のファイルを開きました。')
csv_writer = csv.writer(f_results, lineterminator='\n')
result_items = [
    'Epoch', 'Generator Loss Mean', 'Discriminator Loss Mean',
    'Train Elapsed Time', 'Sample Image File',
    'Generator Saved File', 'Discriminator Saved File'
]
csv_writer.writerow(result_items)
f_results.flush()
csv_idx = {item: i for i, item in enumerate(result_items)}

fig, ax = plt.subplots(1, 1)

logger.info('訓練を開始します。')

# 訓練のループ
for epoch in range(args.epochs):
    # 記録の初期化
    results = ['' for _ in range(len(csv_idx))]
    results[csv_idx['Epoch']] = f'{epoch + 1}'

    # コマンドライン引数でテストモードを指定していたら学習は行わない
    if not args.test:
        # モデルを訓練モードに設定
        # DCGANでは2つのモデルがあるので2つとも設定する
        model_g.train()  # Generatorを訓練モードに設定
        model_d.train()  # Discriminatorを訓練モードに設定
        pbar = tqdm(
            trainloader,
            desc=f'[{epoch + 1}/{args.epochs}] 訓練開始',
            total=len(trainset)//args.batch_size,
            leave=False)
        losses_g, losses_d = [], []
        begin_time = perf_counter()  # 時間計測開始
        for real_images, _ in pbar:
            real_images = real_images.to(auto_device())

            ###################################################################
            # Discriminatorの訓練
            ###################################################################
            model_d.zero_grad()
            output_d_real = model_d(real_images)
            loss_d_real = criterion(output_d_real, real=True, generator=False)
            loss_d_real.backward()

            # 潜在空間からノイズzをサンプルする
            z = torch.randn(args.batch_size, args.z_dim, device=auto_device())
            # 生成画像をGeneratorから得る
            fake_images = model_g(z)
            # .detach()でmodel_gに勾配が伝播しないようにする
            output_d_fake = model_d(fake_images.detach())
            # 偽物画像を偽物とした時の誤差
            loss_d_fake = criterion(output_d_fake, real=False, generator=False)
            loss_d_fake.backward()
            loss_d = loss_d_real + loss_d_fake
            losses_d.append(loss_d.item())
            optimizer_d.step()

            ###################################################################
            # Generatorの訓練
            ###################################################################
            model_g.zero_grad()
            output_g = model_d(fake_images)
            # 偽物画像を本物とした時の誤差
            loss_g = criterion(output_g, real=True, generator=True)
            loss_g.backward()
            losses_g.append(loss_g.item())
            optimizer_g.step()
            pbar.set_description_str(
                f'[{epoch+1}/{args.epochs}] 訓練中... '
                f'<損失: (G={losses_g[-1]:.016f}, D={losses_d[-1]:.016f})>')
        end_time = perf_counter()  # 時間計測終了
        pbar.close()

        # それぞれの項目について記録する
        loss_g_mean = np.mean(losses_g)
        results[csv_idx['Generator Loss Mean']] = f'{loss_g_mean:.016f}'

        loss_d_mean = np.mean(losses_d)
        results[csv_idx['Discriminator Loss Mean']] = f'{loss_d_mean:.016f}'

        train_elapsed_time = end_time - begin_time
        results[csv_idx['Train Elapsed Time']] = f'{train_elapsed_time:.07f}'

    model_g.eval()  # Generatorを評価モードに設定
    model_d.eval()  # Discriminatorを評価モードに設定
    if (
            epoch == 0
            or (epoch + 1) % args.sample_interval == 0
            or epoch + 1 == args.epochs
    ):
        with torch.no_grad():
            sample_images = model_g(sample_z).detach().cpu()
            # Transformは画像1枚に対しての処理なのでforで回す
            for image in sample_images:
                normalize.inverse(image)
            _, _, image_h, image_w = sample_images.size()
            image_grid = vutils.make_grid(
                sample_images, nrow=10, padding=0)
            image_grid = image_grid.numpy().transpose(1, 2, 0)
            image_grid *= 255
            image_grid = image_grid.clip(0, 255).astype(np.uint8)

        if args.plot:
            title_text = f'Generated Images (At {epoch+1} Epochs)'
            fig.canvas.set_window_title(title_text)
            fig.suptitle(title_text)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(image_h))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(image_w))
            # TODO: リモート環境で画像表示は出来ないので代替策を実装する
            ax.imshow(image_grid, cmap='gray')
            plt.pause(0.01)
        sample_image_fname = f'{epoch+1:06d}.png'
        cv2.imwrite(
            str(OUTPUT_SAMPLE_DIR.joinpath(sample_image_fname)),
            image_grid)
        results[csv_idx['Sample Image File']] = sample_image_fname

    # モデルの保存
    if args.save and (
            (epoch + 1) % args.save_interval == 0
            or epoch + 1 == args.epochs):
        print(f'[{epoch+1}/{args.epochs}] モデルの保存中... ', end='')
        # Generatorの保存
        saved_fname_g = OUTPUT_MODEL_DIR.joinpath(
            f'generator_{epoch+1:06d}.pt')
        torch.save(model_g.state_dict(), saved_fname_g)
        results[csv_idx['Generator Saved File']] = saved_fname_g
        # Discriminatorの保存
        saved_fname_d = OUTPUT_MODEL_DIR.joinpath(
            f'discriminator_{epoch+1:06d}.pt')
        torch.save(model_d.state_dict(), saved_fname_d)
        results[csv_idx['Discriminator Saved File']] = saved_fname_d
        print('<完了>')
    # 結果をファイルに書き込みフラッシュする
    csv_writer.writerow(results)
    f_results.flush()

    if not args.test:
        print(
            f'[{epoch+1}/{args.epochs}] 訓練完了. '
            f'<訓練: (経過時間: {train_elapsed_time:.03f}[s/epoch]'
            f', 平均Generator損失: {loss_g_mean:.05f}'
            f', 平均Discriminator損失: {loss_d_mean:.05f}>')
    else:
        print(
            f'[{epoch+1}/{args.epochs}] 画像生成完了. ')

logger.info('実行が終了しました。')
