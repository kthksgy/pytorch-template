from logging import getLogger
from typing import Tuple, Union

import torch
import torch.nn as nn

logger = getLogger('models.dcgan32')


def init_xavier_uniform(layer: nn.Module):
    '''層の初期化をXavier Uniformで行う
    Args:
        layer: 層
    '''
    if hasattr(layer, "weight"):
        torch.nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if hasattr(layer.bias, "data"):
            layer.bias.data.fill_(0)


class GBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride, padding, bias=bias)
        logger.debug('転置畳み込み層を生成しました。')
        self.conv.apply(init_xavier_uniform)
        logger.debug('転置畳み込み層を初期化しました。')
        self.bn = nn.BatchNorm2d(out_channels)
        logger.debug('BatchNormを生成しました。')
        self.activation = nn.LeakyReLU(0.3)
        logger.debug('LeakyReLUを生成しました。')

    def forward(self, inputs):
        return self.activation(self.conv(inputs))


class Generator(nn.Module):
    '''32×32画素の画像生成GANのGenerator
    '''
    def __init__(
        self, in_channels: int, out_channels: int,
        bias: bool = False
    ):
        super().__init__()

        # nn.Sequentialを使って書いても良い
        # (in_channels, 1, 1) -> (256, 8, 8)
        self.block1 = GBlock(
                in_channels, 256, 8, 1, padding=0,
                bias=bias)
        logger.info('転置畳み込みブロック1を生成しました。')
        # (256, 8, 8) -> (128, 16, 16)
        self.block2 = GBlock(
                256, 128, 4, 2, padding=1,
                bias=bias)
        logger.info('転置畳み込みブロック2を生成しました。')
        # (128, 16, 16) -> (64, 32, 32)
        self.block3 = GBlock(
                128, 64, 4, 2, padding=1,
                bias=bias)
        logger.info('転置畳み込みブロック3を生成しました。')
        self.out = nn.Sequential(
            # (64, 32, 32) -> (out_channels, 32, 32)
            nn.ConvTranspose2d(64, out_channels, 1, padding=0, bias=bias),
            nn.Tanh()
        )
        logger.info('出力ブロックを生成しました。')

    def forward(
        self, inputs: torch.Tensor
    ):
        x = inputs.view(inputs.size(0), inputs.size(1), 1, 1)
        # nn.Sequentialを使わず書いても適用方法は同じ
        # 但し、それぞれのブロックを全て順番に適用しないといけない
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.out(x)


class DBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        prob_dropout: float = 0.5
    ):
        super().__init__()
        self.conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size, stride, padding)
        logger.debug('畳み込み層を生成しました。')
        nn.init.xavier_uniform_(self.conv.weight)
        logger.debug('畳み込み層の重みを初期化しました。')
        nn.init.zeros_(self.conv.bias)
        logger.debug('畳み込み層のバイアスを初期化しました。')

        self.dropout = nn.Dropout2d(prob_dropout)
        logger.debug('Dropoutを生成しました。')
        self.activation = nn.LeakyReLU(0.3)
        logger.debug('LeakyReLUを生成しました。')

    def forward(self, inputs):
        return self.activation(self.dropout(self.conv(inputs)))


class Discriminator(nn.Module):
    '''32×32画素の画像生成GANのDiscriminator
    '''
    def __init__(
        self, in_channels: int
    ):
        super().__init__()
        self.main = nn.Sequential(
            # (in_channels, 32, 32) -> (64, 16, 16)
            DBlock(in_channels, 64, 3, stride=2, padding=1),
            # (64, 16, 16) -> (128, 8, 8)
            DBlock(64, 128, 3, stride=2, padding=1),
            # (128, 8, 8) -> (256, 4, 4)
            DBlock(128, 256, 3, stride=2, padding=1),
        )
        logger.info('畳み込みブロック列を生成しました。')
        self.last = nn.Linear(256 * 4 * 4, 1)
        logger.info('出力層を生成しました。')

    def forward(
        self, inputs
    ):
        h = self.main(inputs)
        # Real(1) or Fake(0)を出力する
        return torch.sigmoid(self.last(h.view(-1, 256 * 4 * 4)))
