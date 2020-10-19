from logging import getLogger
from typing import Callable, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

logger = getLogger('models.classification32')


class Block(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        num_classes: int = 0, prob_dropout: float = 0.2,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu
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
        logger.debug('Dropout層を生成しました。')
        self.activation = activation

    def forward(self, inputs):
        # このメソッドはとんでもない回数実行されるため
        # forward内でログの出力は行わない
        return self.activation(self.dropout(self.conv(inputs)))


class Classification32(nn.Module):
    '''32×32画素の画像を分類するモデル
    '''
    def __init__(
        self, in_channels: int,
        num_classes: int = 0
    ):
        super().__init__()
        self.main = nn.Sequential(
            # (in_channels, 32, 32) -> (16, 16, 16)
            Block(in_channels, 16, 3, stride=2, padding=1),
            # (16, 16, 16) -> (24, 8, 8)
            Block(16, 24, 3, stride=2, padding=1),
            # (24, 8, 8) -> (32, 4, 4)
            Block(24, 32, 3, stride=2, padding=1),
            # (32, 4, 4) -> (48, 2, 2)
            Block(32, 48, 3, stride=2, padding=1),
        )
        logger.info('モデルの畳み込みブロック列を定義しました。')
        self.dense = nn.Linear(48, num_classes)
        logger.info('全結合層を定義しました。')

    def forward(
        self, inputs, labels=None, form: str = 'rgb_pixels'
    ):
        h = self.main(inputs)
        h = F.adaptive_avg_pool2d(h, (1, 1))
        return self.dense(h.view(-1, 48))
