import torch
import torch.nn as nn


class Criterion():
    '''GAN用のBinaryCrossEntropy損失関数
    '''
    def __init__(self, batch_size: int, device: torch.device):
        '''
        Args:
            batch_size: モデルの出力のバッチサイズ
            device: モデルと同じデバイス
        '''
        self.__ones = torch.ones(batch_size, 1).to(device)
        self.__zeros = torch.zeros(batch_size, 1).to(device)
        self.__loss_function = nn.BCELoss(reduction='mean')

    def __call__(
        self, outputs: torch.Tensor, real: bool, generator: bool
    ) -> torch.Tensor:
        '''
        Args:
            outputs: Discriminatorの出力
            real: Discriminatorの入力が本物画像だったかどうか
            generator: Discriminatorの入力がGenerator生成画像だったかどうか
        Returns:
            損失値のテンソル
        '''
        if real:
            return self.__loss_function(outputs, self.__ones)
        else:
            return self.__loss_function(outputs, self.__zeros)
