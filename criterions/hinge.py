import torch


class Criterion():
    '''GAN用のHinge損失関数
    '''
    def __init__(self, batch_size: int, device: torch.device):
        '''
        Args:
            batch_size: モデルの出力のバッチサイズ
            device: モデルと同じデバイス
        '''
        self.__ones = torch.ones(batch_size, 1, device=device)
        self.__zeros = torch.zeros(batch_size, 1, device=device)

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
        if generator:
            return -torch.mean(outputs)
        else:
            if real:
                return -torch.mean(torch.min(outputs - 1, self.__zeros))
            else:
                return - torch.mean(torch.min(-outputs - 1, self.__zeros))
