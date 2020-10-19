import torch
import torchvision.transforms as transforms


class Normalize(transforms.Normalize):
    '''transforms.NormalizeのWrapper
    逆変換を追加実装したバージョンのNormalize
    '''
    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            tensor: (C, H, W)形式の画像テンソル
        Returns:
            正規化前の画像テンソル
        '''
        for t, m, s in zip(tensor, self.mean, self.std):
            # 正規化はt.sub_(m).div_(s)
            t.mul_(s).add_(m)
        return tensor
