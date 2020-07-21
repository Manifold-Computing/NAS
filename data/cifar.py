import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class CifarDataset:

  def __init__(
      self,
      configs,
      name: str = 'cifar10',
      batch: int = 32,
  ):
    self.configs = configs
    if name == 'cifar10':
      self._ds = datasets.CIFAR10
    else:
      self._ds = datasets.CIFAR100

    normalize = transforms.Normalize(
        mean=(0.49139968, 0.48215827, 0.44653124),
        std=(0.24703233, 0.24348505, 0.26158768),
    )

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    self.train = torch.utils.data.DataLoader(
        self._ds(
            root=self.configs.data_dir,
            train=True,
            transform=transform,
            download=True,
        ),
        batch_size=self.configs.batch_size,
        shuffle=True,
        num_workers=self.configs.num_workers,
        pin_memory=True,
    )

    self.valid = torch.utils.data.DataLoader(
        self._ds(
            root=self.configs.data_dir,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
        ),
        batch_size=self.configs.batch_size,
        shuffle=False,
        num_workers=self.configs.num_workers,
        pin_memory=True,
    )

    self.test = self.valid


