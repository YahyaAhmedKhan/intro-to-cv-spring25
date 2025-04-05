import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset, DataLoader
from config import DATA_ROOT


def dataset_info(dataset: Dataset) -> str:
    num_samples = ...  # YOUR CODE HERE
    num_classes = ...
    im_shape = ...
    im_dtype = ...
    return (
        f"Dataset with {num_samples} samples, "
        f"{num_classes} classes, image shape {im_shape} "
        f"and dtype {im_dtype}."
    )


def dataloader_info(dataloader: DataLoader):
    num_batches = ...  # YOUR CODE HERE
    num_samples = ...
    batch_size = ...
    batch_shape = ...
    return (
        f"DataLoader with {num_samples} total samples "
        f"split across {num_batches} batches of size {batch_size}. "
        f"Batch shape is {batch_shape}."
    )


def get_dataloader(dataset_name: str, batch_size: int, train: bool, **kwargs) -> DataLoader:
    if dataset_name == "mnist":
        transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        ds = torchvision.datasets.MNIST(
            root=DATA_ROOT / "mnist", train=train, download=True, transform=transform
        )
    elif dataset_name == "cifar10":
        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        ds = torchvision.datasets.CIFAR10(
            root=DATA_ROOT / "cifar10", train=train, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return DataLoader(ds, batch_size=batch_size, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str.lower, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    dl = get_dataloader(args.dataset, args.batch_size, train=True)
    ds = dl.dataset
    print(dataset_info(ds))
    print(dataloader_info(dl))
