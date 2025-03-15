import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LABEL_NAMES = ["background", "kart", "pickup", "nitro", "bomb", "projectile"]


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path: str):
        """
        Dataset initialization, handling image and depth pairs.
        """
        to_tensor = transforms.ToTensor()

        self.data = []
        image_files = sorted(Path(dataset_path).glob("*_im.jpg"))
        for image_file in image_files:
            depth_file = image_file.with_name(image_file.stem.replace('_im', '_depth') + '.png')
            if depth_file.exists():
                image = Image.open(image_file)
                depth = Image.open(depth_file)

                # Convert to tensors
                image_tensor = to_tensor(image)
                depth_tensor = to_tensor(depth)

                # In this case, there may not be an explicit label (unless you have some criterion for generating labels)
                # so we can use dummy labels for now or infer them from the filename.
                label = 0  # Assign a dummy label, or change this according to your needs.

                self.data.append((image_tensor, depth_tensor, label))
            else:
                print(f"Warning: Depth file {depth_file} missing for image {image_file}")

        print(f"Loaded {len(self.data)} samples")  # Add this line

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(dataset_path: str, num_workers: int = 0, batch_size: int = 128, shuffle: bool = False) -> DataLoader:
    dataset = SuperTuxDataset(dataset_path)

    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor):
    """
    Arguments:
        outputs: torch.Tensor, shape (b, num_classes) either logits or probabilities
        labels: torch.Tensor, shape (b,) with the ground truth class labels

    Returns:
        a single torch.Tensor scalar
    """
    outputs_idx = outputs.max(1)[1].type_as(labels)

    return (outputs_idx == labels).float().mean()