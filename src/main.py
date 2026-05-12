import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.models.landmark_model import build_model
from src.data.transforms import get_transforms
from src.data.dataset import LandmarkDataset, load_metadata
from src.training.train import run_model


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def main():
    ### get device
    device = get_device()
    print("Using device:", device)

    ### load in data / metadata
    metadata, hf_dataset = load_metadata()

    ### split images into train and test sets
    train_data, val_data = train_test_split(metadata, test_size=0.2, random_state=42)

    print("Train size:", len(train_data))
    print("Val size:", len(val_data))

    train_transform, val_transform = get_transforms()

    ### make dataset objects
    train_dataset = LandmarkDataset(train_data, hf_dataset, transform=train_transform)
    val_dataset = LandmarkDataset(
        val_data,
        hf_dataset,
        label_map=train_dataset.label_map,
        transform=val_transform
    )

    ### get dataloaders for datasets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    ### make the model
    num_classes = len(train_dataset.label_map)
    model = build_model(num_classes, unfreeze_layer4=True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    ### train model
    run_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)


if __name__ == "__main__":
    main()