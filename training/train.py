import argparse
import yaml
import torch
import timm
from data_loader import get_data_loader

def train(config):
    # Get the data loader
    data_loader = get_data_loader(config["data_dir"], config["batch_size"])

    # Create the model
    model = timm.create_model(config["model_name"], pretrained=True, num_classes=config["num_classes"])
    model.train()

    # Create the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config["num_epochs"]):
        for images, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {loss.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the training config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)
