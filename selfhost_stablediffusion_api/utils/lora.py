import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from typing import Union
from enum import Enum


class TaskType(Enum):
    TEXT_TO_IMAGE = "text_to_image"
    INPAINTING = "inpainting"


class LoraConfig:
    def __init__(self, task_type: TaskType, r: int, lora_alpha: float, lora_dropout: float, target_modules: list[str], bias: str):
        self.task_type = task_type
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.bias = bias


class LoraLayer(nn.Module):
    """
    Implements the LoRA (Low-Rank Adaptation) layer for fine-tuning specific model components.
    """

    def __init__(self, original_layer: nn.Module, r: int, lora_alpha: float, lora_dropout: float):
        super(LoraLayer, self).__init__()
        self.original_layer = original_layer
        self.lora_A = nn.Parameter(
            torch.randn((r, original_layer.in_features)))
        self.lora_B = nn.Parameter(torch.randn(
            (original_layer.out_features, r)))
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(lora_dropout)

    def forward(self, x):
        # Original forward pass
        original_output = self.original_layer(x)
        # LoRA adaptation: low-rank update
        lora_update = torch.matmul(self.lora_A, x)
        lora_update = self.dropout(torch.matmul(self.lora_B, lora_update))
        return original_output + self.scaling * lora_update


def apply_lora_to_model(model: nn.Module, lora_config: LoraConfig) -> nn.Module:
    """
    Applies LoRA to target layers in the model based on the configuration.
    """
    for name, module in model.named_modules():
        if any(target in name for target in lora_config.target_modules):
            lora_layer = LoraLayer(
                original_layer=module,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout
            )
            setattr(model, name, lora_layer)
    return model


def create_lora_config(task_type: Union[TaskType.TEXT_TO_IMAGE, TaskType.INPAINTING]) -> LoraConfig:
    """
    Creates a LoRA configuration based on the specified task type.
    """
    return LoraConfig(
        task_type=task_type,
        r=4,
        lora_alpha=16,
        lora_dropout=0.14,
        target_modules=["transformer", "attention"],
        bias="none"
    )


class ImageDataset(Dataset):
    """
    Custom dataset for loading and transforming images for the fine-tuning process.
    """

    def __init__(self, images: list[Image.Image]):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # Preprocess the image to 512x512 and normalize
        image = image.resize((512, 512))
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(
            2, 0, 1)  # C, H, W format
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
        return image_tensor


def compute_loss(output, target):
    """
    Compute the loss between the model's output and the target image.
    """
    loss_fn = nn.MSELoss()  # Mean Squared Error Loss
    return loss_fn(output, target)


def preprocess_image(image: Image.Image):
    """
    Preprocess the image for input into the model (resize to 512x512 and normalize).
    """
    image = image.resize((512, 512))
    tensor = torch.tensor(image, dtype=torch.float32).permute(
        2, 0, 1)  # Convert to C, H, W format
    tensor = tensor / 255.0  # Normalize pixel values to [0, 1]
    return tensor.unsqueeze(0)  # Add batch dimension


def fine_tune_with_lora(images: list[Image.Image], model: nn.Module, device: torch.device, tasktype: Union[TaskType.TEXT_TO_IMAGE, TaskType.INPAINTING]):
    """
    Fine-tunes the Stable Diffusion model with LoRA using the user's images.
    """
    # create lora_config
    lora_config = create_lora_config(tasktype)

    # Apply LoRA to specific layers in the model
    model = apply_lora_to_model(model, lora_config)
    model.to(device)

    # Step 2: Prepare the dataset and data loader
    dataset = ImageDataset(images)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Step 3: Define optimizer and training configuration
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()  # Use Mean Squared Error as the loss function

    model.train()  # Set the model to training mode

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            # Move batch of images to device (e.g., GPU)
            batch = batch.to(device)

            # Step 4: Forward pass
            outputs = model(batch)

            # Step 5: Compute loss between the model's output and the target (input) image
            loss = loss_fn(outputs, batch)

            # Step 6: Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            running_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

    return model  # Return the fine-tuned model with LoRA adaptations
