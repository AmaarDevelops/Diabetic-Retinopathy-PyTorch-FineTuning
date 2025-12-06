import torch.nn as nn
import torchvision.models as models
import torch


# ----------------------- Phase 1 : Freezing the entire block ----------------------

def get_model_and_setup(num_classes=5, lr=0.001, step_size=2, gamma=0.5):
    """Initializes EfficientNet-B0, replaces classifier, freezes layers, and defines optimizers/schedulers."""

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the Classifier Head
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Optimizer targets ONLY the new classifier layer for Phase 1
    optimizer = torch.optim.SGD(model.classifier[1].parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return model, criterion, optimizer, scheduler, device



