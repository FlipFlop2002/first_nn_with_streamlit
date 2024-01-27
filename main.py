import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from model import AnimalClassificationModel
from torch import nn
from timeit import default_timer as timer
from train_test_functions import train
import os


device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("Animals/")
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(
    root=data_path,
    transform=data_transform,
    target_transform=None
)

test_data = datasets.ImageFolder(
    root=data_path,
    transform=data_transform
)

animal_classes = train_data.classes

# dataloaders
BATCH_SIZE = 8
train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# img, label = next(iter(train_dataloader))
# print(img.shape)
# print("-----------------")
# print(label)

plt.figure(figsize=(10, 7))
plt.imshow(train_data[0][0].permute(1, 2, 0))
plt.show()

# train the model
torch.manual_seed(42)
torch.cuda.manual_seed(42)
model_0 = AnimalClassificationModel(
    input_shape=3,
    hidden_units=10,
    output_shape=len(animal_classes)
    ).to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

EPOCHS = 35
start_time = timer()


model_0_results = train(
    model=model_0,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device,
    num_epochs=EPOCHS
      )

end_time = timer()
print(f"Total training tim {end_time-start_time:.3f} seconds.")

if os.path.exists('models/model_0.pth'):
    print("Model dict already saved!")
else:
    print("Saving model dict")
    torch.save(model_0.state_dict(), "models/model_0.pth")
