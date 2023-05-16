from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# Define the path to your dataset folder
data_path = '<path to your dataset folder>'

# Define the transformation to be applied to each image
transform = transforms.Compose(
    [transforms.Resize((224, 224)),  # Resize the image to 224x224
     transforms.ToTensor(),  # Convert the image to a PyTorch tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Normalize the image

# Define the dataset
dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Define the dataloaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Define the model
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.classes))

# Define the optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=100*len(train_loader))
criterion = nn.CrossEntropyLoss()

# Define the device to be used for training (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)
num_epochs = 50
best_val_acc = 0.0

for epoch in range(num_epochs):
    # Train the model
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    model.train()
    for i, data in enumerate(tqdm(train_loader), 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # Update the learning rate
        scheduler.step()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 batches
            train_acc = 100 * train_correct / train_total
            train_loss = running_loss / 100
            print('[Epoch %d, Batch %d] Train Loss: %.3f, Train Acc: %.2f%%' %
                  (epoch + 1, i + 1, train_loss, train_acc))
            running_loss = 0.0
            train_correct = 0
            train_total = 0

    # Evaluate the model on the validation set
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    with torch.no_grad():
        for data in tqdm(val_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            val_loss += loss.item()

    val_acc = 100 * val_correct / val_total
    val_loss /= len(val_loader)
    print('[Epoch %d] Val Loss: %.3f, Val Acc: %.2f%%' %
          (epoch + 1, val_loss, val_acc))

    # Save the model if it has the best validation accuracy so far
    if val_acc > best_val_acc:
        print('Saving model with Val Acc: %.2f%%' % val_acc)
        torch.save(model.state_dict(), 'resnet18'+ str(val_acc) +'.pth')
        best_val_acc = val_acc

print('Finished Training')
