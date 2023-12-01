# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as image_transforms
import matplotlib.pyplot as plt
import numpy as np

# Configure the device (GPU or CPU)
processing_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters for training
total_epochs = 5
batch_sz = 4
learn_rate = 0.001

# Transforming images from PILImage to Tensors and normalizing them
image_transform = image_transforms.Compose(
    [image_transforms.ToTensor(),
     image_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load and transform the CIFAR10 dataset
train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=image_transform)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=image_transform)

# DataLoader objects for training and testing
data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=batch_sz,
                                          shuffle=True)

data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=batch_sz,
                                         shuffle=False)

# Class labels in CIFAR10 dataset
class_names = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Function to display an image
def display_image(img_tensor):
    img_tensor = img_tensor / 2 + 0.5  # Undo normalization
    np_img = img_tensor.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()

# Displaying random training images
data_iter = iter(data_loader_train)
sample_images, sample_labels = next(data_iter)
display_image(torchvision.utils.make_grid(sample_images))

# Neural network architecture
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer_conv1 = nn.Conv2d(3, 6, 5)
        self.layer_pool = nn.MaxPool2d(2, 2)
        self.layer_conv2 = nn.Conv2d(6, 16, 5)
        self.layer_fc1 = nn.Linear(16 * 5 * 5, 120)
        self.layer_fc2 = nn.Linear(120, 84)
        self.layer_fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.layer_pool(F.relu(self.layer_conv1(x)))
        x = self.layer_pool(F.relu(self.layer_conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.layer_fc1(x))
        x = F.relu(self.layer_fc2(x))
        x = self.layer_fc3(x)
        return x

# Initialize model, loss function, and optimizer
neural_network = NeuralNet().to(processing_device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(neural_network.parameters(), lr=learn_rate)

# Training loop
steps_per_epoch = len(data_loader_train)
for epoch in range(total_epochs):
    for i, (imgs, lbls) in enumerate(data_loader_train):
        imgs = imgs.to(processing_device)
        lbls = lbls.to(processing_device)

        # Forward pass
        predicted_outputs = neural_network(imgs)
        loss = loss_function(predicted_outputs, lbls)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{total_epochs}], Step [{i+1}/{steps_per_epoch}], Loss: {loss.item():.4f}')

print('Training Complete')

# Save the trained model
model_path = './neural_net.pth'
torch.save(neural_network.state_dict(), model_path)

# Testing the model
with torch.no_grad():
    correct_preds = 0
    total_preds = 0
    correct_preds_per_class = [0 for _ in range(10)]
    total_preds_per_class = [0 for _ in range(10)]
    for imgs, lbls in data_loader_test:
        imgs = imgs.to(processing_device)
        lbls = lbls.to(processing_device)
        output = neural_network(imgs)
        _, predicted = torch.max(output, 1)
        total_preds += lbls.size(0)
        correct_preds += (predicted == lbls).sum().item()

        for j in range(batch_sz):
            actual_label = lbls[j]
            predicted_label = predicted[j]
            if (actual_label == predicted_label):
                correct_preds_per_class[actual_label] += 1
            total_preds_per_class[actual_label] += 1

    overall_acc = 100.0 * correct_preds / total_preds
    print(f'Overall Accuracy: {overall_acc} %')

    for i in range(10):
        class_acc = 100.0 * correct_preds_per_class[i] / total_preds_per_class[i]
        print(f'Accuracy of {class_names[i]}: {class_acc} %')
