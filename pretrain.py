from network import Net
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from column_env import RandomObstaclesEnv
env = RandomObstaclesEnv()
N_S = env.observation_space.shape
N_A = env.action_space.n
model = Net(N_S,N_A)
train_data = torch.tensor(np.load('state.npy'))
train_label = torch.tensor(np.load('action.npy')).type(torch.LongTensor)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Create a DataLoader for your dataset
dataset = TensorDataset(train_data, train_label)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training the model
num_epochs = 2000
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        # Forward pass
        logits, _ = model(inputs)
        loss = criterion(logits, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    with torch.no_grad():
        outputs,_ = model(inputs)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).sum().item() / len(labels)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')
    print(f'Accuracy: {accuracy:.2f}')
print('Finished Training')