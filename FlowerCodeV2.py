import torch
import torch.nn as nn
from torchvision.transforms import Normalize
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os

#Step 0: Preprocess Data

train_1_path = r"D:\FlowerData\flowers\train\daisy"
train_0_path = r"D:\FlowerData\flowers\train\dandelion"
test_1_path = r"D:\FlowerData\flowers\test\daisy"
test_0_path = r"D:\FlowerData\flowers\test\dandelion"

def process_images(image_path, target, image_size=(100,100)):
    
    images = os.listdir(image_path)
    num_images = len(images)
    
    image_matrix = np.empty((num_images, 3, *image_size))
    
    if target == 0:
        target_array = np.zeros((num_images,1),dtype=int)
    elif target == 1:
        target_array = np.ones((num_images,1),dtype=int)
        
    count = 0
    for image_file in images:
        #Create full path to item
        full_image_path = os.path.join(image_path, image_file)
        #Manipulate images
        image = Image.open(full_image_path)
        image = image.resize(image_size)
        #Convert to numpy array (**With ToTensor can convert PIL image to pytorch tensor)
        pixel_mat = np.array(image).reshape((3, *image_size))
        image_matrix[count,:] = pixel_mat
        count += 1
    
    return image_matrix, target_array

train_daisy, targets_1 = process_images(train_1_path, 1)
train_dandelion, targets_0 = process_images(train_0_path,0)
test_daisy, test_1 = process_images(test_1_path, 1)
test_dandelion, test_0 = process_images(test_0_path, 0)

test_targets = np.concatenate((test_1, test_0))
X_test = np.concatenate((test_daisy, test_dandelion), axis=0)

#Concatenate the daisy and dandelion photos together

targets = np.concatenate((targets_1,targets_0))
X = np.concatenate((train_daisy, train_dandelion), axis=0)

#Shuffle
random_indices = np.random.permutation(X.shape[0])
X = X[random_indices]
targets = targets[random_indices]

#Convert data to pytorch tensors
X = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy(targets.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
test_targets = torch.from_numpy(test_targets.astype(np.float32))

#Transform data
normalize = Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
X = normalize(X)
X_test = normalize(X_test)

#Define hyper parameters
num_epochs = 100
learning_rate = 0.002
kernel_size = 5
padding = 2
stride = 2
pool_size = 2

#Create our model (Remember: out_size = ((in_size - kernal_size + 2*padding) / stride) + 1)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pool = nn.MaxPool2d(2,stride=stride) #Shrinks by a factor of 2
        self.conv2 = nn.Conv2d(6,12, kernel_size=kernel_size, stride=stride, padding=padding)
        self.fc1 = nn.Linear(6*6*12,30) 
        self.fc2 = nn.Linear(30,60)
        self.fc3 = nn.Linear(60,1)
    def forward(self, x, n_samples):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        #Need to flatten tensor before putting through fully connected linear layers
        out = out.view(-1,6*6*12)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = out.reshape(n_samples,1)
        return out

model = ConvNet()

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Training Loop

total_steps = X.shape[0]
for epoch in range(num_epochs):

    #Shuffle before each epoch
    indices = torch.randperm(X.shape[0])
    X = X[indices]
    targets = targets[indices]

    #Forward pass
    predictions = model.forward(X, X.shape[0])
    targets = targets.float().view_as(predictions) #Ensure target and prediction have the same shape
    l = criterion(torch.sigmoid(predictions), targets)
        
    #Backwards pass
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
        
    print(f'epoch = {epoch+1} / {num_epochs}, loss = {l.item():.4f}')
        
print("DONE TRAINING")

#Model Evaluation
with torch.no_grad():
    n_correct = 0 
    n_samples = X_test.shape[0]
    predictions = torch.sigmoid(model.forward(X_test, n_samples))
    predictions = predictions.round()
    n_correct = test_targets.eq(predictions).sum().item()
    acc = 100.0 * (n_correct / n_samples)

print(acc)