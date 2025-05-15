# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Define the problem
Classify handwritten digits (0–9) using the MNIST dataset.

### STEP 2: Import libraries and dataset
Import required libraries such as TensorFlow/Keras, NumPy, and Matplotlib.
Load the MNIST dataset using keras.datasets.mnist.load_data().


### STEP 3: Preprocess the data
Normalize the image pixel values (scale from 0-255 to 0-1).
Reshape the images to match CNN input shape.


### STEP 4: Build the CNN model
Initialize a Sequential model.
Add convolutional layers with activation (ReLU), followed by pooling layers.
Flatten the output and add Dense layers.
Use a softmax layer for classification.


### STEP 5: Compile and train the model
Compile the model with an optimizer (e.g., Adam), loss function (e.g., categorical crossentropy), and metrics (accuracy).
Train the model using training data and validate using validation split or test data.


### STEP 6: Evaluate and visualize results
Evaluate the model on test data and print accuracy.
Plot training/validation loss and accuracy curves.
Optionally, display a confusion matrix or sample predictions.




## PROGRAM
```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='../Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../Data', train=False, download=True, transform=transform)

train_data
test_data

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

# 1 colour channel, 6 filter(output channel) 3 x 3 kernal , stride = 1
conv1 = nn.Conv2d(1,6,3,1) # ---> 6 filters ---> pooling ---> conv2
# 6 input filters conv1, 16 filters, 3 x 3 kernal, stride = 1
conv2 = nn.Conv2d(6,16,3,1)

# Grab the first MNIST record
for i, (X_train, y_train) in enumerate(train_data):
    break

X_train

X_train.shape

x = X_train.view(1,1,28,28)  # 4D batch ( batch of 1 image)
x

x = F.relu(conv1(x))
x.shape

x = F.max_pool2d(x,2,2)
x.shape
x = F.relu(conv2(x))

x.shape

x = F.max_pool2d(x,2,2)
x.shape
11/2
(((28-2)/2) -2)/2

# flatten 
x.shape
x.view(-1,16*5*5).shape

class ConvolutionalNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(5*5*16,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5*5*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


torch.manual_seed(42)
model = ConvolutionalNetwork()
model

for param in model.parameters():
    print(param.numel())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import time
start_time = time.time()

# Variables ( Trackers)
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

# for loop epochs 
for i in range(epochs):
    
    trn_corr = 0
    tst_corr = 0


    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):
        b+=1
        
        # Apply the model
        y_pred = model(X_train)  # we not flatten X-train here
        loss = criterion(y_pred, y_train)
 
        
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()  # Trure 1 / False 0 sum()
        trn_corr += batch_corr
        
        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print interim results
        if b%600 == 0:
            print(f'epoch: {i}  batch: {b} loss: {loss.item()}')
        
    train_losses.append(loss)
    train_correct.append(trn_corr)
        
    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):

            # Apply the model
            y_val = model(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1] 
            tst_corr += (predicted == y_test).sum()
            
    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)
        
current_time = time.time()
total = current_time - start_time
print(f'Training took {total/60} minutes')

# Detach and convert to NumPy
train_losses = [t.detach().numpy() for t in train_losses]
test_losses = [t.detach().numpy() for t in test_losses]

plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label='validation loss')
plt.title('Loss at the End of Each Epoch\nBy Richard')
plt.legend();
plt.show()

plt.plot([t/600 for t in train_correct], label='training accuracy')
plt.plot([t/100 for t in test_correct], label='validation accuracy')
plt.title('Accuracy at the end of each epoch\nBy Richard')
plt.legend();
plt.show()

# Extract the data all at once, not in batches
test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = model(X_test)  # we don't flatten the data this time
        predicted = torch.max(y_val,1)[1]
        correct += (predicted == y_test).sum()

correct.item()
correct.item()/len(test_data)

# print a row of values for reference
np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}'))
print(np.arange(10).reshape(1,10))
print()

# print the confusion matrix
print(confusion_matrix(predicted.view(-1), y_test.view(-1)))
print("RICHARDSON A")

# single image for test 
plt.imshow(test_data[2019][0].reshape(28,28))
plt.show()

model.eval()
with torch.no_grad():
    new_prediction = model(test_data[2019][0].view(1,1,28,28))

new_prediction.argmax()

# Convert tensors to NumPy arrays
y_true = y_test.view(-1).cpu().numpy()
y_pred = predicted.view(-1).cpu().numpy()

# Print classification report
print("Classification Report by Richard\n")
print(classification_report(y_true, y_pred))


# single image for test 
plt.imshow(test_data[333][0].reshape(28,28))
plt.show()
model.eval()
with torch.no_grad():
    new_prediction = model(test_data[333][0].view(1,1,28,28))

new_prediction.argmax()

test_data[333][1]
```
### Name:Vijayaraj V

### Register Number: 212222230174

### OUTPUT

## Training Loss per Epoch
![image](https://github.com/user-attachments/assets/c3af5f9f-083f-4916-8fab-2eaf4a48a9b2)

![image](https://github.com/user-attachments/assets/8668431d-6a3c-4e77-b314-90d3d77fda59)


## Confusion Matrix
![image](https://github.com/user-attachments/assets/e61ca322-b439-4d8c-bc20-08e96a0eae1d)



## Classification Report
![image](https://github.com/user-attachments/assets/50dc7591-7349-4b89-a28b-ca3217f02a8c)



### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/c3a64f74-a76c-4ab7-8f8a-6a24277fea51)


## RESULT
The CNN model achieved high accuracy on the MNIST dataset, with training and validation losses showing good convergence. The classification report and confusion matrix confirmed strong performance across all digits, with minimal misclassifications. Overall, the model performs reliably in handwritten digit recognition.
