"""
Problem: CIFAR10 veriseti sınıflandırma problemi
CNN
"""
#import library
import torch
import torch.nn as nn # sinir ağı katmanları
import torch.optim as optim # optimizasyon alg
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# load dataset
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # rgb kanallar
    ])
    train_set = torchvision.datasets.CIFAR10(root="./data",download=True,train=True,transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="./data",download=True,train=False,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size,shuffle=False)
    return train_loader,test_loader

# visualize dataset
def imShow(img):
    # verileri normalize etmeden önce geri dönüştürme
    img = img/2+0.5
    np_img = img.np()
    plt.imshow(np.transpose(np_img,(1,2,0))) # 3kanal için doğru sıra
    plt.show()

def get_sample_images(train_loader):
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    return images,labels

def visualize(train_loader,n):
    images, labels = get_sample_images(train_loader)
    plt.figure()
    for i in range(n):
        plt.subplot(1,n,i+1)
        imShow(images[i])
        plt.title(f"Label:{labels[i].item()}")
    plt.show()


# build CNN models
class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel,self).__init__()

        self.conv1 = nn.Conv2d(3,32,kernel_size=3,padding=1)# inputcha = 3 ,outputcha=32 kernel size =3x3
        self.relu = nn.ReLU() #aktivasyon fonk
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.dropout = nn.Dropout(0.2) # hücrelerin %20 si sıfırlanır
        self.fc1 = nn.Linear(64*8*8,128) # giriş = 4096 output = 128
        self.fc2 = nn.Linear(128,10) # output layer

    def forward(self,x):
        """
            image 3x32x32 -> conv(32) -> relu(32) -> pool(16) ->
            conv(16) -> relu(16) -> pool(8) -> image = 8x8
            fc1-> relu -> dropout
            fc2 -> output

        """
        x =  self.pool(self.relu(self.conv1(x))) # conv1
        x =  self.pool(self.relu(self.conv2(x))) # conv2
        x =  x.view(-1,64*8*8) # flatten
        x =  self.dropout(self.relu(self.fc1(x)))
        x =  self.fc2(x)
        return x



#training
def train_model(model,train_loader,criterion,optimizer,epochs=10,):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss/ len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss: .5f}")
    
    plt.figure()
    plt.plot(range(1,epochs+1),train_losses,marker="o",linestyle = "-",label="Train loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.legend()
    plt.show()



#test
def test_model(model,test_loader,dataset_type):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"{dataset_type} acc : {100*correct/total}% ")



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNmodel().to(device)   
    define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(),
    torch.optim.SGD(model.parameters(), lr = 0.001,momentum=0.9)
    )
    train_loader, test_loader = get_data_loaders()
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model=model,train_loader=train_loader,criterion=criterion,optimizer=optimizer,epochs=10)
    test_model(model,test_loader,dataset_type="Test")
    test_model(model,train_loader,dataset_type="Train")