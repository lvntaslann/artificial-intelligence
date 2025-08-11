"""
problem tanımı : veri sıkıştırması -> autoencoders
fashion mnist veriseti

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


# veri seti yükleme ve ön işleme
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(root = "./data", train = True, transform = transform, download = True)
test_dataset = datasets.FashionMNIST(root = "./data", train = False, transform = transform, download = True)

batch_size = 128

train_loader = DataLoader(train_dataset,batch_size = batch_size, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size = batch_size, shuffle=False)

# auto encoders geliştirme
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        #encoder
        self.encoder = nn.Sequential(
            nn.Flatten(), # ->28x28 (2D)
            nn.Linear(28*28,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
        )
        #decoder
        self.decoder = nn.Sequential(
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,28*28),
            nn.Sigmoid(),
            nn.Unflatten(1,(1,28,28)),
        )

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# call back: early stopping
class EarlyStopping:
    def __init__(self,patience=5,min_delta=0.001):
        self.patience = patience # kaç epoch boyunca gelişme olmazsa durdur
        self.min_delta = min_delta # kayıptaki minumum gelişme(threshold)
        self.best_loss = None # en iyi kayıp değeri sakla
        self.counter = 0 # sabit kalan epcoh sayacı

    def __call__(self,loss):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            return True # trainingi durdur
        return False

# model eğitimi
epochs = 10
lr = 0.001
model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = lr)
early_stopping = EarlyStopping(patience = 5, min_delta = 0.001)

def training(model,train_loader,optimizer,criterion,early_stopping,epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.5f}")

        if early_stopping(avg_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

training(model,train_loader,optimizer,criterion,early_stopping,epochs)

# model testi

def compute_ssim(img1,img2,sigma=1.5):
    """
     iki görüntü arasındaki benzerliği hesaplar

    """
    c1 = (0.01*255)**2 # ssim sabitlerinden bir tanesi
    c2 = (0.03*255)**2 

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # goruntulerin ortalamaları
    mu1 = gaussian_filter(img1,sigma)
    mu2 = gaussian_filter(img2,sigma)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1*mu2

    #varyans
    sigma1_sq = gaussian_filter(img1**2,sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2**2,sigma) - mu2_sq
    sigma12 = gaussian_filter(img1*img2,sigma)  - mu1_mu2

    # ssim_map
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def evaluate(model,test_loader,n_images = 10):
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs, _ = batch
            outputs = model(inputs)
            break
    inputs = inputs.numpy()
    outputs = outputs.numpy()
    
    fig, axes = plt.subplots(2,n_images, figsize=(n_images,3))
    ssim_scores = []
    
    for i in range(n_images):
        img1 = np.squeeze(inputs[i]) #orjinal görüntüyü sıkıştır
        img2 = np.squeeze(outputs[i]) # yeniden oluşturulmuş görüntüyü sıkıştır

        ssim_score = compute_ssim(img1,img2) # benzerlik hesapla
        ssim_scores.append(ssim_score)

        axes[0,i].imshow(img1,cmap="gray")
        axes[0,i].axis("off")
        axes[1,i].imshow(img2,cmap="gray")
        axes[1,i].axis("off")
    
    axes[0,0].set_title("original")
    axes[1,0].set_title("Decoded image")
    plt.show()

    avg_ssim = np.mean(ssim_scores)
    print(f"avegare SSIM : {avg_ssim}")


evaluate(model,test_loader,10)
