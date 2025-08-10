"""
images generation: MNIST veri seti

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.utils as utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# create data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
image_size = 28*28

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)) # -1 ile 1 arasına sıkıştırdık
])

dataset = datasets.MNIST(root="./data",train=True,transform=transform,download=True)

dataLoader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

# dicriminator
# fake or real
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size,1024), # 1024 output
            nn.LeakyReLU(0.2),
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1), # output layer gerçek ya da sahte mi ?
            nn.Sigmoid() # sınıflandırma
        )

    
    def forward(self,img):
        return self.model(img.view(-1,image_size)) # görüntüyü düzleştirme

# generator
class Generator(nn.Module): # goruntu(28x28) oluşturma
    def __init__(self,z_dim):
        super(Generator,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,image_size),
            nn.Tanh()
        )
    
    def forward(self,x):
        return self.model(x).view(-1,1,28,28) # çıktıyı 28x28 görüntüye çevirir

# gan training
# hiperparametreler
lr = 0.0002
z_dim = 50 # rastgele gürültü vektör boyutu (gürültü görüntüsü)
epochs = 10

# model başlatma: generator ve discriminator tanımla
generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)
# kayıp fonksiyonu ve optimizasyon algoritmalarının tanımlanması
criterion = nn.BCELoss() # binary cross entropy
g_optimizer = optim.Adam(generator.parameters(),lr = lr, betas = (0.5,0.999)) # generator
d_optimizer = optim.Adam(discriminator.parameters(),lr = lr, betas=(0.5,0.999)) # discriminator

# eğitim döngüsünün başlatılması
for epoch in range(epochs):
    for i, (real_imgs,_) in enumerate(dataLoader): # goruntulerin yüklenmesi
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0) # [128,28,28] mevcut batch boyutu
        real_labels = torch.ones(batch_size,1).to(device) # gerçek görüntüler 1
        fake_labels = torch.zeros(batch_size,1).to(device) # fake görüntüler 0

        #discriminator eğitilmesi
        z = torch.randn(batch_size,z_dim).to(device) # rastgele gürültü üret
        fake_img = generator(z) # sahte görüntü
        real_loss = criterion(discriminator(real_imgs),real_labels) # gerçek görüntü kaybı
        fake_loss = criterion(discriminator(fake_img.detach()),fake_labels) # sahte görüntülerin kaybı
        d_loss = real_loss + fake_loss # toplam discriminator kaybı
        
        d_optimizer.zero_grad() # gradyanları sıfırla
        d_loss.backward() # geriye yayılım
        d_optimizer.step() # parametreleri güncelle


        # generator eğitilmesi
        g_loss = criterion(discriminator(fake_img), real_labels) # generator kaybı
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
    print(f"Epoch [{epoch + 1}/{epochs}], d_loss={d_loss.item():.3f}, g_loss={g_loss.item():.3f}")




# model testing and performance evaluation

# rastgele gürültü ile görüntü oluşturma
with torch.no_grad():
    z = torch.randn(16,z_dim).to(device) # 16 adet rastgele gürültü
    sample_imgs = generator(z) # generator ile sahte görüntü oluşturma
    grid = utils.make_grid(sample_imgs, nrow=4, normalize=True).cpu()
    grid = np.transpose(grid.numpy(), (1, 2, 0))
    plt.imshow(grid)
    plt.show()