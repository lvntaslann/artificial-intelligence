"""
Problem tanımı : mnist veri seti ile rakam sınıflandırması

MNIST
ANN: Yapay sinir ağları

"""

# library
import torch #pytorch kütüphanesi, tensor işlemleri
import torch.nn as nn # yapay sinir ağı katmanları tanımlamak için
import torch.optim as optim # ağırlıkları güncellemek için algortimaları içeren modül
import torch.utils
import torch.utils.data
import torchvision # goruntu işleme ve pretrained modelleri içerir
import torchvision.transforms as transform # goruntu donusumu
import matplotlib.pyplot as plt

# selected device
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loading
def get_data_loaders(batch_size=64):
    transformation = transform.Compose([
        transform.ToTensor(), # görüntüyü 0-255->0-1 tensöre çevirir
        transform.Normalize((0.5,),(0.5,)) # -1 ile 1 arasına normalize edilir
    ])
    train_set = torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=transformation)
    test_set = torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=transformation)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)
    return train_loader,test_loader



# data visulization
def visualize_samples(loader,n):
    images,labels = next(iter(loader)) # ilk batchden görüntü ve etiket alma işlemi
    # print(images[i].shape)
    fig,axes = plt.subplots(1,n,figsize=(10,5)) # n farkli goruntu için gorselleştrime alanı
    for i in range(n):
        axes[i].imshow(images[i].squeeze(),cmap="gray") # gri tonlamalı
        axes[i].set_title(f"Label:{labels[i].item()}") # görüntüye ait sınıfı
        axes[i].axis("off") # eksenleri gizle
    plt.show()


# define ann model
class NeuralNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten() # elimizde bulunan goruntuleri vektor haline cevirme(1d)
        self.fc1 = nn.Linear(28*28,128) # ilk tam bağlı katmanı oluştur input, output
        self.relu = nn.ReLU() # aktivasyon fonksiyonu
        self.fc2 = nn.Linear(128,64) # ikinci tam bağlı katman input, output 
        self.fc3 = nn.Linear(64,10) # cıktı katmanı input, output = sınıf sayısı

    def forward(self,x): # ileri yayılım 
        
        x = self.flatten(x) # initial x = 28/28 -> flatten ile düzleştir
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc3(x) # çıktı katmanı
        return x
    


# loss and optimizer
define_loss_and_optimizer = lambda model:(
    nn.CrossEntropyLoss(), # çok sınıflı sınıflandırma kayıp fonksiyonu
    optim.Adam(model.parameters(),lr=0.001) # ağırlıkları güncellemek için optimizer
)

# train
def train_model(model,train_loader,criterion,optimizer,epochs=10):
    # modeli eğitim moduna alma
    model.train()
    train_losses=[] # her bir epoch sonucunda elde edilen değerleri sakla
    # belirtilen epoch sayisi kadar eğitim yap
    for epoch in range(epochs):
        total_loss = 0
        for images,labels in train_loader: # tum eğitim verileri üzerinde itreasyon gerçekleştir
            images,labels = images.to(device), labels.to(device)
            optimizer.zero_grad() # gradyanları sıfırla
            predictions = model(images) # modeli uygula, forward pro
            loss = criterion(predictions,labels) # loss hesaplama -> y_prediction y_real
            loss.backward() # geri yayılım gradyan hesaplama
            optimizer.step() # ağırlıkları güncelleme
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader) # ortalama kayıp hesaplama
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.3f} ")

    # loss grafiği
    plt.figure()
    plt.plot(range(1,epochs+1),train_losses,marker="o",linestyle="-",label="Train loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train loss")
    plt.legend()
    plt.show()



    
# test
def test_model(model,tets_loader):
    model.eval() # değerlendirme modu
    correct = 0
    total = 0
    with torch.no_grad(): # gradyan hesaplama gereksiz olduğundan kapatıldı /train de yapılıyor
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            prediction = model(images)
            _, predicted = torch.max(prediction,1) # en yüksek olasılıklı sınıfın etiketi
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    print(f"Test acc : {100*correct/total:.3f}%")



if __name__ == "__main__":
    train_loader,test_loader = get_data_loaders()
    visualize_samples(train_loader,5)
    model = NeuralNetwork().to(device)
    criterion,optimizer = define_loss_and_optimizer(model)
    train_model(model,train_loader,criterion,optimizer)
    test_model(model,test_loader)
    