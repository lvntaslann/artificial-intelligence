"""
transfer learning : çiçeklerin sınıflandırılması (102 farklı sınıf)
mobilenet ile transfer learning

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
# veri yükleme ve veri arttırma
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# veri dönüşümü
# 1 klasik donusumler: normalizasyon, tensor dönüşümü
# 2 mobilenete uygun giriş boyutunun ayarlanması
# 3 data augmentation

transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(((0.5,0.5,0.5)), (0.5,0.5,0.5))
])


#oxford flowers 102 veri seti yukleme
train_dataset = datasets.Flowers102(root="./data",split="train",transform = transform_train,download =True)
test_dataset = datasets.Flowers102(root="./data",split="val",transform = transform_test,download =True)


# rastgele 5 ornek seçimi
indices = torch.randint(len(train_dataset),(5,))
samples = [train_dataset[i] for i in indices]

# gorsellestirme
fig, axes = plt.subplots(1,5,figsize = (15,5))
for i, (image,label) in enumerate(samples):
    image = image.numpy().transpose((1,2,0))
    image = (image*0.5) + 0.5 # çiçekleri normal hale çevirme
    axes[i].imshow(image)
    axes[i].set_title(f"label:{label}")
    axes[i].axis("off")
plt.show()


#data loader
train_loader = DataLoader(train_dataset,batch_size = 32, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size = 32,shuffle=False)
# transfer learning tanımlama ve fine tuning
# mobilenet v2 yükleme
model = models.mobilenet_v2(pretrained=True)
# sınıflandırıcı katmanı ekleme
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features,102)
model = model.to(device)
# kayıp fonksiyonu ve optimizer tanımlama
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[1].parameters(),lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)

# model eğitimi
epochs = 10
for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images,labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    print(f"Epoch:{epoch+1}, Loss:{running_loss/len(train_loader):.4f}")
# model kaydetme
torch.save(model.state_dict(),"mobilenet_flowers102.pth")

# model test etme
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images,labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs,1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# confusion matrix
cm = confusion_matrix(all_labels,all_preds)
plt.figure(figsize=(12,12))
sns.heatmap(cm,annot=False,cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(all_labels,all_preds))