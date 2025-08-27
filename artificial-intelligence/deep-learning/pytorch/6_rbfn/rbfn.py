import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# veri seti içeriği aktarılması
df = pd.read_csv("iris/iris.data",header = None)
print(df.head())

X = df.iloc[:,:-1].values
y, _ = pd.factorize(df.iloc[:,-1].values)


# veriyi standardize et
scaler = StandardScaler()
X = scaler.fit_transform(X)

# train test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 42)

def to_tensor(data,target):
    return torch.tensor(data,dtype = torch.float32), torch.tensor(target,dtype=torch.long)


X_train, y_train = to_tensor(X_train,y_train)
X_test, y_test = to_tensor(X_test,y_test)


def rbf_kernel(X,centers,beta):
    return torch.exp(-beta*torch.cdist(X,centers)**2)

# RBFN ve rbf_kernelin tanımlanması
class RBFN(nn.Module):
    def __init__(self,num_centers,input_dim,output_dim):
        super(RBFN,self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers,input_dim))
        self.beta = nn.Parameter(torch.ones(1)*2.0)
        self.linear = nn.Linear(num_centers,output_dim)

    def forward(self,x):
        #rbf çekirdek fonk hesapla
        phi = rbf_kernel(x,self.centers,self.beta)
        return self.linear(phi)


# model eğitimi
num_centers = 10
model = RBFN(input_dim=4, num_centers = num_centers,output_dim=3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.01)


num_epochs=100

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs,y_train)
    loss.backward()
    optimizer.step()

    if(epoch+1) %10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss : {loss.item():.4f}")


# model test
with torch.no_grad():
    y_pred = model(X_test)
    acc = (torch.argmax(y_pred,axis=1)==y_test).float().mean().item()
    print(f"acc:{acc}")