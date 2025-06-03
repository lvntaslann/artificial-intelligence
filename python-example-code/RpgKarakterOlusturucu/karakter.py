import random

class Karakter:
    def __init__(self,isim,tur):
        self.isim = isim
        self.tur = tur
        self.seviye =1
        self.can = 100
        self.xp = 0
        self.yeniXpdegeri=50
        self.ultiSeviye=0

    def __str__(self):
        return f"{self.isim} ({self.tur}) | Seviye: {self.seviye} | Can: {self.can} | XP: {self.xp}"
    

    def saldir(self,diger):
        if self.can<=0:
            self.can +=10
            print(f"{self.isim} baygın! Saldıramaz.")
            return
        if diger.can<=0:
            diger.can+=10
            print(f"{diger.isim} zaten baygın")
            return
        
        if self.ultiSeviye>6:
            hasar = random.randint(30,50)
        else:
            hasar = random.randint(10,30)
        diger.can-=hasar
        self.xp+=10
        print(f"{self.isim}, {diger.isim} adli karaktere {hasar} verildi!")
            
        if self.xp>=self.yeniXpdegeri:
            self.seviye+=1
            self.xp=0
            self.yeniXpdegeri+=10
            if self.seviye>=2:
                self.ultiSeviye+=1
                diger.can +=10
            
            
            print(f"{self.isim} seviye atladı! Yeni seviye:{self.seviye}")

    

