from karakter import Karakter
import random

class Buyucu(Karakter):
    def __init__(self,isim):
        super().__init__(isim,"Büyücü")
        
    def buyucuPasif(self, hasar):
        if self.can < 80 or self.can > 40:
            hasar += 10
            self.xp += 3
            return hasar
        if self.buyuNufuzu>=5:
            hasar+=5
        return hasar
        
            
    def  saldir(self,diger):
        if self.can <= 0:
            self.can += 10
            print(f"{self.isim} baygın! saldıramaz")
            return
        if diger.can <= 0:
            diger.can += 10
            print(f"{diger.isim} zaten baygın")
            return
        
        if self.ultiSeviye > 6:
            hasar = random.randint(25,35)
            if hasar>30:
                diger.zırh -=5
                if diger.zırh<0:
                    diger.zırh=0
        else:
            hasar = random.randint(15,35)
            if hasar>25:
                diger.zırh=0
                if self.buyucuKontrolSkoru>0:
                    self.buyucuKontrolSkoru-=1


        if diger.savasciKontrolSkoru>=3:
            hasar = 0
            diger.savasciKontrolSkoru=0
            print(f"{self.isim}, {diger.isim} adlı karaktere {hasar} hasar verdi!")
        else:
            hasar = self.buyucuPasif(hasar)
            diger.can -= hasar
            self.xp += 15
            print(f"{self.isim}, {diger.isim} adlı karaktere {hasar} büyü hasarı verdi!")
        
        
        if self.xp >= self.yeniXpdegeri:
            self.seviye += 1
            self.zırh+=1
            self.xp = 0
            self.yeniXpdegeri += 10
            if self.seviye >=2:
                self.ultiSeviye += 1
                self.buyuNufuzu+=1
                self.buyucuKontrolSkoru+=1
                diger.can += 10
            print(f"{self.isim} seviye atladı! yeni seviye: {self.seviye}")

        


