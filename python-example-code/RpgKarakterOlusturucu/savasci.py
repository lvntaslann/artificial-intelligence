from karakter import Karakter
import random
class Savasci(Karakter):
    def __init__(self,isim):
        super().__init__(isim,"Savaşçı")

    def savasciPasif(self,hasar):
        if hasar>20:
            self.can+=10
        if self.can<10:
            self.zırh=random.randint(10,20)
        if self.buyuDirenci>=5:
            self.zırh-=1
            hasar+=5
        return hasar
        

    def saldir(self,diger):
        if self.can<=0:
            self.can+=10
            print(f"{self.isim} baygın! saldıramaz.")
            return
        if diger.can <=0:
            diger.can+=10
            print(f"{diger.isim} zaten baygın!")

        if self.ultiSeviye>6:
            hasar = random.randint(30,50)
            if hasar>35:
                self.zırh+=10
        else:
            hasar = random.randint(10,30)
            if hasar>20:
                self.zırh+=5
                if self.savasciKontrolSkoru>0:
                    self.savasciKontrolSkoru-=1

        if self.zırh!=0 and self.zırh>8:
            hasar+=5
        
        
        if diger.buyucuKontrolSkoru>=3:
            hasar = 0
            diger.buyucuKontrolSkoru=0
            print(f"{diger.isim} saldırıdan etkilenmedi! Kontrol etkisi uygulandı.")
        else:
            hasar = self.savasciPasif(hasar)
            diger.can -=hasar
            self.xp +=10
            print(f"{self.isim}, {diger.isim} adlı karaktere {hasar} hasar verdi!")
        
        if self.xp>=self.yeniXpdegeri:
            self.seviye+=1
            self.xp =0
            self.yeniXpdegeri+=10

            if self.seviye>=2:
                self.ultiSeviye+=1
                diger.can+=10
                self.buyuDirenci+=1
                self.savasciKontrolSkoru+=1
            print(f"{self.isim} seviye atladı! yeni seviye: {self.seviye}")
        