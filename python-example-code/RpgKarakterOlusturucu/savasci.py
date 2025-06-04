from karakter import Karakter
import random
class Savasci(Karakter):
    def __init__(self,isim):
        super().__init__(isim,"Savaşçı")

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
        else:
            hasar = random.randint(10,30)

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
            print(f"{self.isim} seviye atladı! yeni seviye: {self.seviye}")
