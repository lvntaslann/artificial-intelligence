from savasci import Savasci
import random
class SehzadeMustafa(Savasci):
    def __init__(self, isim):
        super().__init__(isim)
    

    def mustafa_hasar_hesapla_ve_saldir(self, diger):
        self.hasar+=random.randint(20, 40)
        self.xp += 10
        if self.seviye<2:
            diger.can -= self.hasar
            self.xp += 10
        print(f"{self.isim}, {diger.isim} adlı karaktere {self.hasar} hasar verdi!")

    def saldir(self, diger):
        self.canKontrol(diger)
        if self.tarihgucu>5 and self.can<200:
            if self.capkinlik<5:
                self.adalet+=1
                self.hasar+=1
        
        if self.adalet>5:
            self.sarayGucu+=1
            if self.sarayGucu>5:
                self.hasar+=2
                self.hadsizlik+=1
            else:
                self.hasar-=1
                self.hadsizlik = max(0, self.hadsizlik - 1)
        else:
            self.can+=1
        
        if diger.tur=="Buyucu":
            self.hasar+=10
            self.hadsizlik+=2
        elif diger.tur=="Bilge":
            self.hasar+=2
            self.hadsizlik = max(0, self.hadsizlik - 1)
        elif diger.tur=="Savaşcı":
            self.hasar+=5

        if diger.isim=="Süleyman":
            #hasar verilemez
            if self.hadsizlik<3:
                self.hadsizlik = max(0, self.hadsizlik - 1)
                self.mustafa_hasar_hesapla_ve_saldir(diger)
            else:
                print(f"{self.isim}, Süleyman'a karşı hürmetsizlik yaptı! Hadsizlik puanı: {self.hadsizlik}")
        else:
            if self.seviye<2:
                super().saldir(diger)
            else:
                self.mustafa_hasar_hesapla_ve_saldir(diger)
                

        if self.xp >= self.yeniXpdegeri:
            self.seviye_arttir()
