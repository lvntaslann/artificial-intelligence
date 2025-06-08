from savasci import Savasci
import random

class MuhtesemSuleyman(Savasci):
    def __init__(self, isim):
        super().__init__(isim)
        

    def hurreme_ozel_saldiri(self, diger):
        self.hasar = random.randint(20, 40)
        if self.hasar>30:
            self.ask+=1
        else:
            self.ask-=1
        self.tarihgucu += 5
        if self.tarihgucu > 10:
            self.hasar += 10
            print("Sevgili sevgili bunları yapmayacaktın")
        if self.can<200:
            self.ask-=max(0, self.ask - 1)
        if self.ask==0:
            self.hasar+=20   
        diger.cani_degistir(-self.hasar)
        self.xp += 10
        print(f"{self.isim}, {diger.isim}'ına biricik aşkına... {self.hasar} hasar verdi!")


    def hadsizlik_kontrol(self, diger):
        if hasattr(diger, 'hadsizlik') and diger.hadsizlik > 5:
            self.adalet += 3
            if self.adalet > 10:
                if diger.isim=="Hürrem":
                    diger.fabrika-=1
                    self.capkinlik+=2
                diger.cani_degistir(-diger.can)
                print(f"{diger.isim}, Süleyman'ın adaletiyle yargılandı ve bayıldı!")
            

    def suleyman_hasar_hesapla_ve_saldir(self, diger):
        self.hasar += random.randint(20, 40)
        diger.cani_degistir(-self.hasar)
        self.xp += 10
        print(f"{self.isim}, {diger.isim} adlı karaktere {self.hasar} hasar verdi!")

    def saldir(self, diger):
        self.canKontrol(diger)
        self.hadsizlik_kontrol(diger)
        if self.seviye < 2:
            if diger.isim=="Hürrem":
                self.hurreme_ozel_saldiri(diger)
            else:
                super().saldir(diger)
        else:
            if diger.isim=="Hürrem":
                self.hurreme_ozel_saldiri(diger)
            else:
                self.suleyman_hasar_hesapla_ve_saldir(diger)

        if self.xp >= self.yeniXpdegeri:
            self.seviye_arttir()
