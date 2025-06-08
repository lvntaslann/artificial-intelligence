from buyucu import Buyucu
import random

class HurremSultan(Buyucu):
    def __init__(self, isim):
        super().__init__(isim)

    def suleymana_ozel_saldiri(self, diger):
        self.hasar += random.randint(15, 35)
        self.ask += 2
        if self.ask > 10 and self.ask<20:
            self.sarayKontrolu += 2
        elif self.ask > 20:
             self.fabrika += 1
             if self.fabrika > 10:
                self.sarayKontrolu += 4
        diger.cani_degistir(-self.hasar)
        print(f"{self.isim}, {diger.isim}'ına biricik aşkına... {self.hasar} hasar verdi!")

    def hurrem_hasar_hesapla_ve_saldir(self, diger):
        self.hasar = random.randint(15, 35)
        if self.sarayKontrolu > 10:
            self.hadsizlik += 1
            self.hasar += 5
        diger.cani_degistir(-self.hasar)
        self.xp += 10
        print(f"{self.isim}, {diger.isim} adlı karaktere {self.hasar} hasar verdi!")
        if self.hasar > 20:
            self.hadsizlik += 2
            self.ask -= 2
            if diger.isim=="Süleyman":
                print("Edirne'deki saray yükleniyor...")

    def saldir(self, diger):
        self.canKontrol(diger)

        if self.seviye < 2:
            if diger.isim=="Süleyman":
                self.suleymana_ozel_saldiri(diger)
            else: 
                super().saldir(diger)
                
        else:
            if diger.isim=="Süleyman":
                self.suleymana_ozel_saldiri(diger)
            else: 
                self.hurrem_hasar_hesapla_ve_saldir(diger)


        if self.xp >= self.yeniXpdegeri:
            self.seviye_arttir()
