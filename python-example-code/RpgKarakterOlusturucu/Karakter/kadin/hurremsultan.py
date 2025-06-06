from buyucu import Buyucu
import random

class HurremSultan(Buyucu):
    def __init__(self, isim):
        super().__init__(isim)

    def suleymana_ozel_saldiri(self, diger):
        if diger.isim == "Süleyman":
            self.ask += 2
            if self.ask > 10 and self.ask<20:
                self.sarayKontrolu += 2
            elif self.ask > 20:
                self.fabrika += 1
                if self.fabrika > 10:
                    self.sarayKontrolu += 4

    def hurrem_hasar_hesapla_ve_saldir(self, diger):
        self.hasar = random.randint(15, 35)
        self.suleymana_ozel_saldiri(diger)

        if self.sarayKontrolu > 10:
            self.hadsizlik += 1
            self.hasar += 5

        diger.can -= self.hasar
        self.xp += 10
        print(f"{self.isim}, {diger.isim} adlı karaktere {self.hasar} hasar verdi!")
        if self.hasar > 20:
            self.hadsizlik += 2
            self.ask -= 2
            print("Edirne'deki saray yükleniyor...")

    def saldir(self, diger):
        if self.can <= 0:
            print(f"{self.isim} baygın! Saldıramaz.")
            return
        if diger.can <= 0:
            print(f"{diger.isim} zaten baygın!")
            return

        if self.seviye < 2:
            self.hurrem_hasar_hesapla_ve_saldir(diger)
        else:
            super().saldir(diger)
            self.hurrem_hasar_hesapla_ve_saldir(diger)


        if self.xp >= self.yeniXpdegeri:
            self.seviye_arttir()
