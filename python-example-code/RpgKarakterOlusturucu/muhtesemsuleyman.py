from savasci import Savasci
import random

class MuhtesemSuleyman(Savasci):
    def __init__(self, isim):
        super().__init__(isim)
        

    def hurreme_ozel_saldiri(self, diger):
        if diger.isim == "Hürrem":
            self.tarihgucu += 5
            if self.tarihgucu > 20:
                self.hasar += 10
                print("Sevgili sevgili bunları yapmayacaktın")

    def hadsizlik_kontrol(self, diger):
        if hasattr(diger, 'hadsizlik') and diger.hadsizlik > 5:
            self.adalet += 3
            if self.adalet > 20:
                diger.can = 0
                print(f"{diger.isim}, Süleyman'ın adaletiyle yargılandı ve bayıldı!")

    def suleyman_hasar_hesapla_ve_saldir(self, diger):
        self.hasar = random.randint(20, 40)
        self.hurreme_ozel_saldiri(diger)
        diger.can -= self.hasar
        self.xp += 10
        print(f"{self.isim}, {diger.isim} adlı karaktere {self.hasar} hasar verdi!")

    def saldir(self, diger):
        if self.can <= 0:
            print(f"{self.isim} baygın! Saldıramaz.")
            return
        if diger.can <= 0:
            print(f"{diger.isim} zaten baygın!")
            return

        self.hadsizlik_kontrol(diger)

        if self.seviye > 2:
            super().saldir(diger)  # Savaşçı’nın saldırısını uygula
        else:
            self.suleyman_hasar_hesapla_ve_saldir(diger)

        # Seviye atlama kontrolü
        if self.xp >= self.yeniXpdegeri:
            self.seviye_arttir()
