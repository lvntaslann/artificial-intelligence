from karakter import Karakter
import random
class Savasci(Karakter):
    def __init__(self,isim):
        super().__init__(isim,"Savaşçı")
        self.hasar = 10
        self.tarihgucu = 10
        self.adalet = 2
        self.capkinlik = 1
        self.ask=2
        self.can=300
        self.zırh=2
        self.buyuDirenci =2
        self.buyucuKontrolSkoru = 1
        self.sarayGucu=2
        self.hadsizlik=1

    def seviye_arttir(self):
        self.seviye += 1
        self.xp = 0
        self.yeniXpdegeri += 10
        print(f"{self.isim} seviye atladı! yeni seviye: {self.seviye}")    

    def savasciPasif(self, hasar):
        if hasar > 20:
            self.cani_degistir(10)
        if self.can < 10:
            self.zırh = random.randint(10, 20)
        if self.buyuDirenci >= 5:
            self.zırh -= 1
            hasar += 5
        return hasar

        

    def saldir(self, diger):
        if self.can <= 0:
            self.cani_degistir(-self.can)
            print(f"{self.isim} baygın! saldıramaz.")
            return
        if diger.can <= 0:
            diger.cani_degistir(-diger.can)
            print(f"{diger.isim} zaten baygın!")

        self.hasar = random.randint(30, 50) if self.ultiSeviye > 6 else random.randint(10, 30)
        if self.hasar > 35 and self.ultiSeviye > 6:
            self.zırh += 10
        elif self.hasar > 20:
            self.zırh += 5
            if self.buyucuKontrolSkoru > 0:
                self.buyucuKontrolSkoru -= 1

        if self.zırh > 8:
            self.hasar += 5

        if diger.tur == "Buyucu":
            if diger.savasciKontrolSkoru >= 3:
                diger.savasciKontrolSkoru = 0
                print(f"{diger.isim} saldırıdan etkilenmedi! Kontrol etkisi uygulandı.")
            else:
                self.hasar = self.savasciPasif(self.hasar)
                diger.cani_degistir(-self.hasar)
                print(f"{self.isim}, {diger.isim} adlı karaktere {self.hasar} hasar verdi!")
        else:
            self.hasar = self.savasciPasif(self.hasar)
            diger.cani_degistir(-self.hasar)
            self.xp += 10
            print(f"{self.isim}, {diger.isim} adlı karaktere {self.hasar} hasar verdi!")

        if self.xp >= self.yeniXpdegeri:
            self.seviye_arttir()
            if self.seviye >= 2:
                self.ultiSeviye += 1
                diger.cani_degistir(10)
                self.buyuDirenci += 1
                self.buyucuKontrolSkoru += 1
