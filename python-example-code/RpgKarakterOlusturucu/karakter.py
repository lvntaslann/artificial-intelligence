from abc import ABC, abstractmethod
class Karakter(ABC):
    def __init__(self, isim, tur):
        self.isim = isim
        self.tur = tur
        self.seviye = 1
        self.can = 100
        self.xp = 0
        self.yeniXpdegeri = 50
        self.ultiSeviye = 0
        self.max_can=self.can
        

    def __str__(self):
        return (f"{self.isim} ({self.tur}) | Seviye: {self.seviye} | "
                f"Can: {self.can} | XP: {self.xp} | Zırh: {self.zırh} | Ulti: {self.ultiSeviye}")

    @abstractmethod
    def seviye_arttir(self):
        pass

    @abstractmethod
    def saldir(self, diger):
        pass
    
    def canKontrol(self,diger):
        if self.can <= 0:
            print(f"{self.isim} baygın! Saldıramaz.")
            return
        if diger.can <= 0:
            print(f"{diger.isim} zaten baygın!")
            return
        
    def cani_degistir(self, miktar):
        self.can += miktar
        if self.can < 0:
            self.can = 0
        if self.can > self.max_can:
            self.max_can = self.can    