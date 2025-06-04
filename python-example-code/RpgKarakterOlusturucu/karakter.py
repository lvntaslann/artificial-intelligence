from abc import ABC, abstractmethod
import random

class Karakter(ABC):
    def __init__(self, isim, tur):
        self.isim = isim
        self.tur = tur
        self.seviye = 1
        self.can = 100
        self.xp = 0
        self.yeniXpdegeri = 50
        self.ultiSeviye = 0
        self.zırh=2
        self.buyuDirenci =2
        self.buyuNufuzu=2
        self.buyucuKontrolSkoru = 1
        self.savasciKontrolSkoru = 1

    def __str__(self):
        return (f"{self.isim} ({self.tur}) | Seviye: {self.seviye} | "
                f"Can: {self.can} | XP: {self.xp} | Zırh: {self.zırh} | Ulti: {self.ultiSeviye}")


    @abstractmethod
    def saldir(self, diger):
        pass