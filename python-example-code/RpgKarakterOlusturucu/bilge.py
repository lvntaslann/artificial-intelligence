from karakter import Karakter
class Bilge(Karakter):
    def __init__(self, isim,):
        super().__init__(isim,"Bilge")
        self.bilgelikgucu=2
        self.dusunce=""
        self.zeka=2
        self.hasar=2
    

    def seviye_arttir(self):
        self.seviye += 1
        self.xp = 0
        self.yeniXpdegeri += 10
        print(f"{self.isim} seviye atladı! yeni seviye: {self.seviye}")

    def dusunceKontrol(self,diger):
        if self.dusunce=="Üzgün":
           self.cani_degistir(-5)
           self.bilgelikgucu-=1
           self.hasar-=1
           diger.cani_degistir(-1)
        elif self.dusunce=="Öfke":
           self.cani_degistir(-3)
           self.bilgelikgucu+=2
           self.hasar+=20
           diger.cani_degistir(-10)
        elif self.dusunce=="Coşku":
           self.cani_degistir(30)
           self.hasar=0
           self.bilgelikgucu+=1

    def saldir(self, diger):
        if self.can<=0:
            self.cani_degistir(-self.can)
            print(f"${self.isim} baygın! saldırılamaz")
        
        if self.zeka>5:
            self.bilgelikgucu+=2
            self.can+=10
            self.xp+=10
        
        if self.bilgelikgucu>5:
            self.hasar+=50

        self.dusunceKontrol(diger)

        if diger.hasar>25 and diger.hasar<45:
            self.dusunce="Öfke"
        elif diger.hasar>15 and diger.hasar<25:
            self.dusunce="Üzgün"
        elif diger.hasar<15:
            self.dusunce="Coşku"



        
