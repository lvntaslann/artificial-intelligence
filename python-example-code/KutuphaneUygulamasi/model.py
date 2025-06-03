class Kutuphane:
    def __init__(self):
        self.kitaplar = []

    def kitap_ekle(self,kitap_adi):
        self.kitaplar.append(kitap_adi)
        print(f"{kitap_adi} eklendi.")

    def kitaplari_listele(self):
        if not self.kitaplar:
            print("Kütüphane boş")
        else:
            print("kütüphanedeki kitaplar: ")
            for index, kitap in enumerate(self.kitaplar,1):
                print(f"{index}. {kitap}")

    def kitap_sil(self,index):
        if 1<=index<=len(self.kitaplar):
            silinen = self.kitaplar.pop(index-1)
            print(f"{silinen} silindi.")
        else:
            print("Geçersiz index")


    def kullanici_istegi(self,userInput):
        if userInput=="1":
            kitap_adi = input("Eklemek istediğiniz kitabın adı: ")
            self.kitap_ekle(kitap_adi)
    
        elif userInput =="2":
            silinecek = int(input("Silmek istediğiniz kitabın numarası: "))
            self.kitap_sil(silinecek)
        elif userInput =="3":
            self.kitaplari_listele()
    

