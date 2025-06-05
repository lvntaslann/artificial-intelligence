from karakter import Karakter
from savasci import Savasci
from buyucu import Buyucu
from muhtesemsuleyman import MuhtesemSuleyman
from hurremsultan import HurremSultan
import random

def main():
    k1 = MuhtesemSuleyman("Kanuni")
    k2 = HurremSultan("Hürrem")
    def oyun_dongusu(buyucu, savasci):
        tur = 1
        while buyucu.can > 0 and savasci.can > 0:
            print(f"\n--- TUR {tur} ---")
            if random.choice([True, False]):
                 buyucu.saldir(savasci)
                 if savasci.can <= 0:
                    break
                 savasci.saldir(buyucu)
            else:
                 savasci.saldir(buyucu)
                 if buyucu.can <= 0:
                     break
                 buyucu.saldir(savasci)
            print(f"{buyucu.isim} Can: {buyucu.can}, Seviye: {buyucu.seviye}")
            print(f"{savasci.isim} Can: {savasci.can}, Seviye: {savasci.seviye}")
            tur += 1
        print("\n--- OYUN BİTTİ ---")
        if buyucu.can > 0:
            print(f"Kazanan: {buyucu.isim}")
        else:
            print(f"Kazanan: {savasci.isim}")

    while True:
        print("\n--- Mini RPG ---")
        print("1. Karakterleri göster")
        print(f"2. Saldır ({k1.isim} → {k2.isim})")
        print(f"3. Saldır ({k2.isim} → {k1.isim})")
        print("4. Otomatik Oyun Başlat")
        print("5. Çıkış")

        secim = input("Seçim: ")

        if secim == "1":
            print(k1)
            print(k2)
        elif secim == "2":
            k1.saldir(k2)
        elif secim == "3":
            k2.saldir(k1)
        elif secim == "4":
            oyun_dongusu(buyucu=k2, savasci=k1)
        elif secim == "5":
            print("Oyun sonlandı")
            break
        else:
            print("Geçersiz giriş")


if __name__ == "__main__":
    main()