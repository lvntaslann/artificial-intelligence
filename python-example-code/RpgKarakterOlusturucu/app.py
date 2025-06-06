# app.py

import random
from enums.karakterEnum import karakter_sec

def oyun_dongusu(karakter1, karakter2):
    tur = 1
    while karakter1.can > 0 and karakter2.can > 0:
        print(f"\n--- TUR {tur} ---")
        if random.choice([True, False]):
            karakter1.saldir(karakter2)
            if karakter2.can <= 0:
                karakter2.can = 0
                break
            karakter2.saldir(karakter1)
        else:
            karakter2.saldir(karakter1)
            if karakter1.can <= 0:
                karakter1.can = 0
                break
            karakter1.saldir(karakter2)
        print(f"{karakter1.isim} Can: {karakter1.can}, Seviye: {karakter1.seviye}")
        print(f"{karakter2.isim} Can: {karakter2.can}, Seviye: {karakter2.seviye}")
        tur += 1

    print("\n--- OYUN BİTTİ ---")
    if karakter1.can > 0:
        print(f"Kazanan: {karakter1.isim}")
    else:
        print(f"Kazanan: {karakter2.isim}")

def main():
    karakter = None
    karakter2 = None

    while True:
        print("\n--- Mini RPG ---")
        print("1. Karakter seç")
        print("2. Karakterleri göster")
        print("3. Otomatik Oyun Başlat")
        print("4. Çıkış")

        secim = input("Seçim: ")

        if secim == "1":
            print("Kullanılabilir karakterler: Hürrem, Süleyman, Cihangir, Tuborg Selim, Mehmet, Mustafa, Mihrima, Bilge")

            isim1 = input("1. Karakter ismi: ")
            karakter = karakter_sec(isim1)
            if not karakter:
                print("Geçersiz karakter!")
                continue

            isim2 = input("2. Karakter ismi: ")
            karakter2 = karakter_sec(isim2)
            if not karakter2:
                print("Geçersiz karakter!")
                karakter = None
                continue

        elif secim == "2":
            if karakter and karakter2:
                print(f"Karakter 1: {karakter.isim}, Tür: {type(karakter).__name__}")
                print(f"Karakter 2: {karakter2.isim}, Tür: {type(karakter2).__name__}")
            else:
                print("Lütfen önce karakter seçin.")

        elif secim == "3":
            if karakter and karakter2:
                oyun_dongusu(karakter, karakter2)
            else:
                print("Önce karakter seçmelisin.")

        elif secim == "4":
            print("Oyun sonlandırıldı.")
            break
        else:
            print("Geçersiz giriş.")

if __name__ == "__main__":
    main()
