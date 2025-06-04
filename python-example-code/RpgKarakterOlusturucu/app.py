from karakter import Karakter
from savasci import Savasci
from buyucu import Buyucu

def main():
    k1 = Savasci("Arda")
    k2 = Buyucu("Mira")

    while True:
        print("\n--- Mini RPG ---")
        print("1. Karakterleri göster")
        print("2. Saldır (Arda → Mira)")
        print("3. Saldır (Mira → Arda)")
        print("4. Çıkış")

        secim =input("Seçim: ")

        if secim =="1":
            print(k1)
            print(k2)
    
        elif secim=="2":
            k1.saldir(k2)

        elif secim =="3":
            k2.saldir(k1)
    
        elif secim =="4":
            print("Oyun sonlandı")
            break

        else:
            print("Geçersiz giriş")

if __name__ == "__main__":
    main()