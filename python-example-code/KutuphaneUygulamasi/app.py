from model import Kutuphane

k = Kutuphane()
while True:
    print("1. Kitap ekle")
    print("2. Kitap sil")
    print("3. Kitapları listele")
    print("4. Çıkış")
    secim = input("İşleminizi seçin (1/2/3/4): ")

    if secim == "4":
        print("Çıkılıyor...")
        break
    else:
        k.kullanici_istegi(secim)
