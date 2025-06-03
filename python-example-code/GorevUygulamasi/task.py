gorevler = []

def gorev_ekle():
    yeni_gorev = input("Eklemek istediğiniz görevi yazınız: ")
    if yeni_gorev == "":
        print("Boş görev eklenemez.")
    else:
        gorevler.append(yeni_gorev)
        print("Görev eklendi")


def gorevleri_listele():
    if not gorevler:
        print("Henüz görevler eklenmedi")
    else:
        print("\n---Görevler---")
        for index, gorev in enumerate(gorevler, 1):
            print(f"{index}. {gorev}")


def gorev_sil():
    index = input("Görev numarası girin [1,2,3...]:")
    del gorevler[int(index)-1]
    print("Gorevler ${index} silindi.")
            

while True:
    print("\n ---Görev takip uygulaması---")
    print("1. Görev ekle")
    print("2. Görevleri listele")
    print("3. Çıkış")
    print("4.Görev sil")

    secim = input("Seçiminiz (1/2/3/4): ")

    if secim == "1":
        gorev_ekle()

    elif secim == "2":
        gorevleri_listele()

    elif secim == "3":
        print("Uygulamadan çıkılıyor...")
        break

    elif secim =="4":
        gorev_sil()

    elif secim == "":
        print("Boş geçilemez.")

    else:
        print("Geçersiz seçim! Lütfen 1, 2 veya 3 girin.")
