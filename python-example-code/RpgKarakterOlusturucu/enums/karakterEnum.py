from Karakter.erkek.muhtesemsuleyman import MuhtesemSuleyman
from Karakter.kadin.hurremsultan import HurremSultan
from Karakter.kadin.mihrimahsultan import MihrimaSultan
from Karakter.erkek.sehzadecihangir import SehzadeCihangir
from Karakter.erkek.sehzademehmet import SehzadeMehmet
from Karakter.erkek.sehzademustafa import SehzadeMustafa
from Karakter.erkek.tuborgselim import TuborgSelim

def karakter_sec(argument):
    switcher = {
        "Hürrem": HurremSultan("Hürrem"),
        "Süleyman": MuhtesemSuleyman("Süleyman"),
        "Mihrima": MihrimaSultan("Mihrima"),
        "Mustafa": SehzadeMustafa("Mustafa"),
        "Mehmet": SehzadeMehmet("Mehmet"),
        "Selim": TuborgSelim("Tuborg Selim"),
        "Cihangir": SehzadeCihangir("Cihangir"),
    }
    return switcher.get(argument, None)
