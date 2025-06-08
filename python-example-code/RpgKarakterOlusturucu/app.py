#app.py

import pygame
import sys
import random
from enums.karakterEnum import karakter_sec
from arayuz import Arayuz

def game_loop(first_character, second_character):
    arayuz = Arayuz()
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if first_character.can > 0 and second_character.can > 0:
            if random.choice([True, False]):
                first_character.saldir(second_character)
                if second_character.can <= 0:
                    second_character.can = 0
                pygame.time.wait(500)
                second_character.saldir(first_character)
                if first_character.can <= 0:
                    first_character.can = 0
            else:
                second_character.saldir(first_character)
                if first_character.can <= 0:
                    first_character.can = 0
                pygame.time.wait(500)
                first_character.saldir(second_character)
                if second_character.can <= 0:
                    second_character.can = 0
            pygame.time.wait(1000)

        if first_character.can <= 0 or second_character.can <= 0:
            running = False

        arayuz.screen.fill(arayuz.white)
        arayuz.health_bar(100, 50, 250, 25, first_character.isim, first_character.max_can, first_character.can)
        arayuz.health_bar(450, 50, 250, 25, second_character.isim, second_character.max_can, second_character.can)

        pygame.display.flip()
        clock.tick(30)

    print("Kazanan:", first_character.isim if first_character.can > 0 else second_character.isim)
    pygame.time.wait(2000)

def main():
    pygame.init()
    arayuz = Arayuz()
    clock = pygame.time.Clock()
    running = True

    first_selected = None
    second_selected = None
    selection_order = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if first_selected and second_selected:
                    continue
                pos = pygame.mouse.get_pos()
                for rect, name in arayuz.get_buttons():
                    if rect.collidepoint(pos):
                        if selection_order == 1:
                            first_selected = name
                            selection_order = 2
                        elif selection_order == 2:
                            if name == first_selected:
                                print("Aynı karakteri seçemezsin!")
                            else:
                                second_selected = name

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if first_selected and second_selected:
                        k1 = karakter_sec(first_selected)
                        k2 = karakter_sec(second_selected)
                        if k1 and k2:
                            game_loop(k1, k2)
                        else:
                            print("Karakter seçimi geçersiz!")

        arayuz.ciz(first_selected, second_selected, selection_order)
        clock.tick(30)

    arayuz.quit()
    sys.exit()

if __name__ == "__main__":
    main()
