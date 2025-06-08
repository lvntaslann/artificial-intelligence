import pygame

class Arayuz:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode(size=(self.width, self.height))
        pygame.display.set_caption("KAOS: Topkapı Sarayı")

        # Renkler
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.gray = (200, 200, 200)
        self.blue = (100, 149, 237)
        self.green = (34, 177, 76)
        self.hover_color = (180, 180, 250)

        # Yazı tipi
        self.font = pygame.font.SysFont(None, 36)

        # Karakterler
        self.karakterler = ["Hürrem", "Süleyman", "Mihrima", "Mustafa",
                            "Mehmet", "Selim", "Cihangir"]
        self.buttons = []
        self.button_width = 150
        self.button_height = 50
        self.margin_x = 40
        self.margin_y = 40
        self.cols = 4
        self.start_x = 60
        self.start_y = 120

        self._olustur_butonlar()

    def health_bar(self, x, y, w, h, isim, character_max_hp, character_hp):
        ratio = character_hp / character_max_hp
        pygame.draw.rect(self.screen, "red", (x, y, w, h))
        pygame.draw.rect(self.screen, "green", (x, y, w * ratio, h))

        isim_yazi = self.font.render(f"{isim}", True, self.black)
        can_yazi = self.font.render(f"{character_hp}/{character_max_hp}", True, self.black)

        self.screen.blit(isim_yazi, (x + (w - isim_yazi.get_width()) // 2, y - 30))
        self.screen.blit(can_yazi, (x + (w - can_yazi.get_width()) // 2, y + h + 5))

    def _olustur_butonlar(self):
        self.buttons.clear()
        for index, name in enumerate(self.karakterler):
            row = index // self.cols
            col = index % self.cols
            x = self.start_x + col * (self.button_width + self.margin_x)
            y = self.start_y + row * (self.button_height + self.margin_y)
            rect = pygame.Rect(x, y, self.button_width, self.button_height)
            self.buttons.append((rect, name))

    def ciz(self, first_selected, second_selected, selection_order):
        self.screen.fill(self.white)
        title = self.font.render(f"Karakter Seçimi - Sırada: {selection_order}. Karakter", True, self.black)
        self.screen.blit(title, ((self.width - title.get_width()) // 2, 30))

        mouse_pos = pygame.mouse.get_pos()

        for rect, name in self.buttons:
            is_selected = name == first_selected or name == second_selected
            color = self.green if is_selected else self.gray

            if rect.collidepoint(mouse_pos) and not is_selected:
                color = self.hover_color

            pygame.draw.rect(self.screen, color, rect, border_radius=10)
            if is_selected:
                pygame.draw.rect(self.screen, self.blue, rect, width=3, border_radius=10)

            text = self.font.render(name, True, self.black)
            self.screen.blit(
                text,
                (rect.x + (self.button_width - text.get_width()) // 2,
                 rect.y + (self.button_height - text.get_height()) // 2)
            )

        if first_selected:
            first_text = self.font.render(f"1. Karakter: {first_selected}", True, self.black)
            self.screen.blit(first_text, (50, self.height - 100))

        if second_selected:
            second_text = self.font.render(f"2. Karakter: {second_selected}", True, self.black)
            self.screen.blit(second_text, (self.width - 350, self.height - 100))

        if first_selected and second_selected:
            start_text = self.font.render("Enter'a basarak oyunu başlat", True, self.black)
            self.screen.blit(start_text, ((self.width - start_text.get_width()) // 2, self.height - 50))

        pygame.display.flip()

    def get_buttons(self):
        return self.buttons

    def quit(self):
        pygame.quit()
