from game.entities import *
from .settings import *
import pygame as pg


# Klasa, która zawiera w sobie logikę gry
class Game:
    def __init__(self) -> None:
        # Inicjacja i początkowe ustawienia
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption("Shooting game - AI")
        self.clock = pg.time.Clock()
        self.running = True
        self.font = pg.font.SysFont("Comic Sans", 30)

    # Główna pętla gry
    def run(self) -> None:
        while self.running:

            # Sprawdzanie zdarzeń w grze
            for event in pg.event.get():
                if event.type == pg.QUIT or (
                    event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE
                ):
                    self.running = False

            # Dalsze części pętli gry - rysowanie zaktualizowanych postaci,
            # aktualizowanie ekranu i ustawienie FPSów
            self.draw_characters()
            pg.display.flip()
            self.clock.tick(FPS)

    # Metoda, która rysuje wszystkie obiekty na ekranie
    def draw_characters(self) -> None:
        self.screen.fill(BLACK)
        text: pg.Surface = self.font.render("Hello World", True, WHITE)
        self.screen.blit(text, (20, 20))
