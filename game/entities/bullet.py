import pygame as pg
from game.settings import *


class Bullet:
    def __init__(self, x: int, y: int):
        self.rect = pg.Rect(x, y, 5, 10)
        self.speed = BULLET_SPEED
        self.color = ORANGE

    def update(self):
        movement = self.speed
        self.rect.y -= movement

    def is_off_screen(self):
        return self.rect.bottom < 0

    def draw(self, screen):
        pg.draw.rect(screen, self.color, self.rect)
