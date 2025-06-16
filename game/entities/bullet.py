import pygame as pg
from game.settings import *


class Bullet:
    def __init__(self, x: int, y: int):
        self.x = float(x)
        self.y = float(y)
        self.rect = pg.Rect(x, y, 5, 10)
        self.speed = BULLET_SPEED
        self.color = ORANGE

    def update(self, dt):
        movement = self.speed * dt
        self.y = self.y - movement
        self.rect.y = int(self.y)

    def is_off_screen(self):
        return self.rect.bottom < 0

    def draw(self, screen):
        pg.draw.rect(screen, self.color, self.rect)
