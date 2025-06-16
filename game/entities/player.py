import pygame as pg
from game.settings import *


class Player:
    def __init__(self, x: int, y: int, width: int = 50, height: int = 30):
        self.rect = pg.Rect(x, y, width, height)
        self.speed = PLAYER_SPEED
        self.shoot_cooldown = 0.0
        self.color = WHITE

    def move_left(self):
        movement = self.speed
        if self.rect.left > 0:
            self.rect.x = max(0, self.rect.x - movement)

    def move_right(self):
        movement = self.speed
        if self.rect.right < WIDTH:
            self.rect.x = min(WIDTH - self.rect.width, self.rect.x + movement)

    def can_shoot(self):
        return self.shoot_cooldown <= 0

    def shoot(self):
        if self.can_shoot():
            self.shoot_cooldown = SHOOT_COOLDOWN_SECONDS
            return True
        return False

    def update(self):
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

    def draw(self, screen):
        pg.draw.rect(screen, self.color, self.rect)
