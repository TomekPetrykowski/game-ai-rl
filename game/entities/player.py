import pygame as pg
from game.settings import *


class Player:
    def __init__(self, x: int, y: int, width: int = 50, height: int = 30):
        self.x = float(x)
        self.y = float(y)
        self.rect = pg.Rect(x, y, width, height)
        self.speed = PLAYER_SPEED
        self.shoot_cooldown = 0.0
        self.color = WHITE

    def move_left(self, dt):
        movement = self.speed * dt
        if self.rect.left > 0:
            self.x = max(0, self.x - movement)
            self.rect.x = int(self.x)

    def move_right(self, dt):
        movement = self.speed * dt
        if self.rect.right < WIDTH:
            self.x = min(WIDTH - self.rect.width, self.x + movement)
            self.rect.x = int(self.x)

    def can_shoot(self):
        return self.shoot_cooldown <= 0

    def shoot(self, dt):
        if self.can_shoot():
            self.shoot_cooldown = SHOOT_COOLDOWN_SECONDS
            return True
        return False

    def update(self, dt):
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= dt
        self.rect.x = int(self.x)
        self.rect.y = int(self.y)

    def draw(self, screen):
        pg.draw.rect(screen, self.color, self.rect)
