import pygame as pg
import random
from game.settings import *
from game import *


class Target:
    def __init__(
        self, x: int, y: int, target_type: TargetType = TargetType.OPPONENT, rng=None
    ):
        self.rect = pg.Rect(x, y, 30, 30)
        self.target_type = target_type

        self.rng = rng if rng is not None else random

        self.speed = self.rng.uniform(TARGET_SPEED_MIN, TARGET_SPEED_MAX)
        self.color = (255, 0, 0) if target_type == TargetType.OPPONENT else (0, 0, 255)
        self.reward_value = (
            SHOOT_REWARD_OPPONENT
            if target_type == TargetType.OPPONENT
            else SHOOT_REWARD_ALLY
        )
        self.collision_reward = (
            COLLISION_REWARD_ALLY
            if target_type == TargetType.ALLY
            else COLLISION_REWARD_OPPONENT
        )
        self.no_collision_reward = (
            NO_COLLISION_REWARD_ALLY
            if target_type == TargetType.ALLY
            else NO_COLLISION_REWARD_OPPONENT
        )

    def update(self):
        movement = self.speed
        self.rect.y += movement

    def is_off_screen(self):
        return self.rect.top > HEIGHT

    def draw(self, screen):
        pg.draw.rect(screen, self.color, self.rect)
