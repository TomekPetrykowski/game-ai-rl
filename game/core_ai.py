from game.entities import *
from .settings import *
from .types import *
import pygame as pg
import random
import numpy as np


class ShootingGameEnv:

    def __init__(self, seed=None, max_steps=-1, render_mode=False):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT)) if render_mode else None
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont("Comic Sans", 30) if render_mode else None
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.speed = 1
        self.actions = [
            Action.LEFT.value,
            Action.RIGHT.value,
            Action.SHOOT.value,
            Action.NONE.value,
        ]
        # self.seed(seed)
        self._random = random.Random(seed) if seed is not None else random.Random()
        self._seed = seed
        self.reset()

    # def seed(self, seed=None):
    #     self._random = random.Random(seed)
    #     np.random.seed(seed)
    #     self._seed = seed

    def reset(self):
        self.player = Player(WIDTH // 2 - 25, HEIGHT - 50)
        self.bullets = []
        self.targets = []
        self.score = 0
        self.target_spawn_timer = 0
        self.target_spawn_delay = SPAWN_RATE
        self.done = False
        self.ticks = 0
        # return self.get_state()

    def step(self, action):
        self._handle_action(action)
        self._spawn_targets()
        self._update_entities()
        self._check_collisions()

        self.ticks += 1

        if self.max_steps > 0 and self.ticks > self.max_steps:
            self.done = True

        if self.score < -500:
            self.done = True
            self.score = -500

        if self.score >= 200:
            self.done = True
            self.score = 200

        if self.render_mode:
            self.render()

        # state, reward, done
        return self.get_state(), self.score, self.done

    def get_state(self):
        grid_size = 20
        grid = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
        cell_w, cell_h = WIDTH / grid_size, HEIGHT / grid_size

        # Player
        player_cells = self._get_covered_cells(
            self.player.rect, cell_w, cell_h, grid_size
        )
        for gx, gy in player_cells:
            grid[gy, gx, 0] = 1.0

        # Bullets
        for bullet in self.bullets:
            bullet_cells = self._get_covered_cells(
                bullet.rect, cell_w, cell_h, grid_size
            )
            for gx, gy in bullet_cells:
                grid[gy, gx, 1] += 1.0

        # Targets
        for target in self.targets:
            target_cells = self._get_covered_cells(
                target.rect, cell_w, cell_h, grid_size
            )
            for gx, gy in target_cells:
                if target.target_type == TargetType.OPPONENT:
                    grid[gy, gx, 2] += 1.0
                else:
                    grid[gy, gx, 2] -= 1.0

        return grid

    def _get_covered_cells(self, rect, cell_w, cell_h, grid_size):
        left = int(rect.left // cell_w)
        right = int((rect.right - 1) // cell_w)
        top = int(rect.top // cell_h)
        bottom = int((rect.bottom - 1) // cell_h)
        cells = []
        for gx in range(max(0, left), min(grid_size, right + 1)):
            for gy in range(max(0, top), min(grid_size, bottom + 1)):
                cells.append((gx, gy))
        return cells

    def render(self):
        if not self.screen:
            return

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                exit()

        self.screen.fill(BLACK)
        self.player.draw(self.screen)

        for bullet in self.bullets:
            bullet.draw(self.screen)

        for target in self.targets:
            target.draw(self.screen)

        if self.font:
            score_text = self.font.render(f"Score: {self.score}", True, WHITE)
            self.screen.blit(score_text, (10, 10))

        pg.display.flip()
        self.clock.tick(int(FPS * self.speed))

    def close(self):
        pg.quit()

    def _handle_action(self, action):
        if action == Action.LEFT.value:
            self.player.move_left()
        elif action == Action.RIGHT.value:
            self.player.move_right()
        elif action == Action.SHOOT.value:
            if self.player.shoot():
                bullet = Bullet(self.player.rect.centerx - 2, self.player.rect.top)
                self.bullets.append(bullet)
        else:
            self.player.update()

    def _spawn_targets(self):
        self.target_spawn_timer += 1
        if self.target_spawn_timer >= self.target_spawn_delay:
            self.target_spawn_timer = 0
            x = self._random.randint(0, WIDTH - 30)
            target_type = (
                TargetType.OPPONENT
                if self._random.random() > SPAWN_CHANCE_ALLY
                else TargetType.ALLY
            )
            target = Target(x, -30, target_type, rng=self._random)
            self.targets.append(target)

    def _update_entities(self):
        for bullet in self.bullets[:]:
            bullet.update()
            if bullet.is_off_screen():
                self.bullets.remove(bullet)

        for target in self.targets[:]:
            target.update()
            if target.is_off_screen():
                self.score += target.no_collision_reward
                self.targets.remove(target)

    def _check_collisions(self):
        for bullet in self.bullets[:]:
            for target in self.targets[:]:
                if bullet.rect.colliderect(target.rect):
                    self.score += target.reward_value
                    self.bullets.remove(bullet)
                    self.targets.remove(target)
                    break

        for target in self.targets[:]:
            if target.rect.colliderect(self.player.rect):
                self.score += target.collision_reward
                self.targets.remove(target)
                break
