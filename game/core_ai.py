from game.entities import *
from .settings import *
from .types import *
import pygame as pg
import random
import numpy as np


class ShootingGameEnv:

    def __init__(
        self, seed=1, max_steps=-1, render_mode=False, true_seed=False, endless=False
    ):
        pg.init() if render_mode else None
        self.screen = pg.display.set_mode((WIDTH, HEIGHT)) if render_mode else None
        self.clock = pg.time.Clock() if render_mode else None
        self.font = pg.font.SysFont("Ubuntu", 30) if render_mode else None
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.true_seed = true_seed
        self.endless = endless
        self.speed = 1
        self.actions = [
            Action.NONE.value,
            Action.LEFT.value,
            Action.RIGHT.value,
            Action.SHOOT.value,
        ]
        self._random = random.Random(seed)
        self._seed = seed
        self.reset()

    def reset(self):
        if self.true_seed:
            self._random = random.Random(self._seed)

        self.player = Player(WIDTH // 2 - 25, HEIGHT - 50)
        self.bullets = []
        self.targets = []
        self.score = 0
        self.target_spawn_timer = 0
        self.target_spawn_delay = SPAWN_RATE
        self.done = False
        self.ticks = 0
        self.last_action = None

    def step(self, action):
        self.last_action = action
        prev_player_x = self.player.rect.centerx

        self._handle_action(action)
        self._spawn_targets()
        self._update_entities()
        self._check_collisions()

        reward = self._calculate_positioning_reward(prev_player_x)

        if self.max_steps > 0 and self.ticks > self.max_steps:
            self.done = True

        if not self.endless:
            if self.score < -500 or self.score >= 300:
                self.done = True

        if self.render_mode:
            self.render()

        self.ticks += 1
        # state, reward, score, done
        return self.get_state(), reward, self.score, self.done

    def _calculate_positioning_reward(self, prev_player_x):
        allies = [t for t in self.targets if t.target_type == TargetType.ALLY]
        if not allies:
            return 0.0

        # closest ally (by vertical distance)
        closest_ally = min(
            allies, key=lambda t: abs(t.rect.centery - self.player.rect.centery)
        )

        # current and previous alignment
        current_x_diff = abs(self.player.rect.centerx - closest_ally.rect.centerx)
        prev_x_diff = abs(prev_player_x - closest_ally.rect.centerx)

        # reward for moving towards allies
        if current_x_diff < prev_x_diff:
            return 0.2
        elif current_x_diff > prev_x_diff:
            return -0.1

        # Bonus for being close to the ally
        if current_x_diff < 30:
            return 0.5

        return 0.0

    def get_state(self):
        MAX_ALLIES = 3
        player_x = self.player.rect.centerx / WIDTH

        if hasattr(self, "last_action"):
            if self.last_action == Action.LEFT.value:
                move_dir = -1
            elif self.last_action == Action.RIGHT.value:
                move_dir = 1
            else:
                move_dir = 0
        else:
            move_dir = 0

        allies = [t for t in self.targets if t.target_type == TargetType.ALLY]
        allies.sort(key=lambda t: abs(t.rect.centery - self.player.rect.centery))

        ally_features = []
        for i in range(MAX_ALLIES):
            if i < len(allies):
                ally = allies[i]
                rel_x = (ally.rect.centerx - self.player.rect.centerx) / WIDTH
                rel_y = (ally.rect.centery - self.player.rect.centery) / HEIGHT
                ally_features.extend([rel_x, rel_y])
            else:
                ally_features.extend([0.0, -2.0])  # default

        closest_ally_alignment = 0.0
        if allies:
            closest_ally = allies[0]
            x_diff = abs(closest_ally.rect.centerx - self.player.rect.centerx) / WIDTH
            closest_ally_alignment = max(0.0, 1.0 - x_diff * 2)

        state = [player_x, move_dir, closest_ally_alignment] + ally_features

        return np.array(state, dtype=np.float32)

    def render(self):
        if not self.screen:
            return

        if not self.clock:
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
            target = Target(
                x,
                -30,
                self._random,
                target_type,
            )
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
