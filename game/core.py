from game.entities import *
from .settings import *
from .types import *
import pygame as pg
import random


class Game:
    def __init__(self) -> None:
        # Inicjacja i początkowe ustawienia
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption("Shooting game - AI")
        self.clock = pg.time.Clock()
        self.running = True
        self.font = pg.font.SysFont("Comic Sans", 30)

        # Game entities
        self.player = Player(WIDTH // 2 - 25, HEIGHT - 50)
        self.bullets = []
        self.targets = []

        # Game state
        self.score = 0
        self.target_spawn_timer = 0.0
        self.target_spawn_delay = SPAWN_RATE_SECONDS

    def handle_input(self, dt):
        """Handle player input"""
        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT] or keys[pg.K_a]:
            self.player.move_left(dt)
        if keys[pg.K_RIGHT] or keys[pg.K_d]:
            self.player.move_right(dt)
        if keys[pg.K_SPACE]:
            if self.player.shoot(dt):
                bullet = Bullet(self.player.rect.centerx - 2, self.player.rect.top)
                self.bullets.append(bullet)

    def spawn_targets(self, dt):
        """Spawn targets randomly"""
        self.target_spawn_timer += dt
        if self.target_spawn_timer >= self.target_spawn_delay:
            self.target_spawn_timer = 0.0
            x = random.randint(0, WIDTH - 30)
            target_type = (
                TargetType.OPPONENT
                if random.random() > SPAWN_CHANCE_ALLY
                else TargetType.ALLY
            )
            target = Target(x, -30, target_type)
            self.targets.append(target)

    def update_entities(self, dt):
        """Update all game entities"""
        self.player.update(dt)

        # Update bullets
        for bullet in self.bullets[:]:
            bullet.update(dt)
            if bullet.is_off_screen():
                self.bullets.remove(bullet)

        # Update targets
        for target in self.targets[:]:
            target.update(dt)
            if target.is_off_screen():
                self.score += target.no_collision_reward
                self.targets.remove(target)

    def check_collisions(self):
        """Check for collisions"""
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

    def run(self) -> None:
        while self.running:

            raw_dt = self.clock.tick(FPS) / 1000.0
            dt = raw_dt * GAME_SPEED_MULTIPLIER

            for event in pg.event.get():
                if event.type == pg.QUIT or (
                    event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE
                ):
                    self.running = False

            # Game logic
            self.handle_input(dt)
            self.spawn_targets(dt)
            self.update_entities(dt)
            self.check_collisions()

            # Rendering
            self.draw_everything()
            pg.display.flip()

    def draw_everything(self) -> None:
        """Draw all game objects"""
        self.screen.fill(BLACK)

        # Draw entities
        self.player.draw(self.screen)
        for bullet in self.bullets:
            bullet.draw(self.screen)
        for target in self.targets:
            target.draw(self.screen)

        # Draw UI
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
