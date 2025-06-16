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
        self.target_spawn_timer = 0
        self.target_spawn_delay = SPAWN_RATE

    def handle_input(self):
        """Handle player input"""
        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT] or keys[pg.K_a]:
            self.player.move_left()
        if keys[pg.K_RIGHT] or keys[pg.K_d]:
            self.player.move_right()
        if keys[pg.K_SPACE]:
            if self.player.shoot():
                bullet = Bullet(self.player.rect.centerx - 2, self.player.rect.top)
                self.bullets.append(bullet)

    def spawn_targets(self):
        """Spawn targets randomly"""
        self.target_spawn_timer += 1
        if self.target_spawn_timer >= self.target_spawn_delay:
            self.target_spawn_timer = 0
            x = random.randint(0, WIDTH - 30)
            target_type = (
                TargetType.OPPONENT
                if random.random() > SPAWN_CHANCE_ALLY
                else TargetType.ALLY
            )
            target = Target(x, -30, target_type)
            self.targets.append(target)

    def update_entities(self):
        """Update all game entities"""
        self.player.update()

        # Update bullets
        for bullet in self.bullets[:]:
            bullet.update()
            if bullet.is_off_screen():
                self.bullets.remove(bullet)

        # Update targets
        for target in self.targets[:]:
            target.update()
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

            for event in pg.event.get():
                if event.type == pg.QUIT or (
                    event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE
                ):
                    self.running = False

            # Game logic
            self.handle_input()
            self.spawn_targets()
            self.update_entities()
            self.check_collisions()

            # Rendering
            self.draw_everything()
            pg.display.flip()
            self.clock.tick(FPS)

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
