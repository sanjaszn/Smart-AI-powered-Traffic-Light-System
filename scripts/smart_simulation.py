import pygame
import random
import time

# Initialize pygame
pygame.init()

# Screen setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Traffic Light Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
GREY = (120, 120, 120)

# Clock
clock = pygame.time.Clock()

# Road and signal positions
lanes = {
    'North': {'pos': (375, -50), 'dir': (0, 1)},
    'South': {'pos': (425, HEIGHT + 50), 'dir': (0, -1)},
    'East': {'pos': (WIDTH + 50, 275), 'dir': (-1, 0)},
    'West': {'pos': (-50, 325), 'dir': (1, 0)},
}

signal_positions = {
    'North': (360, 200),
    'South': (440, 400),
    'East': (500, 260),
    'West': (300, 340),
}

# Car class
class Car:
    def __init__(self, direction):
        self.direction = direction
        self.x, self.y = lanes[direction]['pos']
        self.dir_x, self.dir_y = lanes[direction]['dir']
        self.stopped = False

    def move(self, green_direction):
        if self.direction != green_direction:
            self.stopped = True
        else:
            self.stopped = False

        if not self.stopped:
            self.x += self.dir_x * 2
            self.y += self.dir_y * 2

    def draw(self, screen):
        pygame.draw.rect(screen, BLACK, (self.x, self.y, 20, 10))

# Generate cars periodically
spawn_timer = 0
spawn_interval = 60  # frames
cars = []

# Traffic light logic
green_duration = 5  # seconds
last_switch_time = time.time()
green_direction = 'South'  # Start with South
order = ['South', 'East', 'North', 'West']
order_index = 0

# Main loop
running = True
while running:
    screen.fill(GREY)
    current_time = time.time()

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Switch green light
    if current_time - last_switch_time > green_duration:
        order_index = (order_index + 1) % len(order)
        green_direction = order[order_index]
        last_switch_time = current_time

    # Spawn cars
    spawn_timer += 1
    if spawn_timer >= spawn_interval:
        spawn_timer = 0
        new_car = Car(random.choice(['North', 'South', 'East', 'West']))
        cars.append(new_car)

    # Update and draw cars
    for car in cars:
        car.move(green_direction)
        car.draw(screen)

    # Draw traffic lights
    for direction, pos in signal_positions.items():
        color = GREEN if direction == green_direction else RED
        pygame.draw.circle(screen, color, pos, 10)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
