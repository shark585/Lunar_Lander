__credits__ = ["Andrea PIERRÉ"]
import sys
import math
from typing import TYPE_CHECKING, Optional

import numpy as np

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle
from gymnasium.utils.step_api_compatibility import step_api_compatibility


try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e

import pygame


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12
MAIN_ENGINE_Y_LOCATION = (
    4  # The Y location of the main engine on the body of the Lander.
)

VIEWPORT_W = 600
VIEWPORT_H = 400


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.lander == contact.fixtureA.body
            or self.env.lander == contact.fixtureB.body
        ):
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


# Create Environment
# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 400
BLACK = (0,0,0)
WHITE = (255,255,255)
BACKGROUND_COLOR = BLACK
SHAPES = []
PALETTE = []


# Shape class
class Shape: 
    def __init__(self, shape_type, color, position):
        self.shape_type = shape_type
        self.color = color
        self.position = position
        self.rect = None
        self.selected = False

    def draw(self, surface):
        if self.shape_type == 'circle':
            pygame.draw.circle(surface, self.color, self.position, 30)
            self.rect = pygame.Rect(self.position[0] - 30, self.position[1] - 30, 60, 60)
        elif self.shape_type == 'square':
            pygame.draw.rect(surface, self.color, (self.position[0] - 30, self.position[1] - 30, 60, 60))
            self.rect = pygame.Rect(self.position[0] - 30, self.position[1] - 30, 60, 60)
        elif self.shape_type == 'triangle':
            points = [(self.position[0], self.position[1] - 30), 
                    (self.position[0] - 30, self.position[1] + 30), 
                    (self.position[0] + 30, self.position[1] + 30)]
            pygame.draw.polygon(surface, self.color, points)
            self.rect = pygame.Rect(self.position[0] - 30, self.position[1] - 30, 60, 60)
    
    def __repr__(self):
        return "Shape: " + self.shape_type + str(self.position)

#Pygame Script

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Create Your Own Environment")
clock = pygame.time.Clock()
dragging_shape = None
dragging_from_palette = None

running = True

while running:
    screen.fill(BACKGROUND_COLOR)
    pygame.draw.circle(screen, WHITE, (50,50), 30)
    shape_circle = pygame.Rect(20, 20, 60, 60)

    pygame.draw.rect(screen, WHITE, (120, 30, 60, 60))
    shape_square = pygame.Rect(120,30,60,60) 

    points1 = [(250, 20), 
                    (220, 80), 
                    (280, 80)]
    pygame.draw.polygon(screen, WHITE, points1)
    shape_triangle = pygame.Rect(220, 20, 60, 60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            for s in SHAPES:
                print(s)
            running = False
            pygame.quit()



        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            if shape_circle.collidepoint(mouse_x, mouse_y):
                new_shape = Shape('circle',WHITE, event.pos)
                SHAPES.append(new_shape)
                dragging_shape = new_shape

            elif shape_square.collidepoint(mouse_x, mouse_y):
                new_shape = Shape('square', WHITE, event.pos)
                SHAPES.append(new_shape)
                dragging_shape = new_shape

            elif shape_triangle.collidepoint(mouse_x, mouse_y):
                new_shape = Shape('triangle', WHITE, event.pos)
                SHAPES.append(new_shape)
                dragging_shape = new_shape

            # Check if clicking on canvas shapes
            if not dragging_from_palette:
                for shape in SHAPES:
                    if shape.rect and shape.rect.collidepoint(event.pos):
                        dragging_shape = shape
                        shape.selected = True

        if event.type == pygame.MOUSEBUTTONUP:
            if dragging_shape:
                dragging_shape.selected = False
                dragging_shape = None

        if event.type == pygame.MOUSEMOTION:
            if dragging_shape:
                dragging_shape.position = event.pos

    # Draw canvas shapes
    for shape in SHAPES:
        shape.draw(screen)

    pygame.display.flip()
    clock.tick(60)


