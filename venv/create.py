__credits__ = ["Andrea PIERRÉ"]
import sys
import math
import subprocess
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

# Create Environment
# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 400
BLACK = (0,0,0)
WHITE = (255,255,255)
BACKGROUND_COLOR = BLACK
SHAPES = []


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
    for shape in SHAPES:
        shape.draw(screen)
        
    for event in pygame.event.get():

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
    

    pygame.display.flip()
    clock.tick(60)
    if event.type == pygame.QUIT:
            for s in SHAPES:
                print(s)
            running = False
            pygame.quit()

# Drag and Drop Activity Finished



print(SHAPES)

