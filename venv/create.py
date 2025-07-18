import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 400
BLACK = (0,0,0)
WHITE = (255,255,255)
BACKGROUND_COLOR = BLACK
SHAPES = []
PALETTE = []



def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    screen.fill(BLACK)
    pygame.display.set_caption("Drag and Drop Shapes")
    clock = pygame.time.Clock()
    pygame.draw.circle(screen, WHITE, (100, 100), 30)
    pygame.display.flip()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
    clock.tick(60)
main()


'''
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

# Create palette shapes
PALETTE.append(Shape('circle', WHITE, (50, 50)))
PALETTE.append(Shape('square', WHITE, (150, 50)))
PALETTE.append(Shape('triangle', WHITE, (250, 50)))

# Main loop
def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Drag and Drop Shapes")
    clock = pygame.time.Clock()
    dragging_shape = None
    dragging_from_palette = None

    while True:
        screen.fill(BACKGROUND_COLOR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check if clicking on palette shapes
                for shape in PALETTE:
                    if shape.rect and shape.rect.collidepoint(event.pos):
                        new_shape = Shape(dragging_from_palette.shape_type, dragging_from_palette.color,event.pos)
                        SHAPES.append(new_shape)
                        new_shape.draw()
                        dragging_from_palette = shape
                        break

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
                elif dragging_from_palette:
                    # Create a new shape from the palette
                    #new_shape = Shape(dragging_from_palette.shape_type, dragging_from_palette.color, event.pos)
                    #SHAPES.append(new_shape)
                    dragging_from_palette = None

            if event.type == pygame.MOUSEMOTION:
                if dragging_shape:
                    dragging_shape.position = event.pos
                elif dragging_from_palette:
                    # Update position of the shape being dragged from the palette
                    dragging_from_palette.position = event.pos

        # Draw palette shapes
        for shape in PALETTE:
            shape.draw(screen)

        # Draw canvas shapes
        for shape in SHAPES:
            shape.draw(screen)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
'''