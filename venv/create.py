import pygame
import sys
import gymnasium as gym
from gymnasium import spaces
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


class CustomLunarLanderEnv(gym.Env):
    def __init__(self):
        super(CustomLunarLanderEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # Example: 4 actions
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        self.screen = pygame.display.set_mode((width, height))
        self.background = pygame.image.load("drawing.png")

    def reset(self):
        # Reset the environment state
        return self._get_observation()

    def step(self, action):
        # Implement the logic for taking a step in the environment
        # Update the state based on the action
        # Return the new state, reward, done, and info
        assert self.lander is not None

        # Update wind
        assert self.lander is not None, "You forgot to call reset()"
        if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi * 0.01 * self.wind_idx))
                )
                * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = math.tanh(
                math.sin(0.02 * self.torque_idx)
                + (math.sin(math.pi * 0.01 * self.torque_idx))
            ) * (self.turbulence_power)
            self.torque_idx += 1
            self.lander.ApplyTorque(
                (torque_mag),
                True,
            )

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            # 4 is move a bit downwards, +-2 for randomness
            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(
                3.5,  # 3.5 is here to make particle speed adequate
                impulse_pos[0],
                impulse_pos[1],
                m_power,
            )  # particles are just a decoration
            p.ApplyLinearImpulse(
                (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action - 2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(
                (ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        # Compare with / without shaping, referring the state description below
        '''
        state[0]: the horizontal coordinate
        state[1]: the vertical coordinate
        state[2]: the horizontal speed
        state[3]: the vertical speed
        state[4]: the angle
        state[5]: the angular speed
        state[6]: first leg contact
        state[7]: second leg contact
        '''
        reward = 0
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= (
            m_power * 0.30
        )  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power * 0.03

        terminated = False
        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            reward = -100
        if not self.lander.awake:
            terminated = True
            reward = +100

        if self.render_mode == "human":
            self.render()
        return np.array(state, dtype=np.float32), reward, terminated, False, {}
        return observation, reward, done, info

    def render(self, mode='human'):
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()

    def _get_observation(self):
        # Return the current observation
        return self.screen.copy()


# Main loop
def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Create Your Own Environment")
    clock = pygame.time.Clock()
    dragging_shape = None
    dragging_from_palette = None

    while True:
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
                pygame.image.save(screen, 'drawing.png')
                pygame.quit()
                sys.exit()



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

if __name__ == "__main__":
    main()
