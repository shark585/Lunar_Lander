__credits__ = ["Andrea PIERRÉ"]
import subprocess
import io
import os
import glob
import torch
import base64
import minigrid
import sys
import math
from typing import TYPE_CHECKING, Optional
import vlc 
import numpy as np
import matplotlib as plt
import matplotlib
import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle
from gymnasium.utils.step_api_compatibility import step_api_compatibility
#from gymnasium.envs.box2d.lunar_lander import *
from gymnasium.wrappers import RecordVideo
from ale_py import ALEInterface
import warnings


import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.results_plotter import ts2xy, load_results
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env

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


from IPython.display import HTML
from base64 import b64encode
from pyvirtualdisplay import Display
import pygame


# Constants for Lunar Lander

PIXELS_PER_METER = 30
CATEGORY_STATIC = 0x0001  # Example category for static bodies
CATEGORY_DYNAMIC = 0x0002  # Example category for dynamic bodies

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


# Environment Creation Start

# Constants for Pygame
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

# Points Methods

def circle_points(center, radius, num_points=1000):
    return [(int(center[0] + radius * math.cos(2 * math.pi * i / num_points)),
    int(center[1] + radius * math.sin(2 * math.pi * i / num_points)))
    for i in range(num_points)]

def square_points(top_left, size):
    x, y = top_left
    return [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]



#Create Shape in Box2d






# Initialize Pygame
pygame.init()
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

    

    pygame.display.flip()
    clock.tick(60)

    if event.type == pygame.QUIT:
            running = False
            pygame.quit()

# Drag and Drop Activity Finished
for s in SHAPES:
    if s.position[0] > 600 or s.position[0] < 0 or s.position[1] > 400 or s.position[1] < 0:
        i = SHAPES.index(s)
        SHAPES.pop(i)
        continue
    print(s)



#Lunar Lander Start

class ContactDetector(Box2D.b2ContactListener):
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
        
class CustomEnvLunarLander(gym.Env, EzPickle):
    r"""
    ## Description
    This environment is a classic rocket trajectory optimization problem.
    According to Pontryagin's maximum principle, it is optimal to fire the
    engine at full throttle or turn it off. This is the reason why this
    environment has discrete actions: engine on or off.

    There are two environment versions: discrete or continuous.
    The landing pad is always at coordinates (0,0). The coordinates are the
    first two numbers in the state vector.
    Landing outside of the landing pad is possible. Fuel is infinite, so an agent
    can learn to fly and then land on its first attempt.

    To see a heuristic landing, run:
    ```shell
    python gymnasium/envs/box2d/lunar_lander.py
    ```

    ## Action Space
    There are four discrete actions available:
    - 0: do nothing
    - 1: fire left orientation engine
    - 2: fire main engine
    - 3: fire right orientation engine

    ## Observation Space
    The state is an 8-dimensional vector: the coordinates of the lander in `x` & `y`, its linear
    velocities in `x` & `y`, its angle, its angular velocity, and two booleans
    that represent whether each leg is in contact with the ground or not.

    ## Rewards
    After every step a reward is granted. The total reward of an episode is the
    sum of the rewards for all the steps within that episode.

    For each step, the reward:
    - is increased/decreased the closer/further the lander is to the landing pad.
    - is increased/decreased the slower/faster the lander is moving.
    - is decreased the more the lander is tilted (angle not horizontal).
    - is increased by 10 points for each leg that is in contact with the ground.
    - is decreased by 0.03 points each frame a side engine is firing.
    - is decreased by 0.3 points each frame the main engine is firing.

    The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

    An episode is considered a solution if it scores at least 200 points.

    ## Starting State
    The lander starts at the top center of the viewport with a random initial
    force applied to its center of mass.

    ## Episode Termination
    The episode finishes if:
    1) the lander crashes (the lander body gets in contact with the moon);
    2) the lander gets outside of the viewport (`x` coordinate is greater than 1);
    3) the lander is not awake. From the [Box2D docs](https://box2d.org/documentation/md__d_1__git_hub_box2d_docs_dynamics.html#autotoc_md61),
        a body which is not awake is a body which doesn't move and doesn't
        collide with any other body:
    > When Box2D determines that a body (or group of bodies) has come to rest,
    > the body enters a sleep state which has very little CPU overhead. If a
    > body is awake and collides with a sleeping body, then the sleeping body
    > wakes up. Bodies will also wake up if a joint or contact attached to
    > them is destroyed.

    ## Arguments

    Lunar Lander has a large number of arguments

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
    ...                enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<LunarLander<LunarLander-v3>>>>>

    ```

     * `continuous` determines if discrete or continuous actions (corresponding to the throttle of the engines) will be used with the
     action space being `Discrete(4)` or `Box(-1, +1, (2,), dtype=np.float32)` respectively.
     For continuous actions, the first coordinate of an action determines the throttle of the main engine, while the second
     coordinate specifies the throttle of the lateral boosters. Given an action `np.array([main, lateral])`, the main
     engine will be turned off completely if `main < 0` and the throttle scales affinely from 50% to 100% for
     `0 <= main <= 1` (in particular, the main engine doesn't work  with less than 50% power).
     Similarly, if `-0.5 < lateral < 0.5`, the lateral boosters will not fire at all. If `lateral < -0.5`, the left
     booster will fire, and if `lateral > 0.5`, the right booster will fire. Again, the throttle scales affinely
     from 50% to 100% between -1 and -0.5 (and 0.5 and 1, respectively).

    * `gravity` dictates the gravitational constant, this is bounded to be within 0 and -12. Default is -10.0

    * `enable_wind` determines if there will be wind effects applied to the lander. The wind is generated using
     the function `tanh(sin(2 k (t+C)) + sin(pi k (t+C)))` where `k` is set to 0.01 and `C` is sampled randomly between -9999 and 9999.

    * `wind_power` dictates the maximum magnitude of linear wind applied to the craft. The recommended value for
     `wind_power` is between 0.0 and 20.0.

    * `turbulence_power` dictates the maximum magnitude of rotational wind applied to the craft.
     The recommended value for `turbulence_power` is between 0.0 and 2.0.

    ## Version History
    - v3:
        - Reset wind and turbulence offset (`C`) whenever the environment is reset to ensure statistical independence between consecutive episodes (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/954)).
        - Fix non-deterministic behaviour due to not fully destroying the world (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/728)).
        - Changed observation space for `x`, `y`  coordinates from $\pm 1.5$ to $\pm 2.5$, velocities from $\pm 5$ to $\pm 10$ and angles from $\pm \pi$ to $\pm 2\pi$ (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/752)).
    - v2: Count energy spent and in v0.24, added turbulence with wind power and turbulence_power parameters
    - v1: Legs contact with ground added in state vector; contact with ground give +10 reward points, and -10 if then lose contact; reward renormalized to 200; harder initial random push.
    - v0: Initial version

    ## Notes

    There are several unexpected bugs with the implementation of the environment.

    1. The position of the side thrusters on the body of the lander changes, depending on the orientation of the lander.
    This in turn results in an orientation dependent torque being applied to the lander.

    2. The units of the state are not consistent. I.e.
    * The angular velocity is in units of 0.4 radians per second. In order to convert to radians per second, the value needs to be multiplied by a factor of 2.5.

    For the default values of VIEWPORT_W, VIEWPORT_H, SCALE, and FPS, the scale factors equal:
    'x': 10, 'y': 6.666, 'vx': 5, 'vy': 7.5, 'angle': 1, 'angular velocity': 2.5

    After the correction has been made, the units of the state are as follows:
    'x': (units), 'y': (units), 'vx': (units/second), 'vy': (units/second), 'angle': (radians), 'angular velocity': (radians/second)

    <!-- ## References -->

    ## Credits
    Created by Oleg Klimov
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            continuous,
            gravity,
            enable_wind,
            wind_power,
            turbulence_power,
        )

        assert (
            -12.0 < gravity and gravity < 0.0
        ), f"gravity (current value: {gravity}) must be between -12 and 0"
        self.gravity = gravity

        if 0.0 > wind_power or wind_power > 20.0:
            gym.logger.warn(
                f"wind_power value is recommended to be between 0.0 and 20.0, (current value: {wind_power})"
            )
        self.wind_power = wind_power

        if 0.0 > turbulence_power or turbulence_power > 2.0:
            gym.logger.warn(
                f"turbulence_power value is recommended to be between 0.0 and 2.0, (current value: {turbulence_power})"
            )
        self.turbulence_power = turbulence_power

        self.enable_wind = enable_wind

        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True
        self.world = Box2D.b2World(gravity=(0, gravity))
        self.moon = None
        self.lander: Optional[Box2D.b2Body] = None
        self.particles = []

        self.prev_reward = None

        self.continuous = continuous

        low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -2.5,  # x coordinate
                -2.5,  # y coordinate
                # velocity bounds is 5x rated speed
                -10.0,
                -10.0,
                -2 * math.pi,
                -10.0,
                -0.0,
                -0.0,
            ]
        ).astype(np.float32)
        high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                2.5,  # x coordinate
                2.5,  # y coordinate
                # velocity bounds is 5x rated speed
                10.0,
                10.0,
                2 * math.pi,
                10.0,
                1.0,
                1.0,
            ]
        ).astype(np.float32)

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(low, high)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        self.render_mode = render_mode

    def create_shape(self, position, shape_type):
        if shape_type == 'circle':
            body = self.world.CreateStaticBody(position=position)
            body.CreateCircleFixture(radius=30)
        elif shape_type == 'square':
            body = self.world.CreateStaticBody(position=position)
            body.CreatePolygonFixture(box=(60,60))  # Width, Height
        elif shape_type == 'triangle':
            vertices = [Box2D.b2Vec2(position[0], position[1] - 30),
                        Box2D.b2Vec2(position[0] - 30, position[1] + 30),
                        Box2D.b2Vec2(position[0] + 30, position[1] + 30)]
            body = self.world.CreateStaticBody(position=position)
            body.CreatePolygonFixture(vertices=vertices)
    
        return body

    def _destroy(self):
        if not self.moon:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()

        # Bug's workaround for: https://github.com/Farama-Foundation/Gymnasium/issues/728
        # Not sure why the self._destroy() is not enough to clean(reset) the total world environment elements, need more investigation on the root cause,
        # we must create a totally new world for self.reset(), or the bug#728 will happen
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Create Terrain

        self.helipad_y = H / 4
        
        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 100), (W, 100)])
        )
        
        
        #self.moon.filter.maskBits = CATEGORY_DYNAMIC
        #self.moon.filter.categoryBits = CATEGORY_STATIC


        bodies = []
        for s in SHAPES:
            static_body_def = Box2D.b2BodyDef()
            static_body_def.type = Box2D.b2_staticBody
            static_body_def.position = ((s.position[0]) / SCALE, (400-s.position[1])/ SCALE) 
            static_body = self.world.CreateBody(static_body_def)

# Define a shape for the static body
            if s.shape_type == "square":
                box_shape = Box2D.b2PolygonShape()
                box_shape.SetAsBox(1,1)  # Create a box shape
                static_fixture_def = Box2D.b2FixtureDef()
                static_fixture_def.shape = box_shape
            elif s.shape_type == "circle":
                circle_shape = Box2D.b2CircleShape()
                circle_shape.radius = 1.0  # Set the radius of the circle
                static_fixture_def = Box2D.b2FixtureDef()
                static_fixture_def.shape = circle_shape

            elif s.shape_type == "triangle":
                
                triangle_shape = Box2D.b2CheckPolygon()
                vertices = [
                    Box2D.b2Vec2(s.position[0]/30, (400 - s.position[1] + 30)/30),  # Top vertex
                    Box2D.b2Vec2((s.position[0] - 30)/30, (400 - s.position[1] - 30)/30),  # Bottom left vertex
                    Box2D.b2Vec2((s.position[0] + 30)/30, (400 - s.position[1] - 30)/30)   # Bottom right vertex
                ]
                

                triangle_shape.Set(vertices)
                # Set the vertices for the triangle shape using SetAsArray
                #triangle_shape.SetAsArray(vertices, len(vertices))
                static_fixture_def.shape = triangle_shape
# Create a fixture definition for the static body
            #elif s.shape_type == "triangle":

            static_fixture_def.isSensor = False  # Set to True if you want it to be a sensor
            static_fixture_def.filter.categoryBits = CATEGORY_STATIC  # Set category bits
            static_fixture_def.filter.maskBits = CATEGORY_DYNAMIC  # Set mask bits

# Attach the fixture to the static body
            static_body.CreateFixture(static_fixture_def)
            bodies.append(static_body)


        #self.shape = self.world.CreateStaticBody(shapes = edgeShape(vertices = [(0,100), (W,100)]))

        for s in SHAPES:
            if s.shape_type == 'circle':
                points = circle_points((s.position[0], 400 - s.position[1]), 30)

            elif s.shape_type == 'square':
                points = square_points((s.position[0] - 30, 400 - s.position[1] - 30), 60)
                #pygame.draw.rect(surface, self.color, (self.position[0] - 30, self.position[1] - 30, 60, 60))
            
            elif s.shape_type == 'triangle':
                points = [(s.position[0], 400 - s.position[1] + 30), 
                    (s.position[0] - 30, 400 - s.position[1] - 30), 
                    (s.position[0] + 30, 400 - s.position[1] - 30)]


        self.moon.color1 = (WHITE)
        self.moon.color2 = (WHITE)
        
        # Create Lander body
        initial_y = VIEWPORT_H / SCALE
        initial_x = VIEWPORT_W / SCALE / 2
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=CATEGORY_DYNAMIC,
                maskBits = CATEGORY_STATIC,  # collide only with ground
                restitution=0.0,
            ),  # 0.99 bouncy
        )
        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)

        # Apply the initial random impulse to the lander
        self.lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )

        if self.enable_wind:  # Initialize wind pattern based on index
            self.wind_idx = self.np_random.integers(-9999, 9999)
            self.torque_idx = self.np_random.integers(-9999, 9999)

        # Create Lander Legs
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=CATEGORY_DYNAMIC,
                    maskBits= CATEGORY_STATIC,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = (
                    +0.9 - 0.5
                )  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0, 0]) if self.continuous else 0)[0], {}

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=CATEGORY_DYNAMIC,
                maskBits= 0x00,  # collide only with ground
                restitution=0.3,
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all_particle):
        while self.particles and (all_particle or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def step(self, action):
        assert self.lander is not None

        # Update wind and apply to the lander
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
            torque_mag = (
                math.tanh(
                    math.sin(0.02 * self.torque_idx)
                    + (math.sin(math.pi * 0.01 * self.torque_idx))
                )
                * self.turbulence_power
            )
            self.torque_idx += 1
            self.lander.ApplyTorque(
                torque_mag,
                True,
            )

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float64)
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "

        # Apply Engine Impulses

        # Tip is the (X and Y) components of the rotation of the lander.
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))

        # Side is the (-Y and X) components of the rotation of the lander.
        side = (-tip[1], tip[0])

        # Generate two random numbers between -1/SCALE and 1/SCALE.
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
            # The components of the impulse to be applied by the main engine.
            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )

            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(
                    3.5,  # 3.5 is here to make particle speed adequate
                    impulse_pos[0],
                    impulse_pos[1],
                    m_power,
                )
                p.ApplyLinearImpulse(
                    (
                        ox * MAIN_ENGINE_POWER * m_power,
                        oy * MAIN_ENGINE_POWER * m_power,
                    ),
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
            # Orientation/Side engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                # action = 1 is left, action = 3 is right
                direction = action - 2
                s_power = 1.0

            # The components of the impulse to be applied by the side engines.
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )

            # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
            # However, SIDE_ENGINE_HEIGHT is defined as 14
            # This causes the position of the thrust on the body of the lander to change, depending on the orientation of the lander.
            # This in turn results in an orientation dependent torque being applied to the lander.
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (
                        ox * SIDE_ENGINE_POWER * s_power,
                        oy * SIDE_ENGINE_POWER * s_power,
                    ),
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
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(state, dtype=np.float32), reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[box2d]"`'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        pygame.transform.scale(self.surf, (SCALE, SCALE))
        pygame.draw.rect(self.surf, (0,0,0), self.surf.get_rect())

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
            obj.color2 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )

        self._clean_particles(False)

        print('start shapes')
        for s in SHAPES:
            if s.shape_type == 'circle':
                pygame.draw.circle(self.surf, WHITE, (s.position[0],400-s.position[1]),30)
                continue

            elif s.shape_type == 'square':
                points = square_points((s.position[0] - 30, 400 - s.position[1] - 30), 60)
                #pygame.draw.rect(surface, self.color, (self.position[0] - 30, self.position[1] - 30, 60, 60))
            
            elif s.shape_type == 'triangle':
                points = [(s.position[0], 400 - s.position[1] + 30), 
                    (s.position[0] - 30, 400 - s.position[1] - 30), 
                    (s.position[0] + 30, 400 - s.position[1] - 30)]
            pygame.draw.polygon(self.surf, WHITE, points)
            #gfxdraw.aapolygon(self.surf, points, WHITE)
        print('end shapes')


        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )

                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    pygame.draw.aalines(
                        self.surf, color=obj.color2, points=path, closed=True
                    )


        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False



# Main Script

os.makedirs("./video", exist_ok=True)
display = Display(visible=False, size=(600, 400))
_ = display.start()


def render_mp4(videopath: str) -> str:
  """
  Gets a string containing a b4-encoded version of the MP4 video
  at the specified path.
  """
  mp4 = open(videopath, 'rb').read()
  base64_encoded_mp4 = b64encode(mp4).decode()
  return f'<video width=400 controls><source src="data:video/mp4;' \
         f'base64,{base64_encoded_mp4}" type="video/mp4"></video>'


nn_layers = [64, 64]  # This is the configuration of your neural network. Currently, we have two layers, each consisting of 64 neurons.
                      # If you want three layers with 64 neurons each, set the value to [64,64,64] and so on.

learning_rate = 0.001  # This is the step-size with which the gradient descent is carried out.
                       # Tip: Use smaller step-sizes for larger networks.

log_dir = "./tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

env = CustomEnvLunarLander(
        render_mode = "rgb_array",
        enable_wind = True,
        wind_power = 10.0,
        turbulence_power = 1.3,
)

for s in SHAPES:
    b = env.create_shape(s.position, s.shape_type)

env = stable_baselines3.common.monitor.Monitor(env, log_dir)

callback = EvalCallback(env, log_path=log_dir, deterministic=True)  # For evaluating the performance of the agent periodically and logging the results.
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=nn_layers)
model = DQN("MlpPolicy", env,policy_kwargs = policy_kwargs,
            learning_rate=learning_rate,
            batch_size=32,  # for simplicity, we are not doing batch update.
            buffer_size=10000,  # size of experience of replay buffer. Set to 1 as batch update is not done
            learning_starts=1000,  # learning starts immediately!
            gamma=0.99,  # discount facto. range is between 0 and 1.
            tau = .005,  # the soft update coefficient for updating the target network
            target_update_interval=100,  # update the target network immediately.
            train_freq=(4,"step"),  # train the network at every step.
            max_grad_norm = 10,  # the maximum value for the gradient clipping
            exploration_initial_eps = 1,  # initial value of random action probability
            exploration_fraction = 0.1,  # fraction of entire training period over which the exploration rate is reduced
            gradient_steps = 1,  # number of gradient steps
            seed = 1,  # seed for the pseudo random generators
            verbose=1)  # Set verbose to 1 to observe training logs. We encourage you to set the verbose to 1.

env = CustomEnvLunarLander(
        render_mode = "rgb_array",
        enable_wind = True,
        wind_power = 10.0,
        turbulence_power = 1.3,
)
for s in SHAPES:
    b = env.create_shape(s.position, s.shape_type)

env = gym.wrappers.RecordVideo(
    env,
    video_folder="video",
    name_prefix="CustomEnv",
    episode_trigger=lambda episode_id: True
)

observation, _ = env.reset()
total_reward = 0
done = False

while not done:
    action, states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

env.close()
print(f"\nTotal reward: {total_reward}")

# show video
'''
html = render_mp4("video/CustomEnv_pretraining-episode-0.mp4")
HTML(html)
'''

model.learn(total_timesteps=100000, log_interval=10000, callback=callback)
# The performance of the training will be printed every 10000 episodes. Change it to 1, if you wish to
# view the performance at every training episode.

env = CustomEnvLunarLander(
        render_mode = "rgb_array",
        enable_wind = True,
        wind_power = 10.0,
        turbulence_power = 1.3,
)
for s in SHAPES:
    b = env.create_shape(s.position, s.shape_type)

env = gym.wrappers.RecordVideo(
    env,
    video_folder="video",
    name_prefix="CustomEnv_learned",
    episode_trigger=lambda episode_id: True
)

observation, _ = env.reset()
total_reward = 0
done = False

while not done:
    action, states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    
env.close()
print(f"\nTotal reward: {total_reward}")
print("done!")
# show video
command = ['mpv', 'video/CustomEnv_learned-episode-0.mp4']
result = subprocess.run(command, capture_output = True, text = True)

matplotlib.use("Agg")
x, y = ts2xy(load_results(log_dir), 'timesteps')  # Organising the logged results in to a clean format for plotting.
'''
print(x,y)

plt.pyplot(x, y)
plt.ylim([-1000, 300])
plt.xlabel('Timesteps')
plt.ylabel('Episode Rewards')
print("ok")
plt.savefig('foo.png', edgecolor = 'RED', transparent = False)
'''
