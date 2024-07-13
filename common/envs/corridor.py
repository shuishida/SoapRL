from copy import deepcopy

import gymnasium
from gymnasium import spaces
import pygame
import numpy as np


class Object(object):
    pass


class CorridorEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6, 'video.frames_per_second': 6}
    color_palette = {
        "white": (255, 255, 255),
        "blue": (0, 0, 255),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
        "grey": (128, 128, 128),
        "black": (0, 0, 0),
    }

    color_to_obs = {
        "white": 0,
        "blue": 1,
        "red": 2,
        "yellow": 3,
        "grey": 4,
    }

    def __init__(self, width=10, render_mode="rgb_array"):
        self.width = width
        self.height = 3
        self.pix_square_size = 100
        self.render_mode = render_mode

        self.observation_space = spaces.Discrete(5)     # white, blue, red, yellow, grey
        # We have 2 actions, corresponding to "up", "down"
        self.action_space = spaces.Discrete(2)          # 0: up, 1: down
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self._is_target_up = None
        self._agent_pos = None    # 0, 1, 2, 3 (up), 4 (down)
        self._up_pos = self.width
        self._down_pos = self.width + 1
        self._target_pos = None
        self._trap_pos = None

    def _get_obs(self):
        if self._agent_pos == 0:
            color = "blue" if self._is_target_up else "red"
        elif self._agent_pos == self.width:
            color = "yellow" if self._is_target_up else "grey"
        elif self._agent_pos == self.width + 1:
            color = "grey" if self._is_target_up else "yellow"
        else:
            color = "white"
        return self.color_to_obs[color]

    def _get_info(self):
        return dict(
            agent_location=self._agent_location,
        )

    def reset(self, *,
        seed = None,
        options = None):
        super().reset(seed=seed, options=options)
        # Choose the agent's location uniformly at random
        self._agent_pos = 0
        self._is_target_up = np.random.randint(2)
        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_pos, self._trap_pos = (self._up_pos, self._down_pos) if self._is_target_up else (self._down_pos, self._up_pos)

        observation = self._get_obs()
        return observation, {}

    @property
    def _agent_location(self):
        return self.pos_to_loc(self._agent_pos)

    def pos_to_loc(self, pos):
        if pos == self._up_pos:
            return self.width - 1, 0
        elif pos == self._down_pos:
            return self.width - 1, 2
        return pos, 1

    def step(self, action):
        reward = 0

        # An episode is done iff the agent has reached the target
        terminated = self._agent_pos in (self._target_pos, self._trap_pos)
        if terminated:
            if self._agent_pos == self._target_pos:
                reward = 1
                # reward = 2 if self._is_target_up else 1
            else:
                reward = -1

        if self._agent_pos == self.width - 1:
            self._agent_pos = self.width + 1 if action else self.width
        elif self._agent_pos < self.width - 1:
            self._agent_pos += 1

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width * self.pix_square_size, self.height * self.pix_square_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.width * self.pix_square_size, self.height * self.pix_square_size))
        canvas.fill(self.color_palette["black"])

        for color, pos in zip(["blue" if self._is_target_up else "red", "yellow", "grey", *["white"] * (self.width - 1)],
                              [self.pos_to_loc(0), self.pos_to_loc(self._target_pos), self.pos_to_loc(self._trap_pos), *[self.pos_to_loc(i) for i in range(1, self.width)]]):
            # draw the colored state
            pygame.draw.rect(canvas, self.color_palette[color],
                             pygame.Rect(self.pix_square_size * np.array(pos), (self.pix_square_size, self.pix_square_size)))
        # draw the agent
        pygame.draw.circle(canvas, self.color_palette["green"],
                           (np.array(self._agent_location) + 0.5) * self.pix_square_size, self.pix_square_size / 3)

        # Finally, add some gridlines
        for x in range(self.height + 1):
            pygame.draw.line(canvas, 0, (0, self.pix_square_size * x), (self.width * self.pix_square_size, self.pix_square_size * x), width=3)
        for x in range(self.width + 1):
            pygame.draw.line(canvas, 0, (self.pix_square_size * x, 0), (self.pix_square_size * x, self.height * self.pix_square_size), width=3)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
