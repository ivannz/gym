import numpy as np
import os
import gym
from gym import error, spaces
from gym.utils import seeding

try:
    import atari_py
except ImportError as e:
    raise error.DependencyNotInstalled(
            "{}. (HINT: you can install Atari dependencies by running "
            "'pip install gym[atari]'.)".format(e))


def to_ram(ale):
    ram_size = ale.getRAMSize()
    ram = np.zeros((ram_size), dtype=np.uint8)
    ale.getRAM(ram)
    return ram


class AtariEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            game='pong',
            mode=None,
            difficulty=None,
            obs_type='ram',
            frameskip=(2, 5),
            repeat_action_probability=0.,
            full_action_space=False):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""
        if obs_type not in ("ram", "image"):
            raise error.Error(f"Unrecognized observation type: {obs_type}")

        self.game, self.mode = game, mode
        self.difficulty = difficulty

        self.obs_type = obs_type
        self.frameskip = frameskip

        self.repeat_action_probability = repeat_action_probability
        self.full_action_space = full_action_space

        self.seed()

    @property
    def ale(self):
        if not hasattr(self, "ale_"):
            self.ale_ = atari_py.ALEInterface()

        return self.ale_

    @property
    def action_space(self):
        return self.action_space_

    @property
    def observation_space(self):
        return self.observation_space_

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31

        # Tune (or disable) ALE's action repeat:
        # https://github.com/openai/gym/issues/349
        if not isinstance(self.repeat_action_probability, (float, int)):
            raise error.Error("Invalid repeat_action_probability: {!r}"
                              .format(self.repeat_action_probability))

        self.ale.setFloat("repeat_action_probability".encode('utf-8'),
                          self.repeat_action_probability)

        # Empirically, we need to seed before loading the ROM.
        self.ale.setInt(b'random_seed', seed2)

        # set the path to ROM on seed
        if not hasattr(self, "rom_path_"):
            rom_path_ = atari_py.get_game_path(self.game)
            if not os.path.exists(rom_path_):
                raise IOError(f"You asked for game `{self.game}` but"
                              f" path `{rom_path_}` does not exist.")
            self.rom_path_ = rom_path_
        self.ale.loadROM(self.rom_path_)

        # rom relaoded -- repopulate action and observaton space
        if self.full_action_space:
            action_set_ = self.ale.getLegalActionSet()
        else:
            action_set_ = self.ale.getMinimalActionSet()
        self.action_space_ = spaces.Discrete(len(action_set_))
        self.action_set_ = action_set_

        if self.obs_type == 'ram':
            space = spaces.Box(
                low=0, high=255, dtype=np.uint8, shape=(128,))

        elif self.obs_type == 'image':
            screen_width, screen_height = self.ale.getScreenDims()
            space = spaces.Box(
                low=0, high=255, dtype=np.uint8,
                shape=(screen_height, screen_width, 3))

        self.observation_space_ = space

        if self.mode is not None:
            modes = self.ale.getAvailableModes()
            if self.mode not in modes:
                raise RuntimeError(f"Invalid game mode `{self.mode}` for game"
                                   f" `{self.game}`.\nAvailable modes: {modes}")
            self.ale.setMode(self.mode)

        if self.difficulty is not None:
            difficulties = self.ale.getAvailableDifficulties()
            if self.difficulty not in difficulties:
                raise RuntimeError(f"Invalid game difficulty `{self.difficulty}`"
                                   f"for game {self.game}.\nAvailable difficulties: {difficulties}")
            self.ale.setDifficulty(self.difficulty)

        return [seed1, seed2]

    def step(self, a):
        ale, reward = self.ale, 0.0
        action = self.action_set_[a]

        if isinstance(self.frameskip, int):
            num_steps = self.frameskip

        else:
            num_steps = self.np_random.randint(*self.frameskip)

        for _ in range(num_steps):
            reward += ale.act(action)

        ob = self._get_obs()

        return ob, reward, ale.game_over(), {"ale.lives": ale.lives()}

    def _get_image(self):
        return self.ale.getScreenRGB2()

    def _get_ram(self):
        return to_ram(self.ale)

    @property
    def _n_actions(self):
        return len(self.action_set_)

    def _get_obs(self):
        if self.obs_type == 'ram':
            return self._get_ram()

        elif self.obs_type == 'image':
            img = self._get_image()

        return img

    # return: (states, observations)
    def reset(self):
        self.ale.reset_game()
        return self._get_obs()

    def render(self, mode='human'):
        img = self._get_image()
        if mode == 'rgb_array':
            return img

        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if not hasattr(self, "viewer_") or self.viewer_ is None:
                self.viewer_ = rendering.SimpleImageViewer()

            self.viewer_.imshow(img)
            return self.viewer_.isopen

    def close(self):
        if hasattr(self, "viewer_") and self.viewer_ is not None:
            self.viewer_.close()
            self.viewer_ = None

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self.action_set_]

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'UP':      ord('w'),
            'DOWN':    ord('s'),
            'LEFT':    ord('a'),
            'RIGHT':   ord('d'),
            'FIRE':    ord(' '),
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action

    def clone_state(self):
        """Clone emulator state w/o system state. Restoring this state will
        *not* give an identical environment. For complete cloning and restoring
        of the full state, see `{clone,restore}_full_state()`."""
        state_ref = self.ale.cloneState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_state(self, state):
        """Restore emulator state w/o system state."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreState(state_ref)
        self.ale.deleteState(state_ref)

    def clone_full_state(self):
        """Clone emulator state w/ system state including pseudorandomness.
        Restoring this state will give an identical environment."""
        state_ref = self.ale.cloneSystemState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_full_state(self, state):
        """Restore emulator state w/ system state including pseudorandomness."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreSystemState(state_ref)
        self.ale.deleteState(state_ref)


ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}
