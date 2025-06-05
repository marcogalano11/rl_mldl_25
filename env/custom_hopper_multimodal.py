from copy import deepcopy
import numpy as np
import gym
from gym import utils, spaces
from .mujoco_env import MujocoEnv
from PIL import Image
import cv2
from collections import deque
import torch
from torchvision import transforms
import torch.nn as nn

class CustomCombinedExtractor(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__()

        image_shape = observation_space.spaces["image"].shape  # e.g. (8,64,64)
        state_shape = observation_space.spaces["state"].shape  # e.g. (state_dim,)

        self.cnn = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_img = torch.zeros(1, *image_shape)
            cnn_out_dim = self.cnn(dummy_img).shape[1]

        self.state_mlp = nn.Sequential(
            nn.Linear(state_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self._features_dim = cnn_out_dim + 64

    def forward(self, observations):
        img = observations["image"]
        state = observations["state"]
        cnn_features = self.cnn(img)
        state_features = self.state_mlp(state)
        combined = torch.cat((cnn_features, state_features), dim=1)
        return combined



# Image preprocessing (you can move this to a utils module if needed)

def v_crop(pil_img, crop_top=0, crop_bottom=0):
    width, height = pil_img.size
    return pil_img.crop((0, crop_top, width, height - crop_bottom))

def h_crop(pil_img, crop_left=0, crop_right=0):
    width, height = pil_img.size
    return pil_img.crop((crop_left, 0, width - crop_right, height))

def isolate_and_grayscale(pil_img: Image.Image) -> Image.Image:
    image_np = np.array(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([30, 255, 200])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    result = cv2.bitwise_and(image_np, image_np, mask=mask)
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    return Image.fromarray(gray, mode='L')

def preprocess_image(img):
    preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(lambda img: v_crop(img, crop_top=125, crop_bottom=55)), # oppure 135, 65
    transforms.Lambda(lambda img: h_crop(img, crop_left=90, crop_right=90)), # oppure 195, 175
    transforms.Lambda(lambda img: isolate_and_grayscale(img)),
    transforms.Resize((64,64)),  # oppure 150, 65
    transforms.ToTensor()
])
    return preprocess(img)


class CustomHopperMultimodal(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None, n_frames=8, evaluation=None):
        self.evaluation = evaluation
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])
        self.frames = deque([], maxlen=n_frames)
        self.n_frames = n_frames
        self.state_obs_shape = self._get_state_obs().shape

        # Setup image observation space
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0.0, high=1.0, shape=(n_frames, 64, 64), dtype=np.float32),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=self.state_obs_shape, dtype=np.float32)
        })

        if domain == 'source':
            self.sim.model.body_mass[1] *= 0.7

    def _render_image(self):
        frame = self.render(mode='rgb_array')
        return preprocess_image(frame)

    def _get_state_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def _get_obs(self):
        if self.evaluation == "states":
            img_stack = torch.zeros((self.n_frames, 64, 64), dtype=torch.float32)
        else:
            img = self._render_image()
            for _ in range(self.n_frames - len(self.frames)):
                self.frames.append(img.clone())
            self.frames.append(img)
            img_stack = torch.cat(list(self.frames), dim=0)

        if self.evaluation == "images":
            state = np.zeros(self.state_obs_shape)
        else:
            state = self._get_state_obs()

        return {
            "image": img_stack.numpy(),
            "state": state
        }

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        self.frames.clear()
        obs = self._get_obs()
        return obs

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt + alive_bonus - 1e-3 * np.square(a).sum()

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        obs = self._get_obs()

        return obs, reward, done, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def set_random_parameters(self):
        raise NotImplementedError()

    def get_parameters(self):
        return np.array(self.sim.model.body_mass[1:])

    def set_parameters(self, task):
        self.sim.model.body_mass[1:] = task

    def set_mujoco_state(self, state):
        mjstate = deepcopy(self.get_mujoco_state())
        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]
        self.set_sim_state(mjstate)

    def set_sim_state(self, mjstate):
        return self.sim.set_state(mjstate)

    def get_mujoco_state(self):
        return self.sim.get_state()


# Register environments
gym.envs.register(
    id="CustomHopperMultimodal-v0",
    entry_point="%s:CustomHopperMultimodal" % __name__,
    max_episode_steps=500,
)

gym.envs.register(
    id="CustomHopperMultimodal-source-v0",
    entry_point="%s:CustomHopperMultimodal" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "source"}
)

gym.envs.register(
    id="CustomHopperMultimodal-target-v0",
    entry_point="%s:CustomHopperMultimodal" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "target"}
)

