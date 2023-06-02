import random

import cv2
import numpy as np
import skvideo.io
import tqdm


class BackgroundMatting:
    """
    Produce a mask by masking the given color. This is a simple strategy
    but effective for many games.
    """

    def __init__(self, color):
        """
        Args:
            color: a (r, g, b) tuple or single value for grayscale
        """
        self._color = color

    def get_mask(self, img):
        return img == self._color


class ImageSource:
    """
    Source of natural images to be added to a simulated environment.
    """

    def get_image(self):
        """
        Returns:
            an RGB image of [h, w, 3] with a fixed shape.
        """
        pass

    def reset(self):
        """Called when an episode ends."""
        pass


class FixedColorSource(ImageSource):
    def __init__(self, shape, color):
        """
        Args:
            shape: [h, w]
            color: a 3-tuple
        """
        self.arr = np.zeros((shape[0], shape[1], 3))  # B006
        self.arr[:, :] = color

    def get_image(self):
        return self.arr


class RandomColorSource(ImageSource):
    def __init__(self, shape):
        """
        Args:
            shape: [h, w]
        """
        self.shape = shape
        self.arr = None
        self.reset()

    def reset(self):
        self._color = np.random.randint(0, 256, size=(3,))
        self.arr = np.zeros((self.shape[0], self.shape[1], 3))
        self.arr[:, :] = self._color

    def get_image(self):
        return self.arr


class NoiseSource(ImageSource):
    def __init__(self, shape, strength=255):
        """
        Args:
            shape: [h, w]
            strength (int): the strength of noise, in range [0, 255]
        """
        self.shape = shape
        self.strength = strength

    def get_image(self):
        return np.random.randn(self.shape[0], self.shape[1], 3) * self.strength


class RandomImageSource(ImageSource):
    def __init__(self, shape, filelist, total_frames=None, grayscale=False):
        """
        Args:
            shape: [h, w]
            filelist: a list of image files
        """
        self.grayscale = grayscale
        self.total_frames = total_frames
        self.shape = shape
        self.filelist = filelist
        self.build_arr()
        self.current_idx = 0
        self.reset()

    def build_arr(self):
        self.total_frames = self.total_frames if self.total_frames else len(self.filelist)
        self.arr = np.zeros((self.total_frames, self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
        for i in range(self.total_frames):
            # if i % len(self.filelist) == 0: random.shuffle(self.filelist)
            fname = self.filelist[i % len(self.filelist)]
            if self.grayscale:
                im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)[..., None]
            else:
                im = cv2.imread(fname, cv2.IMREAD_COLOR)
            self.arr[i] = cv2.resize(im, (self.shape[1], self.shape[0]))  ## THIS IS NOT A BUG! cv2 uses (width, height)

    def reset(self):
        self._loc = np.random.randint(0, self.total_frames)

    def get_image(self):
        return self.arr[self._loc]


class RandomVideoSource(ImageSource):
    def __init__(self, shape, filelist, total_frames=None, grayscale=False):
        """
        Args:
            shape: [h, w]
            filelist: a list of video files
        """
        self.grayscale = grayscale
        self.total_frames = total_frames
        self.shape = shape
        self.filelist = filelist
        self.build_arr()
        self.current_idx = 0
        self.reset()

    def build_arr(self):  # noqa C901
        if not self.total_frames:
            self.total_frames = 0
            self.arr = None
            random.shuffle(self.filelist)
            for fname in tqdm.tqdm(self.filelist, desc="Loading videos for natural", position=0):
                if self.grayscale:
                    frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
                else:
                    frames = skvideo.io.vread(fname)
                local_arr = np.zeros((frames.shape[0], self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
                for i in tqdm.tqdm(range(frames.shape[0]), desc="video frames", position=1):
                    local_arr[i] = cv2.resize(
                        frames[i], (self.shape[1], self.shape[0])
                    )  ## THIS IS NOT A BUG! cv2 uses (width, height)
                if self.arr is None:
                    self.arr = local_arr
                else:
                    self.arr = np.concatenate([self.arr, local_arr], 0)
                self.total_frames += local_arr.shape[0]
        else:
            self.arr = np.zeros((self.total_frames, self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
            total_frame_i = 0
            file_i = 0
            with tqdm.tqdm(total=self.total_frames, desc="Loading videos for natural") as pbar:
                while total_frame_i < self.total_frames:
                    if file_i % len(self.filelist) == 0:
                        random.shuffle(self.filelist)
                    file_i += 1
                    fname = self.filelist[file_i % len(self.filelist)]
                    if self.grayscale:
                        frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
                    else:
                        frames = skvideo.io.vread(fname)
                    for frame_i in range(frames.shape[0]):
                        if total_frame_i >= self.total_frames:
                            break
                        if self.grayscale:
                            self.arr[total_frame_i] = cv2.resize(frames[frame_i], (self.shape[1], self.shape[0]))[
                                ..., None
                            ]  ## THIS IS NOT A BUG! cv2 uses (width, height)
                        else:
                            self.arr[total_frame_i] = cv2.resize(frames[frame_i], (self.shape[1], self.shape[0]))
                        pbar.update(1)
                        total_frame_i += 1

    def reset(self):
        self._loc = np.random.randint(0, self.total_frames)

    def get_image(self):
        img = self.arr[self._loc % self.total_frames]
        self._loc += 1
        return img
