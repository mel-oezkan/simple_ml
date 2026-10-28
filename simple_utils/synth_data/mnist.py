import torch
from torchvision import datasets, transforms
import numpy as np
import random

import matplotlib.pyplot as plt


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

class Digit:
    def __init__(self, pos_x: int, pos_y: int, size: int):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.size = size


class SampleImage:
    def __init__(
        self, 
        size: int, 
        min_digit_size: int,
        max_digit_size: int, 
        padding: int = 4,
    ):
        self.size = size
        self.digits: list[Digit] = []
        self.canvas = np.zeros((size, size))
        self.padding = padding

        self.digit_bound = (min_digit_size, max_digit_size)

    def reset(self):
        self.digits: list[Digit] = []

        self.canvas = np.zeros((self.size, self.size))

    def place_digit(self):
        canvas_mask = np.zeros((self.size, self.size))

        # generate a random shape given the parameters from init
        digit_size = (self.digit_bound[1] - self.digit_bound[0]) * random.random() + self.digit_bound[0]
        digit_size = int(digit_size)
        

        # drwa the canvas mask
        plt.title(f"Size of digit: {digit_size}")
        m1 = self.draw_canvas_border(canvas_mask, digit_size)
        m2 = self.draw_digit_border(canvas_mask, digit_size)
        canvas_mask = m1 + m2

        self._draw_digits(canvas_mask)
        plt.imshow(canvas_mask)
        plt.show()

        valid_pos = np.nonzero(canvas_mask == 0)
        if len(valid_pos[0]) == 0:
            self._plot_digits()
            raise ValueError("There is no place for a new digit")
        
        idx = np.random.randint(len(valid_pos[0]))
        row = valid_pos[0][idx]
        col = valid_pos[1][idx]
        self.digits.append(Digit(col, row, digit_size))

    def draw_canvas_border(self, mask: np.ndarray, digit_size: int) -> None:
        border_mask = np.zeros_like(mask)
        border_mask[:digit_size, :] = 1
        border_mask[digit_size:, :digit_size] = 1
        return border_mask

    def draw_digit_border(self, mask: np.ndarray, digit_size: int):
        digit_mask = np.zeros_like(mask)
        # for each digit updates 
        for digit in self.digits:
            print("digit")
            digit_mask[
                max(digit.pos_y-digit.size, 0):min(digit.pos_y+digit_size+self.padding, self.size), 
                max(digit.pos_x-digit.size, 0):min(digit.pos_x+digit_size+self.padding, self.size)
            ] += 2

        # if (digit_mask > 2).sum() > 0:
        #     plt.title("debug view")
        #     plt.imshow(digit_mask)
        #     plt.show()
            # raise ValueError("Code Error: There seems to be an overlap in the digits")
        return digit_mask

    def _draw_digits(self, canvas):
        for digit in self.digits:
            canvas[
                max(digit.pos_y-digit.size, 0):digit.pos_y, 
                max(digit.pos_x-digit.size, 0):digit.pos_x
            ] = 5
    def _plot_digits(self):
        canvas = np.zeros_like(self.canvas)
        self._draw_digits(canvas)
        plt.title("final digits")
        plt.imshow(canvas)
        plt.show()

    def draw(self):
        for i in range(5):
            self.place_digit()

        # final image
        canvas = np.zeros_like(self.canvas)
        self._draw_digits(canvas)
        plt.title("final digits")
        plt.imshow(canvas)
        plt.show()