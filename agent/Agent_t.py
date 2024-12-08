import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import pygame
import time


GAME_BOARD_WIDTH = 10
GAME_BOARD_HEIGHT = 20

STATE_INPUT = "short"
T_SPIN_MARK = True
OUTER_MAX = 20
CPU_MAX = (
    99  # num of cpu used to collect samples = min(multiprocessing.cpu_count(), CPU_MAX)
)
OUT_START = 146
GAME_TYPE = "regular"


class Tetromino:
    MASSIVE_WEIGHT = 0.135  # the chance of a piece whose name ends with 'massive' compared with other pieces
    RNG_THRESHOLD = list()
    __POOL = list()

    @classmethod
    def create_pool(cls):
        if len(cls.__POOL) != 0:
            return
        # regular Tetris
        if GAME_TYPE == "regular":
            cls.__POOL.append(
                Tetromino([[1, 0], [2, 0], [0, 1], [1, 1]], 4, -1, 1.0, 0.0, "S", 0, 2)
            )
            cls.__POOL.append(
                Tetromino([[0, 0], [1, 0], [1, 1], [2, 1]], 4, -1, 1.0, 1.0, "Z", 0, 2)
            )
            cls.__POOL.append(
                Tetromino([[0, 1], [1, 1], [2, 1], [3, 1]], 3, -1, 1.5, 1.5, "I", 0, 2)
            )
            cls.__POOL.append(
                Tetromino(
                    [[1, 0], [0, 1], [1, 1], [2, 1], [1, 2]], 4, -1, 1.0, 1.0, "T", 0, 4
                )
            )
            cls.__POOL.append(
                Tetromino([[0, 0], [0, 1], [1, 1], [2, 1]], 4, -1, 0.5, 0.5, "J", 0, 4)
            )
            cls.__POOL.append(
                Tetromino([[2, 0], [0, 1], [1, 1], [2, 1]], 4, -1, 1.5, 0.5, "L", 0, 4)
            )
            cls.__POOL.append(
                Tetromino([[1, 0], [1, 1], [2, 1], [2, 0]], 3, -1, 1.5, 0.5, "O", 0, 1)
            )
        # mini Tetris
        elif GAME_TYPE == "mini":
            cls.__POOL.append(Tetromino([[1, 0], [1, 1]], 0, -1, 1.0, 0.0, "i", 0, 2))
            cls.__POOL.append(Tetromino([[0, 0], [1, 1]], 0, -1, 0.5, 0.5, "/", 0, 2))
            cls.__POOL.append(
                Tetromino([[0, 0], [1, 0], [1, 1]], 0, -1, 0.5, 0.5, "l", 0, 4)
            )

        # extra Tetris
        elif GAME_TYPE == "extra":
            # Tetromino([[x1,y1],[x2,y2]...], begin_x, begin_y, rotate_center_x, rotate_center_y, name, 0, rotate_max)
            # All [x,y] must be in the range [0,0] (left top) to [4,3] (right, bottom).
            # rotate_max is the possible states of the piece by rotation. e.g., an 'O' piece has only one state, an 'S' piece has two, and an 'L' has four.
            cls.__POOL.append(Tetromino([[1, 1]], 4, -1, 1.0, 1.0, "._extra", 0, 1))
            cls.__POOL.append(
                Tetromino([[0, 0], [1, 0]], 4, -1, 0.0, 0.0, "i.extra", 0, 2)
            )
            cls.__POOL.append(
                Tetromino([[0, 0], [1, 0], [2, 0]], 4, -1, 1.0, 0.0, "1.extra", 0, 2)
            )
            cls.__POOL.append(
                Tetromino(
                    [[0, 0], [0, 1], [1, 1], [2, 1], [2, 0]],
                    4,
                    -1,
                    1.0,
                    1.0,
                    "C.extra",
                    0,
                    4,
                )
            )
            cls.__POOL.append(
                Tetromino(
                    [[0, 0], [0, 1], [1, 1], [2, 1], [3, 1]],
                    4,
                    -1,
                    1.0,
                    1.0,
                    "J.extra",
                    0,
                    4,
                )
            )
            cls.__POOL.append(
                Tetromino(
                    [[3, 0], [0, 1], [1, 1], [2, 1], [3, 1]],
                    4,
                    -1,
                    2.0,
                    1.0,
                    "L.extra",
                    0,
                    4,
                )
            )
            cls.__POOL.append(
                Tetromino(
                    [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1]],
                    4,
                    -1,
                    0.5,
                    0.5,
                    "Z.extra",
                    0,
                    4,
                )
            )
            cls.__POOL.append(
                Tetromino(
                    [[0, 1], [1, 1], [1, 0], [2, 0], [3, 0]],
                    4,
                    -1,
                    1.5,
                    0.5,
                    "S.extra",
                    0,
                    4,
                )
            )
            cls.__POOL.append(
                Tetromino(
                    [[0, 0], [1, 0], [2, 0], [1, 1], [1, 2]],
                    3,
                    -1,
                    1.0,
                    1.0,
                    "T.extra",
                    0,
                    4,
                )
            )
            cls.__POOL.append(
                Tetromino(
                    [[1, 0], [1, 1], [1, 2], [2, 2], [3, 2], [3, 1], [3, 0], [2, 0]],
                    3,
                    -1,
                    2.0,
                    1.0,
                    "O.massive",
                    0,
                    1,
                )
            )
            cls.__POOL.append(
                Tetromino(
                    [[0, 0], [0, 1], [0, 2], [1, 1], [2, 1], [2, 0], [2, 2]],
                    3,
                    -1,
                    1,
                    1,
                    "H.massive",
                    0,
                    2,
                )
            )
            cls.__POOL.append(
                Tetromino(
                    [
                        [1, 0],
                        [1, 1],
                        [1, 2],
                        [2, 2],
                        [2, 1],
                        [3, 2],
                        [3, 1],
                        [3, 0],
                        [2, 0],
                    ],
                    3,
                    -1,
                    2.0,
                    1.0,
                    "Donut.massive",
                    0,
                    1,
                )
            )
            cls.__POOL.append(
                Tetromino(
                    [[1, 0], [0, 1], [1, 1], [2, 1], [1, 2], [3, 1]],
                    3,
                    -1,
                    1.0,
                    1.0,
                    "Sword.massive",
                    0,
                    4,
                )
            )
            # cls.__POOL.append(
            #     Tetromino([[0, 0], [1, 1], [2, 0], [2, 2], [0, 2]],
            #               3, -1, 1.0, 1.0, 'Cross.massive', 0, 1)
            # )

        for tet in cls.__POOL:
            if "massive" in tet.type_str:
                cls.RNG_THRESHOLD.append(cls.MASSIVE_WEIGHT)
            else:
                cls.RNG_THRESHOLD.append(1.0)

        rng_sum = sum(cls.RNG_THRESHOLD)
        for i in range(len(cls.RNG_THRESHOLD)):
            cls.RNG_THRESHOLD[i] /= rng_sum

        for i in range(1, len(cls.RNG_THRESHOLD)):
            cls.RNG_THRESHOLD[i] += cls.RNG_THRESHOLD[i - 1]

    @classmethod
    def pool_size(cls):
        return len(cls.__POOL)

    @classmethod
    def type_str_to_num(cls, type_str_arg):
        count = 1  # count start from 1 because 0 is reserved for empty
        for tetromino in cls.__POOL:
            if type_str_arg == tetromino.type_str:
                return count
            count += 1

        # print("type_str:" + type_str_arg + " not found")
        return None

    @classmethod
    def num_to_type_str(cls, num):
        # num start from 1, because 0 is reserved for empty
        return cls.__POOL[num - 1].type_str

    def __init__(self, tet, start_x, start_y, rot_x, rot_y, type_str_arg, rot, rot_max):
        self.tet = []
        for sq in tet:
            self.tet.append(list(sq))  # make sure this is copy
        self.center_x = start_x
        self.center_y = start_y
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.type_str = type_str_arg
        self.rot = rot
        self.rot_max = rot_max

    @classmethod
    def new_tetromino(cls, type_str_arg):
        for tet in cls.__POOL:
            if tet.type_str == type_str_arg:
                return tet.copy()
        # print("type_str is not found")
        return None

    @classmethod
    def new_tetromino_num(cls, type_num):
        return Tetromino.__POOL[type_num].copy()

    @classmethod
    def new_tetromino_fl(cls, rng_fl=None):
        if rng_fl is None:
            rng_fl = random.random()

        for i in range(len(cls.RNG_THRESHOLD)):
            if rng_fl < cls.RNG_THRESHOLD[i]:
                return cls.__POOL[i].copy()

        # print('ERROR: rng_fl must be between 0 and 1')
        return None

    @classmethod
    def random_type_str(cls, rng_fl=None):
        return cls.new_tetromino_fl(rng_fl).type_str

    def copy(self):
        return Tetromino(
            self.tet,
            self.center_x,
            self.center_y,
            self.rot_x,
            self.rot_y,
            self.type_str,
            self.rot,
            self.rot_max,
        )

    # turn +1 rotate counterclockwise
    def move(self, mov):
        (right, down, turn) = mov
        if (self.type_str == "S" or self.type_str == "Z") and turn != 0:
            # for S and Z pieces, it will rotate back if they have been rotated
            if self.rot == 1:
                turn = -1
            else:
                turn = 1

        if turn != 0:
            for sq in self.tet:
                a = sq[0]
                b = sq[1]
                x = self.rot_x
                y = self.rot_y

                sq[0] = round(turn * (b - y) + x)
                sq[1] = round(-turn * (a - x) + y)

        self.center_x += right
        self.center_y += down
        self.rot += turn
        self.rot = self.rot % self.rot_max

        return self

    def to_str(self):
        s = ""
        displaced = self.get_displaced()
        for sq in displaced:
            s += "[" + str(sq[0]) + ", " + str(sq[1]) + "] "
        s += "centerXY: " + str(self.center_x) + ", " + str(self.center_y) + " "
        s += "type: " + self.type_str
        return s

    def to_num(self):
        return self.type_str_to_num(self.type_str)

    def get_displaced(self):
        disp = list()
        for sq in self.tet:
            new_sq = list(sq)
            new_sq[0] = sq[0] + self.center_x
            new_sq[1] = sq[1] + self.center_y
            disp.append(new_sq)
        return disp

    def to_main_grid(self):
        disp = self.get_displaced()
        width = GAME_BOARD_WIDTH
        height = GAME_BOARD_HEIGHT
        row = list()
        grid = list()
        for i in range(width):
            row += [0]
        for j in range(height):
            grid += [list(row)]
        for sq in disp:
            grid[sq[1]][sq[0]] = self.to_num()

        return grid

    def to_above_grid(self):
        disp = self.get_displaced()
        width = GAME_BOARD_WIDTH
        above_grid = [0] * width
        for sq in disp:
            if sq[1] == -1:
                above_grid[sq[0]] = self.to_num()

        return above_grid

    def check_above_grid(self):
        disp = self.get_displaced()
        for sq in disp:
            if sq[1] < 0:
                return True

        return False

    @classmethod
    def to_small_window(cls, type_str):
        small = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        if type_str is None:
            return small  # if hold is None
        tetro = cls.new_tetromino(type_str)
        for sq in tetro.tet:
            a = sq[0]
            b = sq[1]
            small[b][a] = cls.type_str_to_num(type_str)

        return small


Tetromino.create_pool()
# size dependent
shape_main_grid = (1, GAME_BOARD_HEIGHT, GAME_BOARD_WIDTH, 1)
if STATE_INPUT == "short":
    shape_hold_next = (1, 1 * 2 + 1 + 6 * Tetromino.pool_size())
    split_hold_next = 1 * 2 + 1
else:
    shape_hold_next = (1, GAME_BOARD_WIDTH * 2 + 1 + 6 * Tetromino.pool_size())
    split_hold_next = GAME_BOARD_WIDTH * 2 + 1

print("shape_main_grid: ", shape_hold_next)

current_avg_score = 0
rand = random.Random()


def get_reward(add_scores, dones):
    reward = list()
    for i in range(len(add_scores)):
        add_score = add_scores[i].item()
        if dones[i]:
            add_score += -500
        reward.append(add_score)
    return np.array(reward).reshape([-1, 1])


def grid_to_str(grid):
    s = ""
    for row in grid:
        for sq in row:
            s += " " + str(sq)
        s += "\n"
    return s


def copy_2d(grid):
    copied = list()
    for row in grid:
        copied.append(list(row))
    return copied


def text_list_flatten(text_list):
    text = ""
    for s in text_list:
        if not isinstance(s, str):
            print("")
        text += s + ", "
    return text


INITIAL_EX_WIGHT = 0.0
SPIN_SHIFT_FOR_NON_T = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (1, 1, 0),
    (-1, 1, 0),
    (1, -1, 0),
    (-1, -1, 0),
    (0, 2, 0),
    (1, 2, 0),
    (-1, 2, 0),
]

# if you don't want to see some spurious t-spin moves
SPIN_SHIFT_FOR_T = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (1, 1, 0),
    (-1, 1, 0),
    (1, -1, 0),
    (-1, -1, 0),
    (0, 2, 0),
    (1, 2, 0),
    (-1, 2, 0),
]  # disable triple t-spin

# if you allow t spins in various funky ways
# SPIN_SHIFT_FOR_T = [(1, 0, 0), (-1, 0, 0),
#                     (0, 1, 0), (1, 1, 0), (-1, 1, 0),
#                     (0, 2, 0), (1, 2, 0), (-1, 2, 0),
#                     (0, -1, 0), (1, -1, 0), (-1, -1, 0)]  # enable triple t-spin

ACTIONS = ["left", "right", "down", "turn left", "turn right", "drop"]

IDLE_MAX = 9999


class Gamestate:
    def __init__(self, grid=None, seed=None, rd=None, height=0):
        if seed is None:
            self.seed = random.randint(0, round(9e9))
        else:
            self.seed = seed
        self.rand_count = 0

        if rd is None:
            self.rd = random.Random(seed)
        else:
            self.rd = rd

        if grid is None:
            self.grid = self.initial_grid(height)
        else:
            self.grid = list()
            for row in grid:
                self.grid.append(list(row))

        self.tetromino = Tetromino.new_tetromino_fl(self.get_random().random())
        self.hold_type = None
        self.next = list()
        for i in range(5):
            self.next.append(Tetromino.random_type_str(self.get_random().random()))
        self.next_next = Tetromino.random_type_str(self.get_random().random())
        self.n_lines = [0, 0, 0, 0]
        self.t_spins = [0, 0, 0, 0]
        self.game_status = "playing"
        self.is_hold_last = False
        self.ex_weight = INITIAL_EX_WIGHT
        self.score = 0
        self.lines = 0
        self.pieces = 0
        self.idle = 0

    def start(self):
        self.tetromino = Tetromino.new_tetromino(self.next[0])
        self.next[:-1] = self.next[1:]
        self.next[-1] = self.next_next
        self.next_next = Tetromino.random_type_str(self.get_random().random())

    def initial_grid(self, height=0):
        grid = list()
        for _ in range(GAME_BOARD_HEIGHT):
            grid.append([0] * GAME_BOARD_WIDTH)

        if height == 0:
            return grid

        # if height = 15, range(6, 20), saving the first row for random generation
        for i in range(GAME_BOARD_HEIGHT - height, GAME_BOARD_HEIGHT):
            for j in range(GAME_BOARD_WIDTH):
                grid[i][j] = self.get_random().randint(0, Tetromino.pool_size())
            grid[i][self.get_random().randint(0, GAME_BOARD_WIDTH - 1)] = 0

        # for j in range(GAME_BOARD_WIDTH):
        #     grid[GAME_BOARD_HEIGHT - height][j] = self.get_random().randint(0, 1)

        return grid

    def get_random_grid(self):
        grid = list()
        for i in range(GAME_BOARD_HEIGHT):
            row = list()
            for j in range(GAME_BOARD_WIDTH):
                row.append(0)
            grid.append(row)

        height = self.get_random().randint(0, min(15, GAME_BOARD_HEIGHT - 2))

        # if height = 15, range(6, 20), saving the first row for random generation
        for i in range(GAME_BOARD_HEIGHT - height, GAME_BOARD_HEIGHT):
            for j in range(GAME_BOARD_WIDTH):
                grid[i][j] = self.get_random().randint(0, len(Tetromino.pool_size()))
            grid[i][self.get_random().randint(0, GAME_BOARD_WIDTH - 1)] = 0

        all_brick = True
        for j in range(GAME_BOARD_WIDTH):
            grid[GAME_BOARD_HEIGHT - height - 1][j] = self.get_random().randint(0, 1)
            if grid[GAME_BOARD_HEIGHT - height - 1][j] == 0:
                all_brick = False
        if all_brick:
            grid[GAME_BOARD_HEIGHT - height - 1][
                self.get_random().randint(0, GAME_BOARD_WIDTH - 1)
            ] = 0

        return grid

    @staticmethod
    def random_gamestate(seed=None):
        if seed is None:
            large_int = 999999999
            seed = random.randint(0, large_int)
        gamestate = Gamestate(seed=seed)
        gamestate.grid = gamestate.get_random_grid()
        return gamestate

    def copy(self):
        state_copy = Gamestate(self.grid, rd=self.rd)

        state_copy.seed = self.seed
        state_copy.tetromino = self.tetromino.copy()

        state_copy.hold_type = self.hold_type
        state_copy.next = list()
        for s in self.next:
            state_copy.next.append(s)
        state_copy.next_next = self.next_next
        state_copy.n_lines = list(self.n_lines)
        state_copy.t_spins = list(self.t_spins)
        state_copy.game_status = self.game_status
        state_copy.is_hold_last = self.is_hold_last
        state_copy.ex_weight = self.ex_weight
        state_copy.score = self.score
        state_copy.lines = self.lines
        state_copy.pieces = self.pieces
        state_copy.rand_count = self.rand_count
        state_copy.idle = self.idle

        return state_copy

    def define(self, tetromino, next):
        state_copy = Gamestate(self.grid, rd=self.rd)

        state_copy.seed = self.seed
        state_copy.tetromino = tetromino
        state_copy.hold_type = self.hold_type
        state_copy.next = list()
        for s in next:
            state_copy.next.append(s)
        state_copy.next_next = self.next_next
        state_copy.n_lines = list(self.n_lines)
        state_copy.t_spins = list(self.t_spins)
        state_copy.game_status = self.game_status
        state_copy.is_hold_last = self.is_hold_last
        state_copy.ex_weight = self.ex_weight
        state_copy.score = self.score
        state_copy.lines = self.lines
        state_copy.pieces = self.pieces
        state_copy.rand_count = self.rand_count
        state_copy.idle = self.idle

        return state_copy

    def define2(self, tetromino, next, boolean_array):
        # for i in range(GAME_BOARD_HEIGHT):
        #     for j in range(GAME_BOARD_WIDTH):
        #         self.grid[i][j] = self.grid[i][j]
        for i in range(GAME_BOARD_HEIGHT):
            for j in range(GAME_BOARD_WIDTH):
                self.grid[i][j] = 1 if boolean_array[i][j] else 0

        self.seed = self.seed
        self.tetromino = tetromino
        self.hold_type = self.hold_type
        for i in range(len(self.next)):
            self.next[i] = next[i]
        self.next_next = self.next_next
        self.n_lines = list(self.n_lines)
        self.t_spins = list(self.t_spins)
        self.game_status = self.game_status
        self.is_hold_last = self.is_hold_last
        self.ex_weight = self.ex_weight
        self.score = self.score
        self.lines = self.lines
        self.pieces = self.pieces
        self.rand_count = self.rand_count
        self.idle = self.idle

    def copy_value(self, state_original):
        for i in range(GAME_BOARD_HEIGHT):
            for j in range(GAME_BOARD_WIDTH):
                self.grid[i][j] = state_original.grid[i][j]

        self.seed = state_original.seed
        self.tetromino = state_original.tetromino.copy()
        self.hold_type = state_original.hold_type
        for i in range(len(self.next)):
            self.next[i] = state_original.next[i]
        self.next_next = state_original.next_next
        self.n_lines = list(state_original.n_lines)
        self.t_spins = list(state_original.t_spins)
        self.game_status = state_original.game_status
        self.is_hold_last = state_original.is_hold_last
        self.ex_weight = state_original.ex_weight
        self.score = state_original.score
        self.lines = state_original.lines
        self.pieces = state_original.pieces
        self.rand_count = state_original.rand_count
        self.idle = state_original.idle

    def put_tet_to_grid(self, tetro=None):
        grid_copy = copy_2d(self.grid)
        if tetro is None:
            tetro = self.tetromino

        disp = tetro.get_displaced()
        for sq in disp:
            x = sq[0]
            y = sq[1]
            if x < 0 or x > GAME_BOARD_WIDTH or y > GAME_BOARD_HEIGHT:
                continue
            if y < 0:
                continue
            grid_copy[y][x] = tetro.to_num()
        return grid_copy

    def check_collision(self, tetro=None):
        if tetro is None:
            tetro = self.tetromino

        disp = tetro.get_displaced()
        for sq in disp:
            x = sq[0]
            y = sq[1]
            if x < 0 or x >= GAME_BOARD_WIDTH or y >= GAME_BOARD_HEIGHT:
                return True
            if y < 0:
                continue
            if self.grid[y][x] != 0:
                return True
        return False

    def check_t_spin(self):
        if self.tetromino.type_str != "T":
            return False
        check_mov = [(0, -1, 0), (1, 0, 0), (-1, 0, 0)]
        for mov in check_mov:
            tetro = self.tetromino.copy().move(mov)
            if not self.check_collision(tetro):
                return False

        return True

    def check_completed_lines(self, above_grid=None):
        completed_lines = 0
        row_num = 0
        for row in self.grid:
            complete = True
            for sq in row:
                if sq == 0:
                    complete = False
                    break
            if complete:
                self.remove_line(row_num, above_grid=above_grid)
                completed_lines += 1
            row_num += 1

        return completed_lines

    def remove_line(self, row_num, above_grid=None):
        self.grid[1 : row_num + 1] = self.grid[:row_num]
        if above_grid is None:
            new_row = [0] * GAME_BOARD_WIDTH
        else:
            new_row = above_grid[:]
        self.grid[0] = new_row

    def check_clear_board(self):
        for i in reversed(range(GAME_BOARD_HEIGHT)):
            for block in self.grid[i]:
                if block != 0:
                    return False
        return True

    def update_score(self, lines, is_t_spin, is_clear):
        if is_t_spin:
            if lines == 1:
                score_lines = 2
            elif lines == 2:
                score_lines = 4
            elif lines == 3:
                score_lines = 5
            else:
                score_lines = 0
            self.t_spins[lines] += 1
        else:
            score_lines = lines

        add_score = (score_lines + 1) * score_lines / 2 * 10
        # add_score = lines * 10
        # if is_clear:
        #     add_score += 60

        if T_SPIN_MARK and is_t_spin:
            self.score = int(self.score) + add_score + 0.1
            add_score += 0.1
        else:
            self.score += add_score
        self.lines += lines

        if lines != 0:
            self.n_lines[lines - 1] += 1
        self.pieces += 1
        return add_score

    def get_score_text(self):
        s = "score:  " + str(int(self.score)) + "\n"
        s += "lines:  " + str(int(self.lines)) + "\n"
        s += "pieces: " + str(self.pieces) + "\n"
        one_line = ""
        for num in self.n_lines:
            one_line += f"{num} "
        s += "n_lines: " + one_line + "\n"
        one_line = ""
        for num in self.t_spins:
            one_line += f"{num} "
        s += "t_spins: " + one_line + "\n"
        return s

    def get_info_text(self):
        # s = "unfinished info text \n"
        s = "seed: " + str(self.seed)
        return s

    def soft_drop(self):
        tetro = self.tetromino
        down = 0
        while not self.check_collision(tetro.move((0, 1, 0))):
            down += 1
        tetro.move((0, -1, 0))
        return down

    def hard_drop(self):
        self.soft_drop()
        return self.process_down_collision()

    def process_down_collision(self):
        is_t_spin = self.check_t_spin()
        is_above_grid = self.tetromino.check_above_grid()
        above_grid = self.tetromino.to_above_grid()
        self.freeze()
        completed_lines = self.check_completed_lines(above_grid=above_grid)
        is_clear = self.check_clear_board()
        add_score = self.update_score(completed_lines, is_t_spin, is_clear)
        if self.check_collision() or (is_above_grid and completed_lines == 0):
            self.game_status = "gameover"
            done = True
        else:
            done = False
        return add_score, done

    def process_turn(self):  # return true if turn is successful
        if self.check_collision():
            success = False
            shifted = None
            if self.tetromino.type_str.lower() == "t":
                spin_moves = SPIN_SHIFT_FOR_T
            else:
                spin_moves = SPIN_SHIFT_FOR_NON_T
            for mov in spin_moves:
                shifted = self.tetromino.copy().move(mov)
                if not self.check_collision(shifted):
                    success = True
                    break
            if success:
                self.tetromino = shifted
            return success
        else:
            return True

    def process_left_right(self):
        if self.check_collision():
            return False
        else:
            return True

    @classmethod
    def cls_put_tet_to_grid(cls, grid, tetro):
        grid_copy = copy_2d(grid)
        disp = tetro.get_displaced()
        collide = False
        for sq in disp:
            x = sq[0]
            y = sq[1]
            if grid_copy[y][x] != 0:
                collide = True
            grid_copy[y][x] = tetro.to_num()
        return grid_copy, collide

    def hold(self):
        if self.is_hold_last:
            return False

        new_hold_type = self.tetromino.type_str
        if self.hold_type is None:
            self.tetromino = Tetromino.new_tetromino(self.next[0])
            self.next[:-1] = self.next[1:]
            self.next[-1] = self.next_next
            self.next_next = Tetromino.random_type_str(self.get_random().random())
        else:
            self.tetromino = Tetromino.new_tetromino(self.hold_type)

        self.hold_type = new_hold_type
        self.is_hold_last = True
        self.pieces += 1

        if self.check_collision():
            self.game_status = "gameover"
        return True

    def freeze(self):
        self.grid = self.put_tet_to_grid()
        self.tetromino = Tetromino.new_tetromino(self.next[0])
        self.next[:-1] = self.next[1:]
        self.next[-1] = self.next_next
        self.next_next = Tetromino.random_type_str(self.get_random().random())
        self.is_hold_last = False

    def get_random(self):
        # self.rand_count += 1
        # return random.Random(self.rand_count * self.seed)
        return self.rd

    def check_up_collision(self):
        self.tetromino.move((0, -1, 0))
        collision = self.check_collision()
        self.tetromino.move((0, 1, 0))
        return collision

    def get_turn_expansion(self):
        state_turn = self.copy()
        states_turn = [state_turn.copy()]
        moves_turn = [[]]
        for i in range(1, self.tetromino.rot_max):
            state_turn.tetromino.move((0, 0, 1))
            success = state_turn.process_turn()
            if not success:
                # usually not a concern until the end when
                # there is a slight chance that you cannot turn
                break
            state = state_turn.copy()
            states_turn += [state]
            moves_turn += [["turn left"] * i]

        return states_turn, moves_turn

    def get_left_right_expansion(self, moves_turn):
        # move 0
        states_lr = [self.copy()]
        moves_lr = [moves_turn]

        # move left
        state_copy = self.copy()
        left = 0
        while True:
            state_copy.tetromino.move((-1, 0, 0))
            if state_copy.check_collision():
                break
            else:
                left += 1
                moves = moves_turn + ["left"] * left
                states_lr += [state_copy.copy()]
                moves_lr += [moves]

        # move right
        state_copy = self.copy()
        right = 0
        while True:
            state_copy.tetromino.move((1, 0, 0))
            if state_copy.check_collision():
                break
            else:
                right += 1
                moves = moves_turn + ["right"] * right
                states_lr += [state_copy.copy()]
                moves_lr += [moves]

        # # soft drop
        # for s, m in list(zip(states_lr, moves_lr)):
        #     s.soft_drop()
        #     m += ["soft"]

        return states_lr, moves_lr

    def get_tuck_spin_expansion(self, moves_lr):
        # move 0
        states_ts = [self.copy()]
        moves_ts = [moves_lr]

        # move left
        state_copy = self.copy()
        left = 0
        while True:
            state_copy.tetromino.move((-1, 0, 0))
            if state_copy.check_collision():
                break
            elif not state_copy.check_up_collision():
                break
            else:
                left += 1
                moves = moves_lr + ["left"] * left
                states_ts += [state_copy.copy()]
                moves_ts += [moves]

        # move right
        state_copy = self.copy()
        right = 0
        while True:
            state_copy.tetromino.move((1, 0, 0))
            if state_copy.check_collision():
                break
            elif not state_copy.check_up_collision():
                break
            else:
                right += 1
                moves = moves_lr + ["right"] * right
                states_ts += [state_copy.copy()]
                moves_ts += [moves]

        if self.tetromino.rot_max == 1:
            return states_ts, moves_ts

        more_states_ts = list()
        more_moves_ts = list()
        for i in range(len(states_ts)):
            state_copy = states_ts[i].copy()
            state_copy.tetromino.move((0, 0, 1))
            if state_copy.process_turn() and state_copy.check_up_collision():
                more_states_ts += [state_copy]
                more_moves_ts.append(moves_ts[i] + ["turn left"] * 1)

                if self.tetromino.rot_max > 2:
                    state_copy = state_copy.copy()
                    state_copy.tetromino.move((0, 0, 1))
                    if state_copy.process_turn() and state_copy.check_up_collision():
                        more_states_ts += [state_copy]
                        more_moves_ts.append(moves_ts[i] + ["turn left"] * 2)

            if self.tetromino.rot_max == 2:
                continue

            state_copy = states_ts[i].copy()
            state_copy.tetromino.move((0, 0, -1))
            if state_copy.process_turn() and state_copy.check_up_collision():
                more_states_ts += [state_copy]
                more_moves_ts.append(moves_ts[i] + ["turn right"] * 1)

                if self.tetromino.rot_max > 2:
                    state_copy = state_copy.copy()
                    state_copy.tetromino.move((0, 0, -1))
                    if state_copy.process_turn() and state_copy.check_up_collision():
                        more_states_ts += [state_copy]
                        more_moves_ts.append(moves_ts[i] + ["turn right"] * 2)

        return states_ts + more_states_ts, moves_ts + more_moves_ts

    def get_height_sum(self):
        heights = self.get_heights()
        return sum(heights)

    def get_hole_depth(self):
        depth = [0] * GAME_BOARD_WIDTH
        highest_brick = 0
        for j in range(GAME_BOARD_WIDTH):
            has_found_brick = False
            for i in range(GAME_BOARD_HEIGHT):
                if not has_found_brick:
                    if self.grid[i][j] > 0:
                        has_found_brick = True
                        highest_brick = i
                elif self.grid[i][j] == 0:
                    depth[j] = i - highest_brick
                    break
        return depth

    def get_heights(self):
        heights = [0] * GAME_BOARD_WIDTH
        for j in range(GAME_BOARD_WIDTH):
            for i in range(GAME_BOARD_HEIGHT):
                if self.grid[i][j] > 0:
                    heights[j] = GAME_BOARD_HEIGHT - i
                    break
        return heights


class Game:
    def __init__(self, gui=None, seed=None, height=0):
        self.gui = gui
        self.seed = seed
        self.current_state = Gamestate(seed=seed, height=height)
        self.all_possible_states = []
        self.height = height

    def act(self, action):
        if self.current_state.game_status == "gameover":
            return self.get_state_ac(), 0, True, False

        success = False
        done = False
        add_score = 0
        action = action.lower()

        copy_state = self.current_state.copy()

        if action == "left":
            copy_state.tetromino.move((-1, 0, 0))
            success = copy_state.process_left_right()
        elif action == "right":
            copy_state.tetromino.move((1, 0, 0))
            success = copy_state.process_left_right()
        elif action == "turn left":
            copy_state.tetromino.move((0, 0, 1))
            success = copy_state.process_turn()
        elif action == "turn right":
            copy_state.tetromino.move((0, 0, -1))
            success = copy_state.process_turn()
        elif action == "down":
            copy_state.tetromino.move((0, 1, 0))
            if copy_state.check_collision():
                copy_state.tetromino.move((0, -1, 0))
                add_score, done = copy_state.process_down_collision()
            success = True  # move down will take effect no matter what
        elif action == "drop":
            add_score, done = copy_state.hard_drop()
            success = True
        elif action == "hold":
            success = copy_state.hold()
        else:
            print(str(action) + " action is not found. Please check.")

        if success:
            self.current_state = copy_state

        if action == "down" or action == "drop" or action == "hold":
            self.current_state.idle = 0
        elif self.current_state.idle >= IDLE_MAX:
            self.current_state.idle = 0
            self.current_state.tetromino.move((0, 1, 0))
            if self.current_state.check_collision():
                self.current_state.tetromino.move((0, -1, 0))
                add_score, done = self.current_state.process_down_collision()
            success = True  # move down will take effect no matter what
        else:
            self.current_state.idle += 1

        return self.get_state_ac(), add_score, done, success

    def render(self):
        if self.gui is not None:
            self.update_gui()
            self.gui.redraw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass

    def restart(self, height=None):
        if height is None:
            self.current_state = Gamestate(seed=self.seed, height=self.height)
        else:
            self.current_state = Gamestate(seed=self.seed, height=height)
        self.current_state.start()

    def update_gui(self, gamestate=None, is_display_current=True):
        if self.gui is None:
            return
        if gamestate is None:
            gamestate = self.current_state

        if is_display_current:
            above_grid = gamestate.tetromino.to_above_grid()
            main_grid = copy_2d(gamestate.put_tet_to_grid())
        else:
            above_grid = [0] * GAME_BOARD_WIDTH
            main_grid = copy_2d(gamestate.grid)

        hold_grid = Tetromino.to_small_window(gamestate.hold_type)
        next_grids = list()
        for n in gamestate.next:
            next_grids.append(Tetromino.to_small_window(n))
        self.gui.update_grids_color((main_grid, hold_grid, next_grids), above_grid)

        self.gui.set_score_text(gamestate.get_score_text())
        self.gui.set_info_text(gamestate.get_info_text())

    def run(self):
        is_run = True
        while is_run:
            if self.gui is not None:
                self.update_gui()
                self.gui.redraw()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_run = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.act("left")
                    if event.key == pygame.K_d:
                        self.act("right")
                    if event.key == pygame.K_s:
                        self.act("down")
                    if event.key == pygame.K_j:
                        self.act("turn left")
                    if event.key == pygame.K_k:
                        self.act("turn right")
                    if event.key == pygame.K_SPACE:
                        self.act("drop")
                    if event.key == pygame.K_q:
                        self.act("hold")
                    if event.key == pygame.K_r:
                        self.current_state = Gamestate(seed=self.seed)
                        self.current_state.start()
                    if event.key == pygame.K_1:
                        self.display_all_possible_state()
                    if event.key == pygame.K_i:
                        self.info_print()
                    if event.key == pygame.K_2:
                        # changing current tetromino
                        pool_size = Tetromino.pool_size()
                        num = self.current_state.tetromino.to_num()
                        # remember the return num has already been increased by 1, leaving room for 0
                        if num >= pool_size:
                            num = num - pool_size
                        self.current_state.tetromino = Tetromino.new_tetromino_num(num)

    def info_print(self):
        # print(self.current_state.score)

        return None

    def reset(self, height=None):
        if height is None:
            self.restart()
        else:
            self.restart(height=height)
        return self.get_state_ac()

    # ac means actor critic
    def get_state_ac(self):
        return [self.get_main_grid_np_ac(), self.get_hold_next_np_ac()]

    def get_main_grid_np_ac(self):
        tet_to_grid = self.current_state.tetromino.to_main_grid()
        buffer = []
        for i in range(len(self.current_state.grid)):
            for j in range(len(self.current_state.grid[i])):
                buffer.append([self.current_state.grid[i][j]])
                buffer.append([tet_to_grid[i][j]])

        buffer = np.reshape(
            np.array(buffer), [1, GAME_BOARD_HEIGHT, GAME_BOARD_WIDTH, 2]
        )
        buffer = (buffer > 0) * 2 - 1
        return buffer

    def get_hold_next_np_ac(self):
        buffer = Tetromino.to_small_window(self.current_state.hold_type)
        for tetro_type in self.current_state.next:
            for row in Tetromino.to_small_window(tetro_type):
                buffer.append(row)
        buffer = np.reshape(np.array(buffer), [1, 18, 4, 1])
        buffer = (buffer > 0) * 2 - 1
        return buffer

    def step(self, action=None, chosen=None):
        if action is not None:
            return self.act(action)
        elif chosen is not None:
            self.current_state = self.all_possible_states[chosen]
            return self.get_state_dqn_conv2d(self.current_state)
        else:
            # print('something is wrong with the args in step()')
            return None

    def is_done(self):
        if self.current_state.game_status == "gameover":
            return True
        else:
            return False

    @staticmethod
    def get_state_dqn_conv2d(gamestate):
        return Game.get_main_grid_np_dqn(gamestate), Game.get_hold_next_np_dqn(
            gamestate
        )

    @staticmethod
    def get_main_grid_np_dqn(gamestate):
        buffer = []
        for i in range(len(gamestate.grid)):
            for j in range(len(gamestate.grid[i])):
                buffer.append([gamestate.grid[i][j]])

        buffer = np.reshape(
            np.array(buffer), [1, GAME_BOARD_HEIGHT, GAME_BOARD_WIDTH, 1]
        )
        buffer = buffer > 0
        return buffer

    @staticmethod
    def get_hole_np_dqn(gamestate):
        buffer = gamestate.get_hole_depth() + gamestate.get_heights()
        return np.reshape(np.array(buffer), [1, GAME_BOARD_WIDTH * 2, 1])

    @staticmethod
    def get_hold_next_np_dqn(gamestate):
        # part1: heights + hold_depth. len -> 20
        if STATE_INPUT == "short":
            buffer1 = [sum(gamestate.get_heights())] + [sum(gamestate.get_hole_depth())]
        else:
            buffer1 = gamestate.get_heights() + gamestate.get_hole_depth()

        # part2: current 1; hold 1; next 4
        # next will always be the last for convenience, because of the change in the last one
        # hold has one more position to record if last step is 'hold'
        hold_num = 1
        current_num = 1
        next_num = 4
        pool_size = Tetromino.pool_size()
        buffer2 = [0] * (pool_size * (hold_num + current_num + next_num) + hold_num)

        if hold_num == 1:
            if gamestate.is_hold_last:
                buffer2[0] = 1
            if gamestate.hold_type is not None:
                tetro_type_num = Tetromino.type_str_to_num(gamestate.hold_type) - 1
                buffer2[tetro_type_num + hold_num] = 1

        tetro_type_num = Tetromino.type_str_to_num(gamestate.tetromino.type_str) - 1
        buffer2[hold_num + hold_num * pool_size + tetro_type_num] = 1

        for i in range(next_num):
            tetro_type_num = Tetromino.type_str_to_num(gamestate.next[i]) - 1
            buffer2[
                hold_num + (i + hold_num + current_num) * pool_size + tetro_type_num
            ] = 1

        return np.reshape(np.array(buffer1 + buffer2, dtype="int8"), [1, -1])

    def get_all_possible_gamestates(self, gamestate=None):
        if gamestate is None:
            gamestate_original = self.current_state.copy()
        else:
            gamestate_original = gamestate.copy()

        states_lr_all = list()
        moves_lr_all = list()
        ss, ms = gamestate_original.get_turn_expansion()
        for s, m in list(zip(ss, ms)):
            s_lr, m_lr = s.get_left_right_expansion(m)
            states_lr_all += s_lr
            moves_lr_all += m_lr

        gamestates = list()
        moves = list()
        for s, m in list(zip(states_lr_all, moves_lr_all)):
            s_ts, m_ts = s.get_tuck_spin_expansion(m)
            gamestates += s_ts
            moves += m_ts

        add_scores = list()
        dones = list()

        # press down
        for s, m in list(zip(gamestates, moves)):
            add_score, done = s.hard_drop()
            m += ["drop"]
            add_scores += [add_score]
            dones += [done]

        is_include_hold = False
        is_new_hold = False
        # hold
        if (
            gamestate_original.hold_type != gamestate_original.tetromino.type_str
            and not gamestate_original.is_hold_last
        ):
            is_include_hold = True
            if gamestate_original.hold_type is None:
                is_new_hold = True
            gamestate_original.hold()
            gamestates += [gamestate_original.copy()]
            moves += [["hold"]]
            add_scores += [0]
            if gamestate_original.game_status == "gameover":
                dones += [True]
            else:
                dones += [False]

        # gamestate is for GameMini; state is for neural network
        self.all_possible_states = gamestates

        return gamestates, moves, add_scores, dones, is_include_hold, is_new_hold

    def get_all_possible_states_conv2d(self):
        gamestates, moves, add_scores, dones, is_include_hold, is_new_hold = (
            self.get_all_possible_gamestates(self.current_state)
        )

        mains = list()
        hold_next = list()
        for gamestate in gamestates:
            in1, in2 = Game.get_state_dqn_conv2d(gamestate)
            mains.append(in1)
            hold_next.append(in2)

        return (
            [np.concatenate(mains), np.concatenate(hold_next)],
            np.array([add_scores]).reshape([len(add_scores), 1]),
            dones,
            is_include_hold,
            is_new_hold,
            moves,
        )

    def get_all_possible_states_conv2d_hung(self, state_hung=None):
        self.current_state = state_hung
        gamestates, moves, add_scores, dones, is_include_hold, is_new_hold = (
            self.get_all_possible_gamestates(state_hung)
        )

        mains = list()
        hold_next = list()
        for gamestate in gamestates:
            in1, in2 = Game.get_state_dqn_conv2d(gamestate)
            mains.append(in1)
            hold_next.append(in2)

        return (
            [np.concatenate(mains), np.concatenate(hold_next)],
            np.array([add_scores]).reshape([len(add_scores), 1]),
            dones,
            is_include_hold,
            is_new_hold,
            moves,
        )

    def display_all_possible_state(self):
        if self.gui is None:
            return

        states, moves, _, _, _, _ = self.get_all_possible_gamestates()
        for s, m in zip(states, moves):
            self.update_gui(s, is_display_current=False)
            self.gui.set_info_text(text_list_flatten(m))
            self.gui.redraw()
            time.sleep(0.1)


# dir_path = os.path.dirname(os.path.realpath(__file__))
# weight_file_path = os.path.join(dir_path, 'outer_14')
# model  = keras.models.load_model(weight_file_path)
# model.summary()


# env_hung = Game(height=0)


class Conv2DModel(nn.Module):
    def __init__(self):
        super(Conv2DModel, self).__init__()

        # Main Grid Convolutional Branch (a and b)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=6, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(15, 5), stride=(1, 1))
        self.avgpool1 = nn.AvgPool2d(kernel_size=(15, 5))

        self.conv2 = nn.Conv2d(1, 256, kernel_size=4, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(17, 7), stride=(1, 1))
        self.avgpool2 = nn.AvgPool2d(kernel_size=(17, 7))

        # Hold Next input
        self.flatten = nn.Flatten()

        # Fully connected layers after convolution
        self.fc1 = nn.Linear(640 + shape_hold_next[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.critic_output = nn.Linear(64, 1)

    def forward(self, main_grid_input, hold_next_input):
        # Convolution and Pooling Layers for main grid input
        main_grid_input = main_grid_input.permute(0, 3, 1, 2)
        x1 = self.conv1(main_grid_input)
        x1_max = self.maxpool1(x1)
        x1_max = self.flatten(x1_max)
        x1_avg = self.avgpool1(x1)
        x1_avg = self.flatten(x1_avg)

        x2 = self.conv2(main_grid_input)
        x2_max = self.maxpool2(x2)
        x2_max = self.flatten(x2_max)
        x2_avg = self.avgpool2(x2)
        x2_avg = self.flatten(x2_avg)

        # print("====================================")
        # print("x1_max : ", x1_max.shape)
        # print("x1_avg : ", x1_avg.shape)
        # print("x2_max : ", x2_max.shape)
        # print("x2_avg : ", x2_avg.shape)
        # print("hold_next_input : ", hold_next_input.shape)
        # print("====================================")

        # Concatenate all features
        x = torch.cat([x1_max, x1_avg, x2_max, x2_avg, hold_next_input], dim=-1)
        # print("x : ", x.shape)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Output Layer
        output = self.critic_output(x)
        return output


class Agent:
    def __init__(self, turn):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        weight_file_path = os.path.join(dir_path, "outer_432_t")
        self.model = Conv2DModel()
        self.model.load_state_dict(
            torch.load(weight_file_path, map_location=torch.device("cpu"))
        )
        self.current_actions = []
        self.ok = False
        self.env_hung = Game(height=0)

    def convert_matrix_to_labels(self, matrix):
        labels = ["I", "O", "J", "L", "Z", "S", "T"]
        converted_labels = []
        for row in matrix:
            index = np.where(row == 1)[0][0]
            converted_labels.append(labels[index])
        return converted_labels

    def get_actions(self, state):
        array_2d = np.squeeze(state, axis=-1)
        broad_2010 = array_2d[:, :10]

        # Xử lý hàng rác
        count = 0
        for row in broad_2010:
            if sum(row) == 10:
                count += 1
        if count != 0:
            broad_2010 = broad_2010[:-count]

        new_row = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for i in range(count):
            broad_2010 = np.insert(broad_2010, 0, new_row, axis=0)
        # print(broad_2010.shape)

        threshold = 1.0
        boolean_array = broad_2010 == threshold
        boolean_array = boolean_array.tolist()
        # print("boolean:",boolean_array.shape)
        # output_array_3d = np.expand_dims(boolean_array, axis=-1)
        feature = array_2d[:, 10:17]
        # hold_duy = feature[0]
        piece_next_duy = feature[1:6]
        converted_piece = self.convert_matrix_to_labels(piece_next_duy)
        piece_current_duy = feature[6]
        piece_current_duy_convert = self.convert_matrix_to_labels([piece_current_duy])
        Tetromino_hung = Tetromino.new_tetromino(piece_current_duy_convert[0])
        piece_next_hungs = converted_piece
        self.env_hung.current_state.define2(
            Tetromino_hung, piece_next_hungs, boolean_array
        )
        states, add_scores, dones, _, _, moves = (
            self.env_hung.get_all_possible_states_conv2d()
        )
        rewards = get_reward(add_scores, dones)
        # q = rewards + self.model(states).cpu().numpy()
        print("====================================")
        for st in states:
            print(st.shape)
            # print("states : ", states[0].shape)
        with torch.no_grad():
            torch.tensor(state, dtype=torch.float32)
            q = (
                rewards
                + self.model(
                    torch.tensor(states[0], dtype=torch.float32),
                    torch.tensor(states[1], dtype=torch.float32),
                )
                .cpu()
                .numpy()
            )
        best = np.argmax(q)

        best_moves = moves[best]
        best_moves_convert = []
        if not self.ok:
            self.ok = True
            best_moves_convert.append(0)

        count_turns_right = best_moves.count("turn right")
        count_turns_left = best_moves.count("turn left")
        # print("Tetromino",Tetromino_hung.type_str)
        if not ("hold" in best_moves):
            if Tetromino_hung.type_str == "I" or Tetromino_hung.type_str == "O":
                if "right" in best_moves:
                    best_moves.remove("right")
                else:
                    drop_index = best_moves.index("drop")
                    best_moves.insert(drop_index, "left")

            if Tetromino_hung.type_str == "Z" or Tetromino_hung.type_str == "S":
                if count_turns_left > 0:
                    for i in range(count_turns_left):
                        best_moves.remove("turn left")
                    best_moves.insert(0, "turn right")

                    if Tetromino_hung.type_str == "Z":
                        if "right" in best_moves:
                            best_moves.remove("right")
                        else:
                            drop_index = best_moves.index("drop")
                            best_moves.insert(drop_index, "left")

            # if Tetromino_hung.type_str=='T':
            #     if count_turns_left==2:
            #         if 'right' in best_moves:
            #             best_moves.remove('right')
            #         else:
            #             drop_index = best_moves.index('drop')
            #             best_moves.insert(drop_index, 'left')

            if Tetromino_hung.type_str == "J":
                if count_turns_left == 2 or count_turns_left == 3:
                    if "right" in best_moves:
                        best_moves.remove("right")
                    else:
                        drop_index = best_moves.index("drop")
                        best_moves.insert(drop_index, "left")
            # if Tetromino_hung.type_str=='J':
            #     if count_turns_left==3 or count_turns_left==2:
            #         if 'right' in best_moves:
            #             best_moves.remove('right')
            #         else:
            #             drop_index = best_moves.index('drop')
            #             best_moves.insert(drop_index, 'left')
            #     if count_turns_left==2:
            #         if 'right' in best_moves:
            #             best_moves.remove('right')
            #         else:
            #             drop_index = best_moves.index('drop')
            #             best_moves.insert(drop_index, 'left')

            if Tetromino_hung.type_str == "L":
                if count_turns_left == 2 or count_turns_left == 1:
                    if "left" in best_moves:
                        best_moves.remove("left")
                    else:
                        drop_index = best_moves.index("drop")
                        best_moves.insert(drop_index, "right")

        for move in best_moves:
            if move == "left":
                best_moves_convert.append(6)
                self.env_hung.step(action=move)
            elif move == "right":
                best_moves_convert.append(5)
                self.env_hung.step(action=move)
            elif move == "turn left":
                best_moves_convert.append(4)
                self.env_hung.step(action=move)
            elif move == "turn right":
                best_moves_convert.append(3)
                self.env_hung.step(action=move)
            elif move == "hold":
                best_moves_convert.append(1)
                self.env_hung.step(action=move)
            elif move == "drop":
                best_moves_convert.append(2)
                self.env_hung.step(action=move)
            elif move == "down":
                best_moves_convert.append(7)
                self.env_hung.step(action=move)
            else:
                best_moves_convert.append(2)
                self.env_hung.step(action=move)
        self.env_hung.step(chosen=best)
        # print("grid:", self.env_hung.current_state.grid)
        return best_moves_convert

    def choose_action(self, state):
        if len(self.current_actions) > 0:
            return self.current_actions.pop(0)
        else:
            self.current_actions = self.get_actions(state)
            return self.current_actions.pop(0)

    # def choose_action(self, obs):
    #     return random.randint(0, 7)
