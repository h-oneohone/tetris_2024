import numpy as np

ipieces = [[[0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]],
                          [[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [1, 1, 1, 1],
                           [0, 0, 0, 0]],
                                         [[0, 1, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 1, 0, 0]],
                                                        [[0, 0, 0, 0],
                                                         [1, 1, 1, 1],
                                                         [0, 0, 0, 0],
                                                         [0, 0, 0, 0]]]
opieces = [[[0, 0, 0, 0],
            [0, 2, 2, 0],
            [0, 2, 2, 0],
            [0, 0, 0, 0]],
                          [[0, 0, 0, 0],
                           [0, 2, 2, 0],
                           [0, 2, 2, 0],
                           [0, 0, 0, 0]],
                                         [[0, 0, 0, 0],
                                          [0, 2, 2, 0],
                                          [0, 2, 2, 0],
                                          [0, 0, 0, 0]],
                                                        [[0, 0, 0, 0],
                                                         [0, 2, 2, 0],
                                                         [0, 2, 2, 0],
                                                         [0, 0, 0, 0]]]

jpieces = [[[0, 3, 3, 0],
            [0, 0, 3, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 0]],
                          [[0, 0, 0, 0],
                           [0, 3, 3, 3],
                           [0, 3, 0, 0],
                           [0, 0, 0, 0]],
                                         [[0, 0, 3, 0],
                                          [0, 0, 3, 0],
                                          [0, 0, 3, 3],
                                          [0, 0, 0, 0]],
                                                        [[0, 0, 0, 3],
                                                         [0, 3, 3, 3],
                                                         [0, 0, 0, 0],
                                                         [0, 0, 0, 0]]]

lpieces = [[[0, 0, 4, 0],
            [0, 0, 4, 0],
            [0, 4, 4, 0],
            [0, 0, 0, 0]],
                          [[0, 0, 0, 0],
                           [0, 4, 4, 4],
                           [0, 0, 0, 4],
                           [0, 0, 0, 0]],
                                         [[0, 0, 4, 4],
                                          [0, 0, 4, 0],
                                          [0, 0, 4, 0],
                                          [0, 0, 0, 0]],
                                                        [[0, 4, 0, 0],
                                                         [0, 4, 4, 4],
                                                         [0, 0, 0, 0],
                                                         [0, 0, 0, 0]]]
zpieces = [[[0, 5, 0, 0],
            [0, 5, 5, 0],
            [0, 0, 5, 0],
            [0, 0, 0, 0]],
                          [[0, 0, 0, 0],
                           [0, 5, 5, 0],
                           [5, 5, 0, 0],
                           [0, 0, 0, 0]],
                                         [[0, 5, 0, 0],
                                          [0, 5, 5, 0],
                                          [0, 0, 5, 0],
                                          [0, 0, 0, 0]],
                                                        [[0, 0, 5, 5],
                                                         [0, 5, 5, 0],
                                                         [0, 0, 0, 0],
                                                         [0, 0, 0, 0]]]
spieces = [[[0, 0, 6, 0],
            [0, 6, 6, 0],
            [0, 6, 0, 0],
            [0, 0, 0, 0]],
                          [[0, 0, 0, 0],
                           [0, 6, 6, 0],
                           [0, 0, 6, 6],
                           [0, 0, 0, 0]],
                                         [[0, 0, 6, 0],
                                          [0, 6, 6, 0],
                                          [0, 6, 0, 0],
                                          [0, 0, 0, 0]],
                                                        [[6, 6, 0, 0],
                                                         [0, 6, 6, 0],
                                                         [0, 0, 0, 0],
                                                         [0, 0, 0, 0]]]


tpieces = [[[0, 0, 7, 0],
            [0, 7, 7, 0],
            [0, 0, 7, 0],
            [0, 0, 0, 0]],
                          [[0, 0, 0, 0],
                           [0, 7, 7, 7],
                           [0, 0, 7, 0],
                           [0, 0, 0, 0]],
                                         [[0, 0, 7, 0],
                                          [0, 0, 7, 7],
                                          [0, 0, 7, 0],
                                          [0, 0, 0, 0]],
                                                        [[0, 0, 7, 0],
                                                         [0, 7, 7, 7],
                                                         [0, 0, 0, 0],
                                                         [0, 0, 0, 0]]]


class Agent:
    def __init__(self, turn):
        self.GRID_WIDTH = 10
        self.GRID_DEPTH = 20
        self.PIECE_NUM2TYPE = {1: 'I', 2: 'O', 3: 'J', 4: 'L', 5: 'Z', 6: 'S', 7: 'T', 8: 'G'}
        self.BLOCK_LENGTH = 4
        self.BLOCK_WIDTH = 4
        self.px = 4
        self.py = -2
        self.width = 10
        self.height = 20

        self.current_actions = []
        self.line_cleared = 0

        self.best_genes = [0.6319, 0.0415, 0.2665, 0.3827, 0.4746, 0.0402, 0.1174, -0.4870]
        # self.best_genes = [0.7310, 0.1258, 0.0579, 0.1915, 0.4057, 0.1351, 0.2595, -0.0798]
        # self.best_genes = [0.6149, 0.0771, 0.383, 0.1545, 0.4882, 0.2583, 0.1381, -0.2664]
        self.genes = ['holeCountMultiplier', 'maximumLineHeightMultiplier', 'addedShapeHeightMultiplier', 'pillarCountMultiplier',
                      'blocksInRightMostLaneMultiplier', 'nonTetrisClearPenalty', 'bumpinessMultiplier', 'tetrisRewardMultiplier']
        self.PIECES_DICT = {
            'I': ipieces, 'O': opieces, 'J': jpieces,
            'L': lpieces, 'Z': zpieces, 'S': spieces,
            'T': tpieces
        }

    def state_to_infos(self, state):
        state = np.squeeze(state)
        matrix_block = np.transpose(state[:, :10], (1, 0))  # 10 x 20
        matrix_block = np.where(matrix_block < 1, 0, matrix_block)

        feature_vector = state[:, 10:17]
        hold_shape_vector = feature_vector[0]
        next_shape_vector = feature_vector[1:6]
        current_shape_vector = feature_vector[6]

        return matrix_block, hold_shape_vector, next_shape_vector, current_shape_vector

    def shape_to_block(self, shape):
        return self.PIECES_DICT[shape]

    def vector_to_shape(self, vector):
        for i in range(len(vector)):
            if vector[i] == 1:
                return self.PIECE_NUM2TYPE[i + 1]
        return 0

    def get_feasible(self, now_block):
        feasibles = []

        for x in range(self.BLOCK_WIDTH):
            for y in range(self.BLOCK_LENGTH):
                if now_block[x][y] > 0:
                    feasibles.append([x, y])
        return feasibles

    def collide(self, grid, block, px, py):
        feasibles = self.get_feasible(block)

        for pos in feasibles:
            if px + pos[0] > self.GRID_WIDTH - 1:
                return True
            if px + pos[0] < 0:
                return True
            if py + pos[1] > len(grid[0]) - 1:
                return True
            if py + pos[1] < 0:
                continue
            if grid[px + pos[0]][py + pos[1]] > 0:
                return True
        return False

    def collide_down(self, grid, block, px, py):
        return self.collide(grid, block, px, py + 1)

    def collide_left(self, grid, block, px, py):
        return self.collide(grid, block, px - 1, py)

    def collide_right(self, grid, block, px, py):
        return self.collide(grid, block, px + 1, py)

    def hard_drop(self, grid, block, px, py):
        x = y = 0
        if self.collide_down(grid, block, px, py) == False:
            x = 1
        if x == 1:
            while True:
                py += 1
                y += 1
                if self.collide_down(grid, block, px, py) == True:
                    break
        return y

    def final_position(self, grid, block, px, py):
        y = self.hard_drop(grid, block, px, py)
        py += y
        excess = len(grid[0]) - self.GRID_DEPTH
        block_height = 0

        for x in range(self.BLOCK_WIDTH):
            for y in range(self.BLOCK_LENGTH):
                if block[x][y] > 0:
                    if self.GRID_WIDTH > px + x > -1 and len(grid[0]) > py + y > -1:
                        grid[px + x][py + y - excess] = 1
                        block_height = max(block_height, self.GRID_DEPTH - (py + y - excess))
        return grid, block_height

    def all_final_positions_moved_shape(self, matrix_block, block, rotated):
        pos = []
        for move in range(-self.GRID_WIDTH // 2, self.GRID_WIDTH // 2 + 1):
            matrix_block_c = matrix_block.copy()
            if rotated > 1:
                actions = [4 for i in range(4 - rotated)]
            else:
                actions = [3 for i in range(rotated)]

            if move > 0:
                if self.collide_right(matrix_block_c, block, self.px + move - 1, self.py) == False:
                    matrix_block_c, block_height = self.final_position(matrix_block_c, block, self.px + move, self.py)
                    actions += [5 for _ in range(move)]
                else:
                    continue
            elif move < 0:
                if self.collide_left(matrix_block_c, block, self.px + move + 1, self.py) == False:
                    matrix_block_c, block_height = self.final_position(matrix_block_c, block, self.px + move, self.py)
                    actions += [6 for _ in range(-move)]
                else:
                    continue
            else:
                matrix_block_c, block_height = self.final_position(matrix_block_c, block, self.px + move, self.py)
            actions.append(2)
            pos.append([actions, matrix_block_c, block_height])
        return pos

    def all_final_positions(self, matrix_block, shape):
        pos = []
        if shape == 'O':
            block = self.shape_to_block(shape)[0]
            pos = self.all_final_positions_moved_shape(matrix_block, block, -1)

        elif shape in ['I', 'Z', 'S']:
            for i in range(2, 4):
                block = self.shape_to_block(shape)[i]
                pos += self.all_final_positions_moved_shape(matrix_block, block, i)
        else:
            for i in range(4):
                block = self.shape_to_block(shape)[i]
                pos += self.all_final_positions_moved_shape(matrix_block, block, i)

        return pos

    def count_holes(self, matrix_block):
        n_hole = 0
        for x in range(self.width):
            for y in reversed(range(self.height - 1)):
                if sum(matrix_block[:, y + 1]) == 0:
                    continue
                pre_occupied = (matrix_block[x, y + 1] == 1)
                if not pre_occupied and matrix_block[x, y] == 1:
                    n_hole += 1
        return n_hole

    def calc_max_line_height(self, matrix_block):
        max_line_height = 0
        for i in range(self.width):
            for j in range(self.height):
                if matrix_block[i][j] == 1:
                    max_line_height = max(max_line_height, self.height - j)
                    break;

        return max_line_height

    def count_pillars(self, matrix_block):
        pillar_cnt = 0
        for i in range(self.width):
            current_pillar_height_L = 0
            current_pillar_height_R = 0

            for j in reversed(range(self.height - 1)):
                if (i > 0 and matrix_block[i][j] != 0) and matrix_block[i - 1][j] == 0:
                    current_pillar_height_L += 1
                else:
                    if current_pillar_height_L >= 3:
                        pillar_cnt += current_pillar_height_L
                    current_pillar_height_L = 0

                if (i < self.width - 2 and matrix_block[i][j] != 0 and matrix_block[i + 1][j] == 0):
                    current_pillar_height_R += 1
                else:
                    if current_pillar_height_R >= 3:
                        pillar_cnt += current_pillar_height_R
                    current_pillar_height_R = 0

            if current_pillar_height_R >= 3:
                pillar_cnt += current_pillar_height_R
            if current_pillar_height_L >= 3:
                pillar_cnt += current_pillar_height_L

        return pillar_cnt

    def count_number_of_block_in_right_lane(self, matrix_block):
        blocks_in_right_lane = 0
        for j in range(self.height):
            if matrix_block[self.width - 1][j] != 0:
                blocks_in_right_lane += 1

        return blocks_in_right_lane

    def calc_bumpiness(self, matrix_block):
        bumpiness = 0
        previous_line_height = 0
        for i in range(self.width - 1):
            for j in range(self.height):
                if matrix_block[i][j] != 0:
                    current_line_height = self.height - j
                    if i != 0:
                        bumpiness += abs(previous_line_height - current_line_height)
                    previous_line_height = current_line_height
                    break

        return bumpiness

    def clear_lines(self, matrix_block):
        cleared = 0
        matrix_block = matrix_block.tolist()
        for y in reversed(range(self.height)):
            y = -(y + 1)
            row = 0
            for x in range(self.width):
                if matrix_block[x][y] == 1:
                    row += 1

            if row == self.width:
                cleared += 1
                for i in range(self.width):
                    del matrix_block[i][y]
                    matrix_block[i] = [0] + matrix_block[i]

        return np.array(matrix_block), cleared

    def calc_cost(self, matrix_block, block_height):
        dict_genes = {key: value for key, value in zip(self.genes, self.best_genes)}

        matrix_block, line_cleared = self.clear_lines(matrix_block)
        hole_cnt = self.count_holes(matrix_block)
        max_line_height = self.calc_max_line_height(matrix_block)
        pillar_cnt = self.count_pillars(matrix_block)
        block_in_right_lane = self.count_number_of_block_in_right_lane(matrix_block)
        bumpiness = self.calc_bumpiness(matrix_block)

        lines_clear_which_arent_tetrises = 1 if (line_cleared > 0 and line_cleared < 3) else 0
        tetrises = 1 if line_cleared == 4 else 0

        return (dict_genes['holeCountMultiplier'] * hole_cnt +
                dict_genes['maximumLineHeightMultiplier'] * max_line_height +
                dict_genes['pillarCountMultiplier'] * pillar_cnt +
                dict_genes['blocksInRightMostLaneMultiplier'] * block_in_right_lane +
                dict_genes['nonTetrisClearPenalty'] * lines_clear_which_arent_tetrises +
                dict_genes['bumpinessMultiplier'] * bumpiness +
                dict_genes['addedShapeHeightMultiplier'] * block_height +
                dict_genes['tetrisRewardMultiplier'] * tetrises), line_cleared

    def find_best_movement_plan(self, all_matrix_block):
        min_holes = 10000
        min_holes_matrix = []

        for [actions, matrix, shape_height] in all_matrix_block:
            matrix_holes = self.count_holes(matrix)
            if matrix_holes < min_holes:
                min_holes = matrix_holes

        for [actions, matrix, shape_height] in all_matrix_block:
            matrix_holes = self.count_holes(matrix)

            if matrix_holes == min_holes:
                min_holes_matrix.append([actions, matrix, shape_height])

        min_score = 1e9
        min_score_matrix_actions = []
        min_score_matrix_line_cleared = 0

        for [actions, matrix, shape_height] in min_holes_matrix:
            score, line_cleared = self.calc_cost(matrix, shape_height)
            if min_score > score:
                min_score = score
                min_score_matrix_actions = actions
                min_score_matrix_line_cleared = line_cleared
        return min_score_matrix_actions, min_score_matrix_line_cleared

    def calc_best_movement_plan(self, state):
        matrix_block, hold_block_vector, next_blocks_vector, current_block_vector = self.state_to_infos(state)
        current_shape = self.vector_to_shape(current_block_vector)
        hold_shape = self.vector_to_shape(hold_block_vector)
        next_shape = self.vector_to_shape(next_blocks_vector[0])

        candidates_shape = {
            0: current_shape,
            1: hold_shape if hold_shape != 0 else next_shape
        }

        all_possible_matrix_block_shape = []
        for hold, shape in candidates_shape.items():
            end_matrix_block_shape = self.all_final_positions(matrix_block, shape)
            if hold == 1:
                for actions in end_matrix_block_shape:
                    actions[0] = [1] + actions[0]
            all_possible_matrix_block_shape += end_matrix_block_shape

        best_actions, best_line_cleared = self.find_best_movement_plan(all_possible_matrix_block_shape)

        self.line_cleared = best_line_cleared

        return best_actions

    def get_actions(self, state):
        actions = self.calc_best_movement_plan(state)
        return actions

    def choose_action(self, state):
        if len(self.current_actions) > 0:
            self.line_cleared = 0
            return self.current_actions.pop(0)
        else:
            self.current_actions = self.get_actions(state)
            return self.current_actions.pop(0)