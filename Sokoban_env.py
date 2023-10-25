import gym
from gym.spaces.discrete import Discrete
import numpy as np
import pkg_resources
import imageio.v2 as imageio
from itertools import permutations

MAPS = {
    "level_1":[
        "##########",
        "##########",
        "##########",
        "##########",
        "#@ $   . #",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
    ],
    "level_2":[
        "##########",
        "##########",
        "##########",
        "##########",
        "#  $   . #",
        "#@       #",
        "##########",
        "##########",
        "##########",
        "##########",
    ],
    "level_3":[
        "##########",
        "##########",
        "##########",
        "#      . #",
        "#  $     #",
        "#@       #",
        "##########",
        "##########",
        "##########",
        "##########",
    ],
    "special_1":[
        "##########",
        "##########",
        "##########",
        "##########",
        "#? @   .$#",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
    ],
    "special_2":[
        "##########",
        "##########",
        "##########",
        "##########",
        "#$     . #",
        "#@      ?#",
        "##########",
        "##########",
        "##########",
        "##########",
    ],
    "special_3":[
        "##########",
        "##########",
        "##########",
        "#      . #",
        "#$       #",
        "#@      ?#",
        "##########",
        "##########",
        "##########",
        "##########",
    ],

}
class Sokoban_v2(gym.Env):

    def __init__(
        self,
        max_steps=120,
        map_name="special_3",
    ):
        self.map_name = map_name
        self.max_steps = max_steps
        self.num_pull = 0

        self.room_fixed, self.room_state, self.box_mapping = self.select_room()
        self.targets_locs = self.get_locs(TARGET_CODE["target"])
        self.boxes_locs = self.get_locs(TARGET_CODE["box"])
        self.agent_locs = self.get_locs(TARGET_CODE["agent"])

        self.num_boxes = len(np.where(self.room_state == 4)[0])
        self.action_space = Discrete(len(ACTION_LOOKUP))
        self.all_state = self.get_all_state(self.room_fixed, self.num_boxes)
        self.num_states = Discrete(len(self.all_state))
        self.observation_space = Discrete(len(np.where(self.room_state !=0 )[0]))

        self.states_num = (self.observation_space.n - 1) * 2

        # Reward
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -1
        self.reward_box_on_target = 1
        self.reward_finished = 10
        self.reward_last = 0

    def get_locs(self, object_code):
        """     
        wall: 0,
        road: 1,
        target: 2,
        box: 4,
        agent: 5,
        ----------
        return: ([row, col])
        """
        object_rows, object_cols = np.where(self.room_state == object_code)
        
        return (object_rows, object_cols)
    
    def get_all_state(self, room_fixed, num_box):
        # Mảng ban đầu
        arr = list(room_fixed[room_fixed != 0])
        check = False
        if 7 in arr:
            arr_2 = arr.copy()
            arr_2[arr.index(7)] = 1
            check = True

        # Tìm tất cả các vị trí có thể đặt 2 giá trị 4 và 5
        positions = [i for i, x in enumerate(arr) if (x != 0)]

        # Tạo tất cả các hoán vị có thể của 2 vị trí trong số các vị trí đã tìm được

        # Tạo tất cả các trường hợp xáo trộn 2 giá trị 4 và 5
        results = []
        for perm in permutations(positions, num_box + 1):
            # Tạo một bản sao của mảng ban đầu
            temp = arr.copy()
            if check: temp_2 = arr_2.copy()
            for i in range (num_box +1):
                # Đặt giá trị 4 và 5 vào 2 vị trí trong hoán vị
                if i < num_box:
                    temp[perm[i]] = 4
                    if check: temp_2[perm[i]] = 4
                else: 
                    temp[perm[i]] = 5
                    if check: temp_2[perm[i]] = 5
                if temp[perm[i]] == 4 and arr[perm[i]] == 2:
                    temp[perm[i]] = 3
                    if check: temp_2[perm[i]] = 3
                # Thêm trường hợp xáo trộn này vào kết quả
            results.append(temp)
            if check: results.append(temp_2)
        return results
    
    def encode_state(self, all_states, state):
        return all_states.index(state)
    
    def get_state(self, agent_locs, boxes_locs):
        state = None
        return state

    def step(self, action):
        assert action in ACTION_LOOKUP

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False
        moved_player = False  # Initialize moved_player to False

        if action == 0:
            moved_player = False

        # All push actions are in the range of [0, 3]
        if action < 5:
            moved_player, moved_box = self._push(action)

        elif action < 9:
            moved_player = self._move(action)

        else:
            moved_player, moved_box = self._pull(action)

        self._calc_reward()

        done = self._check_if_done()
        state = list(self.room_state[self.room_state != 0])
        observation = self.encode_state(self.all_state, state)

        #self.boxes_locs = self.get_locs(TARGET_CODE["box"])
        #self.agent_locs = self.get_locs(TARGET_CODE["agent"])

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()

        return observation, self.reward_last, done, info

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if (
            new_box_position[0] >= self.room_state.shape[0]
            or new_box_position[1] >= self.room_state.shape[1]
        ):
            return False, False

        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [
            1,
            2,
        ]
        if can_push_box:
            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position

            if self.room_state[new_position[0], new_position[1]] == 7:
                self.room_state[new_position[0], new_position[1]] = 1
                self.room_fixed[new_position[0], new_position[1]] = 1  # Set the special box to an empty field
                self.num_pull = 1  # Set self.num_pull to 1

            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = self.room_fixed[
                current_position[0], current_position[1]
            ]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 5) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # Check if the field in the moving direction is an empty field or a target
        if self.room_state[new_position[0], new_position[1]] in [1, 2, 7]:
            self.player_position = new_position

            # Check if the player moved to a special box
            if self.room_state[new_position[0], new_position[1]] == 7:
                self.room_state[new_position[0], new_position[1]] = 1
                self.room_fixed[new_position[0], new_position[1]] = 1  # Set the special box to an empty field
                self.num_pull = 1  # Set self.num_pull to 1
            
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = self.room_fixed[
                current_position[0], current_position[1]
            ]

            return True

        return False

    def _pull(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()
        pull_content_position = self.player_position - change

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.num_pull == 0:
            return False, False
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            box_next_to_player = self.room_state[pull_content_position[0], pull_content_position[1]] in [3, 4]
            if box_next_to_player:
                # Move Box
                box_type = 4
                if self.room_fixed[current_position[0], current_position[1]] == 2:
                    box_type = 3
                self.room_state[current_position[0], current_position[1]] = box_type
                self.room_state[pull_content_position[0], pull_content_position[1]] = \
                    self.room_fixed[pull_content_position[0], pull_content_position[1]]
                self.num_pull = 0

            return True, box_next_to_player

        return False, False

    def _calc_reward(self):
        """
        Calculate Reward Based on
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        # count boxes off or on the target
        empty_targets = self.room_state == 2
        player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
        total_targets = empty_targets | player_on_target

        current_boxes_on_target = self.num_boxes - np.where(total_targets)[0].shape[0]

        # Add the reward if a box is pushed on the target and give a
        # penalty if a box is pushed off the target.
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.boxes_on_target:
            self.reward_last += self.penalty_box_off_target

        game_won = self._check_if_all_boxes_on_target()
        if game_won:
            self.reward_last += self.reward_finished

        self.boxes_on_target = current_boxes_on_target
        return self.reward_last
        
        

    def _check_if_done(self):
        # Check if the game is over either through reaching the maximum number
        # of available steps or by pushing all boxes on the targets.
        return self._check_if_all_boxes_on_target() or self._check_if_maxsteps()

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = (
            np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        )
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return self.max_steps == self.num_env_steps

    def reset(self):
        self.room_fixed, self.room_state, self.box_mapping = self.select_room()

        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        state = list(self.room_state[self.room_state != 0])
        starting_observation = self.encode_state(self.all_state, state)
        # starting_observation = self.room_state

        return starting_observation

    def select_room(self):
        selected_map = MAPS[self.map_name]
        room_fixed, room_state, box_mapping = self.generate_room(selected_map)
        return room_fixed, room_state, box_mapping

    def render(self):
        img = self.get_image()
        return img

    def get_image(self):
        img = self.room_to_rgb(self.room_state, self.room_fixed)

        return img

    def room_to_rgb(self, room, room_structure=None):
        """
        Creates an RGB image of the room.
        :param room:
        :param room_structure:
        :return:
        """
        resource_package = __name__

        room = np.array(room)
        if not room_structure is None:
            # Change the ID of a player on a target
            room[(room == 5) & (room_structure == 2)] = 6

        # Load images, representing the corresponding situation
        box_filename = pkg_resources.resource_filename(
            resource_package, "/".join(("surface", "snail.png"))
        )
        box = imageio.imread(box_filename)

        box_on_target_filename = pkg_resources.resource_filename(
            resource_package, "/".join(("surface", "snail_on_target.png"))
        )
        box_on_target = imageio.imread(box_on_target_filename)

        box_target_filename = pkg_resources.resource_filename(
            resource_package, "/".join(("surface", "target.bmp"))
        )
        box_target = imageio.imread(box_target_filename)

        floor_filename = pkg_resources.resource_filename(
            resource_package, "/".join(("surface", "floor.png"))
        )
        floor = imageio.imread(floor_filename)

        player_filename = pkg_resources.resource_filename(
            resource_package, "/".join(("surface", "smuft.png"))
        )
        player = imageio.imread(player_filename)

        player_on_target_filename = pkg_resources.resource_filename(
            resource_package, "/".join(("surface", "smuft_on_target.bmp"))
        )
        player_on_target = imageio.imread(player_on_target_filename)

        wall_filename = pkg_resources.resource_filename(
            resource_package, "/".join(("surface", "grass.bmp"))
        )
        wall = imageio.imread(wall_filename)

        special_filename = pkg_resources.resource_filename(
            resource_package, "/".join(("surface", "special.bmp"))
        )
        special = imageio.imread(special_filename)

        surfaces = [
            wall,
            floor,
            box_target,
            box_on_target,
            box,
            player,
            player_on_target,
            special,
        ]

        # Assemble the new rgb_room, with all loaded images
        room_rgb = np.zeros(
            shape=(room.shape[0] * 16, room.shape[1] * 16, 3), dtype=np.uint8
        )
        for i in range(room.shape[0]):
            x_i = i * 16

            for j in range(room.shape[1]):
                y_j = j * 16
                surfaces_id = room[i, j]

                room_rgb[x_i : (x_i + 16), y_j : (y_j + 16), :] = surfaces[surfaces_id]

        return room_rgb

    def generate_room(self, select_map):
        room_fixed = []
        room_state = []

        targets = []
        boxes = []
        for row in select_map:
            room_f = []
            room_s = []

            for e in row:
                if e == "#":
                    room_f.append(0)
                    room_s.append(0)

                elif e == "@":
                    self.player_position = np.array([len(room_fixed), len(room_f)])
                    room_f.append(1)
                    room_s.append(5)

                elif e == "$":
                    boxes.append((len(room_fixed), len(room_f)))
                    room_f.append(1)
                    room_s.append(4)

                elif e == ".":
                    targets.append((len(room_fixed), len(room_f)))
                    room_f.append(2)
                    room_s.append(2)

                elif e == "?":
                    room_f.append(7)
                    room_s.append(7)

                else:
                    room_f.append(1)
                    room_s.append(1)

            room_fixed.append(room_f)
            room_state.append(room_s)

        # used for replay in room generation, unused here because pre-generated levels
        box_mapping = {}

        return np.array(room_fixed), np.array(room_state), box_mapping
    
    
    def available_states(self,state):
        self.available_states = []
        for i in range(10):
            for j in range(10):
                if self.room_state[i][j] != 0:
                    self.available_states.append((i, j))
        return self.available_states
                
    def get_next_states(self,state):
        self.new_states = []
        for action, i in enumerate(CHANGE_COORDINATES.values()):
            state_ = state + np.array(i)
            if tuple(state_) in self.available_states:
                self.new_states.append((state_, action))
        return self.new_states
    def create_transition_pos(self):
        self.P = {}
        for state in self.available_states:
            for next_states, action in self.get_next_states(state):
                self.P[(state, action)] = [(next_state, 1.0 / len(next_states)) for next_state in next_states]
        return self.P
 
    def close(self):
        return None
    
    


ACTION_LOOKUP = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right',
    5: 'move up',
    6: 'move down',
    7: 'move left',
    8: 'move right',
    9: 'pull up',
    10: 'pull down',
    11: 'pull left',
    12: 'pull right',
}
# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

TARGET_CODE = {
    "wall": 0,
    "road": 1,
    "target": 2,
    "box": 4,
    "agent": 5,
    "special": 7,
}