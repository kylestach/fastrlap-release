import numpy as np
import random

GOAL_THRESHOLD = 0.3

class GoalGraph():
    """
    A graph of goals that the car can drive to. Once the car arrives at a goal,
    the goal will be changed to one of its successors.
    """
    def __init__(self, scale, goal_config='corners', goal_threshold=GOAL_THRESHOLD):
        if goal_config == 'full':
            self.goals = [
                (0.05, 0.05),
                (0.33, -0.15),
                (0.3, -0.45),
                (0.65, -0.1),
                (0.525, 0.2),
                (0.6, 0.75),
                (0.3, 0.6),
                (-0.2, 0.6),
                (-0.5, 0.57),
                (-0.6, 0.05),
                (-0.7, -0.6),
                (0.0, -0.8),
            ]

            self.graph = {
                0: [1, 6, 7, 9, 10, 11],
                1: [0, 2, 3, 4],
                2: [1, 3, 11],
                3: [1, 2, 4],
                4: [1, 3, 6],
                5: [4, 6],
                6: [0, 4, 5, 7],
                7: [0, 6, 8],
                8: [7, 9],
                9: [0, 8, 10],
                10: [0, 9, 11],
                11: [0, 2, 10],
            }

            self.start_headings = [(0, 2 * np.pi)] * len(self.goals)
        elif goal_config == 'edges':
            self.goals = [
                (0.9, 0.0),
                (0.0, 0.9),
                (-0.9, 0.0),
                (0.0, -0.9),
            ]

            self.start_headings = [
                (a - 0.1, a + 0.1) for a in [
                    np.deg2rad(90),
                    np.deg2rad(180),
                    np.deg2rad(-90),
                    np.deg2rad(0),
                ]
            ]

            self.graph = {
                0: [1],
                1: [2],
                2: [3],
                3: [0],
            }
        elif goal_config == 'corners':
            self.goals = [
                (0.9, 0.9),
                (-0.9, 0.9),
                (-0.9, -0.9),
                (0.9, -0.9),
            ]

            self.start_headings = [
                (a - 0.1, a + 0.1) for a in [
                    np.deg2rad(135),
                    -np.deg2rad(135),
                    -np.deg2rad(45),
                    np.deg2rad(45),
                ]
            ]

            self.graph = {
                0: [1],
                1: [2],
                2: [3],
                3: [0],
            }
        elif goal_config == 'ring_dense':
            self.goals = [
                (0.9, 0.9),
                (0.0, 0.9),
                (-0.9, 0.9),
                (-0.9, 0.0),
                (-0.9, -0.9),
                (0.0, -0.9),
                (0.9, -0.9),
                (0.9, 0.0),
            ]

            self.start_headings = [
                (a - 0.1, a + 0.1) for a in [
                    np.deg2rad(135),
                    np.deg2rad(-180),
                    -np.deg2rad(135),
                    np.deg2rad(-90),
                    -np.deg2rad(45),
                    np.deg2rad(0),
                    -np.deg2rad(45),
                    np.deg2rad(90),
                ]
            ]

            self.graph = {
                0: [1, 2],
                1: [2, 3],
                2: [3, 4],
                3: [4, 5],
                4: [5, 6],
                5: [6, 7],
                6: [7, 0],
                7: [0, 1],
            }
        elif goal_config == 'ring_small_inner':
            self.goals = [
                (-0.10888671875, 0.609375),
                (0.07421875, 0.072265625),
                (-0.072265625, -0.220703125),
                (-0.5849609375, 0.072265625),
                (-0.49951171875, 0.67041015625),
            ]

            self.start_headings = [
                (a - 0.1, a + 0.1) for a in [
                    np.deg2rad(-45),
                    np.deg2rad(-110),
                    np.deg2rad(-120),
                    np.deg2rad(110),
                    np.deg2rad(45),
                ]
            ]

            self.graph = {
                0: [1],
                1: [2],
                2: [3],
                3: [4],
                4: [0],
            }
        elif goal_config == 'small_inner_1fork':
            self.goals = [
                (-0.10888671875, 0.609375),
                (0.07421875, 0.072265625),
                (-0.072265625, -0.220703125),
                (-0.5849609375, 0.072265625),
                (-0.49951171875, 0.67041015625),
                (-0.2919921875, -0.5380859375),
                (-0.6826171875, -0.46484375)
            ]

            self.start_headings = [
                (a - 0.1, a + 0.1) for a in [
                    np.deg2rad(-45),
                    np.deg2rad(-110),
                    np.deg2rad(-120),
                    np.deg2rad(110),
                    np.deg2rad(45),
                    np.deg2rad(-160),
                    np.deg2rad(90),
                ]
            ]

            self.graph = {
                0: [1],
                1: [2],
                2: [3, 5],
                3: [4],
                4: [0],
                5: [6],
                6: [3],
            }
        elif goal_config == 'small_inner_graph':
            self.goals = [
                (-0.10888671875, 0.609375),
                (0.0498046875, 0.1943359375),
                (-0.072265625, -0.220703125),
                (-0.5849609375, 0.072265625),
                (-0.49951171875, 0.67041015625),
                (-0.2919921875, -0.5380859375),
                (-0.6826171875, -0.46484375),
                (0.2939453125, -0.0498046875),
                (0.2939453125, -0.4892578125),
                (0.025390625, -0.806640625),
                (0.28, 0.53),
            ]

            self.start_headings = [
                (a - 0.3, a + 0.3) for a in [
                    np.deg2rad(-45),
                    np.deg2rad(-90),
                    np.deg2rad(-120),
                    np.deg2rad(110),
                    np.deg2rad(45),
                    np.deg2rad(-160),
                    np.deg2rad(90),
                    np.deg2rad(-30),
                    np.deg2rad(-110),
                    np.deg2rad(120),
                    np.deg2rad(90),
                ]
            ]

            self.graph = {
                0: [1],
                1: [2, 7, 10],
                2: [3, 5],
                3: [4],
                4: [0],
                5: [6],
                6: [3],
                7: [8, 10],
                8: [9],
                9: [5],
                10: [0],
            }
        elif goal_config == 'small_inner_graph_reversed':
            self.goals = [
                (-0.10888671875, 0.609375),
                (0.0498046875, 0.1943359375),
                (-0.072265625, -0.220703125),
                (-0.5849609375, 0.072265625),
                (-0.49951171875, 0.67041015625),
                (-0.2919921875, -0.5380859375),
                (-0.6826171875, -0.46484375),
                (0.2939453125, -0.0498046875),
                (0.2939453125, -0.4892578125),
                (0.025390625, -0.806640625),
                (0.28, 0.53),
            ]

            self.start_headings = [
                (a - 0.3 + np.pi, a + 0.3 + np.pi) for a in [
                    np.deg2rad(-45),
                    np.deg2rad(-90),
                    np.deg2rad(-120),
                    np.deg2rad(110),
                    np.deg2rad(45),
                    np.deg2rad(-160),
                    np.deg2rad(90),
                    np.deg2rad(-30),
                    np.deg2rad(-110),
                    np.deg2rad(120),
                    np.deg2rad(90),
                ]
            ]

            self.graph = {
                0: [4, 10],
                1: [0],
                2: [1],
                3: [2, 6],
                4: [3],
                5: [2, 9],
                6: [5],
                7: [1],
                8: [7],
                9: [8],
                10: [1, 7],
            }
        else:
            raise ValueError(f"No such goal graph for {goal_config}")

        self.goal_reprs = []
        self.edge_reprs = []

        self.current_start_idx = 0
        self.current_goal_idx = 0

        self.goal_threshold = goal_threshold
        self.scale = scale

    @property
    def current_start(self):
        return np.array(self.goals[self.current_start_idx]) * self.scale, self.start_headings[self.current_start_idx]

    @property
    def current_goal(self):
        return np.array(self.goals[self.current_goal_idx]) * self.scale

    def is_complete(self):
        return self._ticks_at_current_goal > 0

    def set_goal(self, goal_idx, physics):
        """
        Set a new goal and update the renderables to match.
        """
        for idx, repr in enumerate(self.goal_reprs):
            opacity = 1.0 if idx == goal_idx else 0.0
            if physics:
                physics.bind(repr).rgba = (*repr.rgba[:3], opacity)
            else:
                repr.rgba = (*repr.rgba[:3], opacity)

        self.current_goal_idx = goal_idx
        self._ticks_at_current_goal = 0

    def tick(self, car_pos, physics):
        """
        Update the goal if the car was at the current goal for at least one tick.
        We need the delay so that the car can get the high reward for reaching
        the goal before the goal changes.
        """
        if self.is_complete():
            self.current_start_idx = self.current_goal_idx
            self.set_goal(random.choice(self.graph[self.current_start_idx]), physics)
            return True

        if np.linalg.norm(np.array(car_pos)[:2] - self.current_goal) < self.goal_threshold:
            self._ticks_at_current_goal += 1
        else:
            self._ticks_at_current_goal = 0

        return False

    def reset(self, physics):
        self.current_start_idx = random.randint(0, len(self.goals) - 1)
        self.set_goal(random.choice(self.graph[self.current_start_idx]), physics)
        self._ticks_at_current_goal = 0

    def add_renderables(self, mjcf_root, height_lookup, show_edges=False):
        """
        Add renderables to the mjcf root to visualize the goals and (optionally) edges.
        """
        RENDER_HEIGHT_OFFSET = 5.0

        self.goal_reprs = [
            mjcf_root.worldbody.add('site',
                                    type="sphere",
                                    size="0.08",
                                    rgba=(0.0, 1.0, 0.0, 0.5),
                                    pos=(g[0] * self.scale, g[1] * self.scale, height_lookup((g[0] * self.scale, g[1] * self.scale)) + RENDER_HEIGHT_OFFSET))
            for g in self.goals
        ]

        self.edge_reprs = [
            mjcf_root.worldbody.add('site',
                                    type="cylinder",
                                    size="0.04",
                                    rgba=(1, 1, 1, 0.5),
                                    fromto=(self.goals[s][0] * self.scale, self.goals[s][1] * self.scale, height_lookup((self.goals[s][0] * self.scale, self.goals[s][1] * self.scale)) + RENDER_HEIGHT_OFFSET,
                                            self.goals[g][0] * self.scale, self.goals[g][1] * self.scale, height_lookup((self.goals[g][0] * self.scale, self.goals[g][1] * self.scale)) + RENDER_HEIGHT_OFFSET))
            for s in self.graph for g in self.graph[s]
            if show_edges
        ]
