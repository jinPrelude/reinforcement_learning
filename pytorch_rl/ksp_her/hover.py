import krpc
import time
import numpy as np

"""
Kerbal Space Program reinforcement learning environment



"""

# hover_v0 returns continuous reward
class hover_v0:
    def __init__(self, sas=True, max_altitude = 1000, max_step=100):
        self.conn = krpc.connect(name='hover')
        self.vessel = self.conn.space_center.active_vessel
        self.step_count = 0
        self.done = False
        self.reward = 0
        self.max_altitude = max_altitude
        self.observation_space = 1

        # Action space : 0.0 ~ 1.0 (Thrust ratio)
        self.action_space = 1
        self.action_max = 1.
        self.action_min = 0.0

        self.initial_throttle = self.action_min

        # Initializing
        self.sas = sas
        self.goal = 100
        self.max_step = max_step

    # reset() returns target_altitude, while step() don't.
    def reset(self):

        # Quicksave initial state
        self.conn.space_center.quicksave()

        # Initialize sas
        self.vessel.control.sas = self.sas

        # Initialize throttle
        self.vessel.control.throttle = self.initial_throttle

        # Set target altitude
        self.goal = np.random.randint(100, self.max_altitude)

        print('Target altitude : ', self.goal)
        self.step_count = 0
        self.done = False
        self.reward = 0

        # Launch
        self.vessel.control.activate_next_stage()

        return self.vessel, self.goal

    def step(self, action):
        self.decision(action)

        if self.step_count >= self.max_step :
            # Revert to launch pad when the step is reached to the max_step.
            self.done = True
            self.conn.space_center.quickload()
        else :
            self.step_count += 1

            # Return the reward according to the distance between the target altitude and current altitude
            self.reward = -abs(self.vessel.flight().mean_altitude - self.goal)

        time.sleep(0.085) # Computing time considered.

        # obs, reward, done
        return self.vessel, self.reward, self.done

    # Return action
    def decision(self, action):
        self.vessel.control.throttle = float(action[0])

# hover_v1 returns sparse reward
class hover_v1:
    def __init__(self, sas=True, max_altitude = 1000, max_step=100, epsilon=1):
        self.conn = krpc.connect(name='hover')
        self.vessel = self.conn.space_center.active_vessel

        # speed
        self.srf_frame = self.vessel.orbit.body.reference_frame

        self.step_count = 0
        self.done = False
        self.reward = 0
        self.max_altitude = max_altitude
        self.observation_space = 2
        self.goal_space = 1

        # error tolerance(meter).
        # If epsilon is 1 and target_altitude is 100m, the error tolerance is between 99m and 101m
        self.epsilon = epsilon

        # Action space : 0.0 ~ 1.0 (Thrust ratio)
        self.action_space = 1
        self.action_max = 1.
        self.action_min = 0.0

        self.initial_throttle = self.action_min

        # Initializing
        self.sas = sas
        self.goal = 100
        self.max_step = max_step

    def reset(self):

        # Quicksave initial state
        self.conn.space_center.quicksave()

        # Initialize sas
        self.vessel.control.sas = self.sas

        # Initialize throttle
        self.vessel.control.throttle = self.initial_throttle

        # Set target altitude
        self.goal = np.random.randint(100, self.max_altitude)

        print('Target altitude : ', self.goal)
        self.step_count = 0
        self.done = False
        self.reward = 0

        # Launch
        self.vessel.control.activate_next_stage()

        return [self.vessel.thrust, self.vessel.flight().speed], [self.goal]

    def r(self, speed, altitude, goal):
        if (altitude <= goal + self.epsilon) and \
                (altitude >= goal - self.epsilon) and \
                (speed < 2.):
            return 1
        else:
            return 0

    def step(self, action):
        self.decision(action)

        if self.step_count >= self.max_step :
            # Revert to launch pad
            self.done = True
            self.conn.space_center.quickload()
        else :
            self.step_count += 1

            # Return the reward if the current altitude is between the error tolerance.
            self.reward = self.r(self.vessel.flight().speed, self.vessel.flight().mean_altitude, self.goal)

        time.sleep(0.085) # Computing time considered.

        # obs, reward, done
        # print('thrust : ', self.vessel.thrust)
        # print('speed : ', self.vessel.flight(self.srf_frame).speed)
        # print()
        return [self.vessel.thrust, self.vessel.flight().vertical_speed], [self.vessel.flight().mean_altitude], self.reward, self.done


    # Return action
    def decision(self, action):
        self.vessel.control.throttle = float(action[0])