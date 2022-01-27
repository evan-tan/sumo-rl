import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import traci
from gym import spaces
from collections import deque


class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    Default observation space is a vector R^(#greenPhases + 2 * #lanes)
    s = [current phase one-hot encoded, density for each lane, queue for each lane]
    You can change this by modifing self.observation_space and the method _compute_observations()

    Action space is which green phase is going to be open for the next delta_time seconds
    """

    def __init__(
        self,
        env,
        ts_id,
        delta_time,
        yellow_time,
        min_green,
        max_green,
        begin_time,
        sumo,
    ):
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.last_flow = 0.0
        self.last_pressure = 0.0
        self.last_reward = None
        self.sumo = sumo

        self.press_reward = 0
        self.flow_rew = 0
        self.wait_reward = 0
        self.last_urgency_reward = 0
        self.last_wait_MA_reward = 0
        self._vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        self._max_capacity = 100  # number of cars
        num_steps = 20
        # urgency reward queue
        self._urq = deque(maxlen=num_steps)
        self._wait_x = deque(maxlen=num_steps)
        self._wait_10x = deque(maxlen=num_steps * 10)
        self.build_phases()

        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id))
        )  # Remove duplicates and keep order
        self.out_lanes = [
            link[0][1]
            for link in self.sumo.trafficlight.getControlledLinks(self.id)
            if link
        ]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_lenght = {
            lane: self.sumo.lane.getLength(lane) for lane in self.lanes
        }

        self.observation_space = spaces.Box(
            low=np.zeros(
                self.num_green_phases + 1 + 2 * len(self.lanes), dtype=np.float32
            ),
            high=np.ones(
                self.num_green_phases + 1 + 2 * len(self.lanes), dtype=np.float32
            ),
        )
        self.discrete_observation_space = spaces.Tuple(
            (
                spaces.Discrete(self.num_green_phases),  # Green Phase
                spaces.Discrete(
                    2
                ),  # Binary variable active if min_green seconds already elapsed
                *(
                    spaces.Discrete(10) for _ in range(2 * len(self.lanes))
                ),  # Density and stopped-density for each lane
            )
        )
        self.action_space = spaces.Discrete(self.num_green_phases)

    def build_phases(self):
        phases = self.sumo.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[
            0
        ].phases
        if self.env.fixed_ts:
            self.num_green_phases = (
                len(phases) // 2
            )  # Number of green phases == number of phases (green+yellow) divided by 2
            return

        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if "y" not in state and (state.count("r") + state.count("s") != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue
                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (
                        p2.state[s] == "r" or p2.state[s] == "s"
                    ):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i, j)] = len(self.all_phases)
                self.all_phases.append(
                    self.sumo.trafficlight.Phase(self.yellow_time, yellow_state)
                )

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    @property
    def time_to_act(self):
        return self.next_action_time == self.env.sim_step

    def update(self):
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.green_phase].state
            )
            self.is_yellow = False

    def set_next_phase(self, new_phase):
        """
        Sets what will be the next green phase and sets yellow phase if the next phase is different than the current

        :param new_phase: (int) Number between [0..num_green_phases]
        """
        new_phase = int(new_phase)
        if (
            self.green_phase == new_phase
            or self.time_since_last_phase_change < self.yellow_time + self.min_green
        ):
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.green_phase].state
            )
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            # self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id,
                self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state,
            )
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def compute_observation(self):
        phase_id = [
            1 if self.green_phase == i else 0 for i in range(self.num_green_phases)
        ]  # one-hot encoding
        min_green = [
            0
            if self.time_since_last_phase_change < self.min_green + self.yellow_time
            else 1
        ]
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def _compute_waiting_MA_reward(self):
        # avg waiting time running total
        wait_time, num_vehicles = self.get_waiting_time_per_lane()
        acc_waiting_time = (sum(wait_time) / 100) / num_vehicles
        self._wait_x.append(acc_waiting_time)
        self._wait_10x.append(acc_waiting_time)

        # negative sign because a higher waiting time is worse
        wait_reward = -(np.mean(self._wait_x) - np.mean(self._wait_10x))

        return wait_reward

    def compute_reward(self):
        # d_pressure = self._pressure_reward()
        # r_pressure = d_pressure / 1e3
        # r_flow = self.flow_reward()
        # r_wait = self._waiting_time_reward()
        # # for r_wait, signs already handled in function
        # reward = 5 * r_pressure + 2 * (r_wait/20) + r_flow

        # self.press_reward = d_pressure
        # self.flow_rew = r_flow
        # self.wait_reward = r_wait

        # urgency reward
        urgency_reward = self._compute_urgency_reward()
        wait_MA_reward = self._compute_waiting_MA_reward()

        self.last_urgency_reward = urgency_reward
        self.last_wait_MA_reward = wait_MA_reward

        self.last_reward = urgency_reward
        # return 0.5*(urgency_reward + wait_MA_reward)
        return urgency_reward

    def _compute_urgency_reward(self):
        """Compute urgency across all lanes"""
        # urgency metric
        max_speed = 13.89
        queue = np.array(self.get_lanes_queue())
        density = np.array(self.get_lanes_density())
        # elem wise multiplication
        urgency = queue * density
        # range [-0.5, 0.5] -> [-1, 1]
        avg_speed = np.array(self.get_mean_speed())
        throughput = (avg_speed / max_speed - 0.5) * 2
        return (urgency * throughput).sum()

    def flow_reward(self):
        curr_flow = self._current_flow()
        d_flow = curr_flow - self.last_flow
        self.last_flow = curr_flow
        # print(f"Delta Flow: {d_flow}")
        return d_flow

    def _current_flow(self):
        average_speed = []
        for lane in self.lanes:
            vehicle_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            for vehicle in vehicle_list:
                v_speed = self.sumo.vehicle.getSpeed(vehicle)
                average_speed.append(v_speed)

        if len(average_speed) > 0:
            flow = sum(average_speed) / len(average_speed)
        else:
            flow = 0
        return flow

    def _pressure_reward(self):
        curr_pressure = -self.get_pressure()
        d_pressure = curr_pressure - self.last_pressure
        self.last_pressure = curr_pressure
        return d_pressure

    def _queue_average_reward(self):
        new_average = np.mean(self.get_total_halted())
        reward = self.last_measure - new_average
        self.last_measure = new_average
        return reward

    def _queue_reward(self):
        return -((sum(self.get_total_halted())) ** 2)

    def _waiting_time_reward(self):
        wait_time, _ = self.get_waiting_time_per_lane()
        ts_wait = sum(wait_time) / self._max_capacity
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _waiting_time_reward2(self):
        wait_time, _ = self.get_waiting_time_per_lane()
        ts_wait = sum(wait_time)
        self.last_measure = ts_wait
        if ts_wait == 0:
            reward = 1.0
        else:
            reward = 1.0 / ts_wait
        return reward

    def _waiting_time_reward3(self):
        wait_time, _ = self.get_waiting_time_per_lane()
        ts_wait = sum(wait_time)
        reward = -ts_wait
        self.last_measure = ts_wait
        return reward

    def get_waiting_time_per_lane(self):
        wait_time_per_lane = []
        num_vehicles = 0
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            num_vehicles += len(veh_list)
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [
                            self.env.vehicles[veh][lane]
                            for lane in self.env.vehicles[veh].keys()
                            if lane != veh_lane
                        ]
                    )
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane, num_vehicles

    def get_pressure(self):
        return abs(
            sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes)
            - sum(
                self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes
            )
        )

    #####
    def get_out_lanes_density(self):
        return [
            min(
                1,
                self.sumo.lane.getLastStepVehicleNumber(lane)
                / (self.sumo.lane.getLength(lane) / self._vehicle_size_min_gap),
            )
            for lane in self.out_lanes
        ]

    def get_lanes_density(self):
        return [
            min(
                1,
                self.sumo.lane.getLastStepVehicleNumber(lane)
                / (self.lanes_lenght[lane] / self._vehicle_size_min_gap),
            )
            for lane in self.lanes
        ]

    def get_lanes_queue(self):
        return [
            min(
                1,
                self.sumo.lane.getLastStepHaltingNumber(lane)
                / (self.lanes_lenght[lane] / self._vehicle_size_min_gap),
            )
            for lane in self.lanes
        ]

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list

    #######

    def get_total_halted(self):
        return [self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes]

    def get_mean_speed(self):
        return [self.sumo.lane.getLastStepMeanSpeed(lane) for lane in self.lanes]

    def get_occupancy(self):
        return [self.sumo.lane.getLastStepOccupancy(lane) for lane in self.lanes]

    def get_vehicle_number(self):
        return [self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes]

    def get_travel_time(self):
        return [self.sumo.lane.getTraveltime(lane) for lane in self.lanes]
