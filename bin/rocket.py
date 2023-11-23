
# import modules
import numpy as np
import abc

from define import *
from environment import Environment
from rocket_stage import *
from logger import Logger
import coordinate_system as cs


class SimulationStatus():
	"""Structure that stores interface information with Simulator
	"""
	def __init__(self, status, num=None, target_time=None) -> None:
		self.status = status # int -> 0: initialize, 1: in loop, 2: when one loop ended, 3: simulation finished
		self.num = num
		self.target_time = target_time


class RocketStatus():
	"""Structure that stores interface information with Simulator
	"""
	def __init__(self, obj=None, y0=None, calc_start_time=None, is_continue=True, is_flying=None, max_values=None, stage_name=None, impact_point=None, id=None) -> None:
		self.obj = obj
		self.y0 = y0
		self.calc_start_time = calc_start_time
		self.is_continue = is_continue
		self.is_flying = is_flying
		self.max_values = max_values
		self.stage_name = stage_name
		self.impact_point = impact_point
		self.id = id


class IRocketStatus(metaclass=abc.ABCMeta):
	"""Defines the interfaces required by classes run from the Simulator
	"""
	@abc.abstractclassmethod
	def get_status(self, simulation_status:SimulationStatus) -> RocketStatus:
		"""Return rocket state depending on simulation status
		"""
		raise NotImplementedError()


class IOdeSolver(metaclass=abc.ABCMeta):
	"""Defines the interfaces required by classes run from the ode solver
	"""
	@abc.abstractclassmethod
	def __call__(self, t, x):
		"""Define the differential equation to be solved by the solver

		Args:
			t (float): time [s]
			x (numpy.ndarray[7]): the following input variables.
				x[0]:   mass [kg]
				x[1:3]: position(x, y, z) in ECI coordinate system [m]
				x[4:6]: velocity(vx, vy, vz) in ECI coordinate system [m/s]

		Returns:
			numpy.ndarray[7]: differentiation result(dm, dx, dy, dz, dvx, dvy, dvz)
		"""
		raise NotImplementedError()

	@abc.abstractclassmethod
	def observer(self, t, x):
		"""Return the result of the calculation
		Args:
			t (float): time [s]
			x (numpy.ndarray[7]): the following input variables.
				x[0]:   mass [kg]
				x[1:3]: position(x, y, z) in ECI coordinate system [m]
				x[4:6]: velocity(vx, vy, vz) in ECI coordinate system [m/s]
		"""
		raise NotImplementedError()


class Rocket(IRocketStatus, IOdeSolver):
	def __init__(self, rocket_conf, id=None):
		self.rf = rocket_conf
		self.id = (id or "")

		#TODO: 何とかしたい
		self.rs = []
		add_one = True
		for key, value in self.rf.items():
			if value["staging"]["separation_time"] <= settings["sim_settings"]["calculate_condition"]["end"]:
				self.rs.append(RocketStage(key, value))
			elif add_one:
				self.rs.append(RocketStage(key, value))
				add_one = False
		#TODO: 何とかしたい
		self.rs[0].state.flight_mode = FlightMode(settings["sim_settings"]["flight_mode"][0])
		self.fo = []
		self.current_num = None
		self.env = Environment()
		self.max_values = {"alt": 0.0, "downrange": 0.0}
		self.log = Logger(self.id, settings["ProjectName"])
		self.target_time = None


	def _get_current_stage(self, index):
		"""Returns the currently calculated RocketStage

		Args:
			index (int): integer counting up

		Returns:
			RocketStage: current stage
		"""
		if index < len(self.rs):
			return self.rs[index]
		elif index - len(self.rs) < len(self.fo):
			return self.fo[index - len(self.rs)]
		else:
			raise AttributeError("Rocket index range error.")


	def __call__(self, t, x):
		""" Perform flight analysis simulation according to the following order:.
				1. Get information about the current stage to be analyzed
				2. Update position
				3. If the altitude is less than 0 (falling to the ground), end the analysis
				4. Calculate surrounding environment information (gravity, atmosphere, wind, etc.) at the current flight position
				5. Update rocket flight status, propulsion system status, etc.
				6. Perform flight analysis
				7. Organize information such as analysis status, events, confirmation of jettison objects, maximum altitude/distance, etc.

		Args:
			t (float): time
			x (numpy.ndarray[7]): the following input variables.
				x[0]:   mass [kg]
				x[1:3]: position(x, y, z) in ECI coordinate system [m]
				x[4:6]: velocity(vx, vy, vz) in ECI coordinate system [m/s]

		Returns:
			dx (numpy.ndarray[7]): the following result variables.
				x[0]:   mass [kg]
				x[1:3]: position(x, y, z) in ECI coordinate system [m]
				x[4:6]: velocity(vx, vy, vz) in ECI coordinate system [m/s]
		"""
		stage = self._get_current_stage(self.current_num)
		stage.update_position(t, x)

		# It does not calculate after the rocket falls to the ground
		if stage.state.LLH.pos[2] < 0:
			if stage.state.flag.is_flying:
				stage.state.flag.is_flying = False
				stage.state.impact_point = stage.state.LLH.pos
			return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

		self.env_info = self.env.get_EnvInfo(altitude=stage.state.LLH.pos[2], latitude=stage.state.LLH.pos[0])
		stage.update_rocket_profile(t, x, self.env_info)
		stage.update_proplusion_profile(t, self.env_info)
		dx = stage.flight_simulation(t, x, self.env_info)
		res = stage.update_status(t, x)
		self.fo = self.fo + res

		self.max_values = {"alt": stage.state.LLH.pos[2] if self.max_values["alt"] < stage.state.LLH.pos[2] else self.max_values["alt"],
						   "downrange": stage.state.downrange if self.max_values["downrange"] < stage.state.downrange else self.max_values["downrange"]}

		return dx


	def observer(self, t, x):
		stage = self._get_current_stage(self.current_num)
		if stage.state.flag.is_flying and not self.log.is_dupulicated(stage.state.name, t) and not settings["sim_settings"]["simulation_target_mode"] == "fast":
			self.log.register(stage.state.name, t, x, stage.state, self.env_info, self.id)
		if stage.state.flag.is_flying and ((self.current_num < len(self.rs) and t < stage.conf["staging"]["separation_time"]) or self.current_num == len(self.rs)-1) and not self.log.is_dupulicated("merged", t):
			self.log.register("merged", t, x, stage.state, self.env_info, self.id)
		# 飛行終了 or (分離済 かつ "fast"モード, ただし最終段の場合を除く) のとき計算を終了する(計算高速化).
		if stage.state.flag.is_flying == False or (stage.state.flag.separated is not None and settings["sim_settings"]["simulation_target_mode"] == "fast" and self.current_num != len(self.rs)-1):
			return ["exit", t, None]
		else:
			# 分離物がある場合, 3つめの引数に現在のロケットの状態を返す
			if stage.state.flag.deployed is not None:
				new_x = copy.deepcopy(stage.state.flag.deployed)
				stage.state.flag.deployed = None
				return ["deployed", t, new_x]
			else:
				return ["continue", t, None]


	def get_status(self, simulation_status:SimulationStatus) -> RocketStatus:
		# Status when initialized flight simulation each rocket stage
		if simulation_status.status == 0:
			self.is_initial_step = True
			self.current_num = simulation_status.num
			if self.current_num < len(self.rs):
				mass = self.rs[self.current_num].conf["mass"][0]
				if self.current_num == 0:
					calc_start_time = settings["sim_settings"]["calculate_condition"]["start"]
					posLLH = settings["sim_settings"]["launch"]["position"]
					posECI = cs.get_pos_ECEF2ECI(posECEF=cs.get_pos_LLH2ECEF(posLLH=posLLH), sec=0.0)
					velNED = settings["sim_settings"]["launch"]["velocity"]
					velECI = cs.vel_ECI_ECIframe(dcmNED2ECI=cs.get_DCM_ECI2NED(dcmECEF2NED=cs.get_DCM_ECEF2NED(posLLH=posLLH), dcmECI2ECEF=cs.get_DCM_ECI2ECEF(sec=0.0)).T,
												 vel_ECEF_NEDframe=velNED, posECI=posECI)
				else:
					calc_start_time = self.rs[self.current_num-1].state.flag.separated[0]
					posECI = self.rs[self.current_num-1].state.flag.separated[1].pos
					velECI = self.rs[self.current_num-1].state.flag.separated[1].vel

			# 搭載ステージの計算が終わった後はflying objectが無いか確認して計算を続行する
			elif self.current_num - len(self.rs) < len(self.fo):
				p = self.fo[self.current_num - len(self.rs)].property
				mass = p["mass"]
				calc_start_time = p["separation_time"]
				posECI = p["separated"].pos
				velECI = p["separated"].vel
			else:
				return RocketStatus(y0=None, calc_start_time=None, is_continue=False)

			return RocketStatus(
				y0 = np.array([mass, posECI[0], posECI[1], posECI[2], velECI[0], velECI[1], velECI[2]]),
				calc_start_time = calc_start_time)

		# check rocket stage is flying when in simulation loop
		elif simulation_status.status == 1:
			stage = self._get_current_stage(self.current_num)
			self.target_time = simulation_status.target_time if simulation_status.target_time is not None else 0
			return RocketStatus(is_flying=stage.state.flag.is_flying)

		# status when each stage simulation end
		elif simulation_status.status == 2:
			stage = self._get_current_stage(self.current_num)
			return RocketStatus(stage_name=stage.state.name, impact_point=stage.state.impact_point)

		# status when simulation finished
		elif simulation_status.status == 3:
			return RocketStatus(max_values=self.max_values, id=self.id)
