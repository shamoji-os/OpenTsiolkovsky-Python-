
# import modules
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import copy

from define import *
import coordinate_system as cs
from rocket_state import *
from calculation import *


class RocketStage():
	def __init__(self, name, conf):
		self.conf = conf
		self.state = RocketState(name)
		self.previous_time = -settings["constants"]["math"]["infinity"]
		if len(self.conf["mass"]) == 1:
			self.state.flight_mode = FlightMode("ballistic")


	def update_position(self, t, x):
		# update attitude
		if self.state.flight_mode != FlightMode("inertial_object"):
			# TODO:多分const値のときに動かない
			# TODO:trajectory mode 1 のとき、一旦姿勢→prateに変換して処理していることとlaunch attitudeからの積算処理をしている関係上、設定ファイルと入力ファイルの初期姿勢がずれているとオフセット誤差のような形で効いてしまう
			self.state.NED.attitude = settings["sim_settings"]["launch"]["attitude"] + settings["sim_settings"]["launch"]["offset"] + integrate_program_rate(settings["trajectory"]["data"], t)
			"""
			if settings["trajectory"]["mode"] == 0:
				self.state.NED.attitude = settings["sim_settings"]["launch"]["attitude"] + settings["sim_settings"]["launch"]["offset"] + integrate_program_rate(settings["trajectory"]["data"], t)
			elif settings["trajectory"]["mode"] == 1:
				self.state.NED.attitude = settings["sim_settings"]["launch"]["offset"] + np.array([settings["trajectory"]["data"][i](t) for i in range(3)])
			"""
		# update position
		self.state.ECI.pos  = np.array([x[1], x[2], x[3]])
		self.state.dcm.ECI2ECEF = cs.get_DCM_ECI2ECEF(t)
		self.state.ECEF.pos = cs.get_pos_ECI2ECEF(posECI=self.state.ECI.pos, dcmECI2ECEF=self.state.dcm.ECI2ECEF)
		self.state.LLH.pos  = cs.get_pos_ECEF2LLH(posECEF=self.state.ECEF.pos)

		# prepare the dcm
		self.state.dcm.ECEF2NED = cs.get_DCM_ECEF2NED(self.state.LLH.pos)
		self.state.dcm.ECI2NED = cs.get_DCM_ECI2NED(dcmECEF2NED=self.state.dcm.ECEF2NED, dcmECI2ECEF=self.state.dcm.ECI2ECEF)
		self.state.dcm.NED2BODY = cs.get_DCM_NED2BODY(azi=np.deg2rad(self.state.NED.attitude[1]), ele=np.deg2rad(self.state.NED.attitude[0]))
		self.state.dcm.ECI2BODY = cs.get_DCM_ECI2BODY(dcmNED2BODY=self.state.dcm.NED2BODY, dcmECI2NED=self.state.dcm.ECI2NED)
		self.state.dcm.NED2ECEF = self.state.dcm.ECEF2NED.T
		self.state.dcm.NED2ECI = self.state.dcm.ECI2NED.T
		self.state.dcm.BODY2ECI = self.state.dcm.ECI2BODY.T
		self.state.dcm.ECEF2NED_init = cs.get_DCM_ECEF2NED(settings["sim_settings"]["launch"]["position"])
		self.state.dcm.ECI2NED_init = cs.get_DCM_ECI2NED(dcmECEF2NED=self.state.dcm.ECEF2NED_init, dcmECI2ECEF=cs.get_DCM_ECI2ECEF(0.0))


	def update_rocket_profile(self, t, x, env_info):
		# update velocity
		self.state.ECI.vel = np.array([x[4], x[5], x[6]])
		self.state.NED.vel = cs.vel_ECEF_NEDframe(dcmECI2NED=self.state.dcm.ECI2NED, vel_ECI_ECIframe=self.state.ECI.vel, posECI=self.state.ECI.pos)
		env_info.wind.velNED = cs.vel_wind_NEDframe(wind_speed=env_info.wind.speed, wind_direction=env_info.wind.direction)
		self.state.AIR.velNED = self.state.NED.vel - env_info.wind.velNED # 対気相対速度(NED座標系)
		self.state.AIR.velBODY = self.state.dcm.NED2BODY @ (self.state.NED.vel - env_info.wind.velNED)   # 対気相対速度(BODY座標系)
		self.state.angle_of_attack = cs.aoa(vel_AIR_BODYframe=self.state.AIR.velBODY)

		if self.state.flight_mode == FlightMode("ballistic") or self.state.flight_mode == FlightMode("inertial_object"):
			self.state.mach = np.linalg.norm(self.state.AIR.velNED) / env_info.air.speed
			self.state.dynamic_pressure = 0.5 * env_info.air.density * np.linalg.norm(self.state.AIR.velNED)**2
		else:
			self.state.mach = np.linalg.norm(self.state.AIR.velBODY) / env_info.air.speed
			self.state.dynamic_pressure = 0.5 * env_info.air.density * np.linalg.norm(self.state.AIR.velBODY)**2

		# update air coefficient
		s = self.state.structure
		#TODO: CA(coefficient axis), Cy(coeffficient yaw), Cz(coefficient pitch)の各入力に対応させる
		s.CA = interp1d(self.conf["structure"]["CA"]["M"], self.conf["structure"]["CA"]["CA"], bounds_error=False, fill_value=(self.conf["structure"]["CA"]["CA"].iloc[0], self.conf["structure"]["CA"]["CA"].iloc[-1]))(self.state.mach) if isinstance(self.conf["structure"]["CA"], pd.DataFrame) else self.conf["structure"]["CA"]
		if isinstance(self.conf["structure"]["CN"], pd.DataFrame):
			s.alpha = np.rad2deg(self.state.angle_of_attack[0])
			s.angle_abs_pitch = np.abs(s.alpha)
			s.angle_sign_pitch = 0.0 if s.angle_abs_pitch < 1e-9 else s.alpha / s.angle_abs_pitch
			s.CN_pitch = s.angle_sign_pitch * interp_matrix2d(self.state.mach, s.angle_abs_pitch, np.array(self.conf["structure"]["CN"]), bounds_error=False)
			#TODO:6DOF機能
			#if self.is_consider_neutrality:
			#	self.pos_CP_pitch = interp_matrix2d(self.mach_number, self.angle_abs_pitch, np.array(self.Xcp_matrix))
			s.beta = np.rad2deg(self.state.angle_of_attack[1])
			s.angle_abs_yaw = np.abs(s.beta)
			s.angle_sign_yaw = 0.0 if s.angle_abs_yaw < 1e-9 else s.beta / s.angle_abs_yaw
			s.CN_yaw = s.angle_sign_yaw * interp_matrix2d(self.state.mach, s.angle_abs_yaw, np.array(self.conf["structure"]["CN"]), bounds_error=False)
			#TODO:6DOF機能
			#if self.is_consider_neutrality:
			#	self.pos_CP_yaw = interp_matrix2d(self.mach_number, self.angle_abs_yaw, np.array(self.Xcp_matrix))
		else:
			s.CN_pitch = self.conf["structure"]["CN"]
			s.CN_yaw = self.conf["structure"]["CN"]
			#TODO:6DOF機能
			#if self.is_consider_neutrality:
			#	raise AttributeError("ERROR: you must specify the non-constant CN when you use neutrality calculation.\n")


	def update_proplusion_profile(self, t, env_info):
		#TODO: 複数推進系搭載バージョンに対応していない, 再着火機能未対応
		s = self.state.propulsion
		p = self.conf["propulsion"]["motor"]
		Isp_matrix = p["Isp"]
		thrust_matrix = p["thrust"]

		# Update propulsion characteristics(thrust, Isp, pressure, etc...) to the current time situation TODO:coef機能は分岐も複雑になりかつ使用予定もなかったためオミットしている
		s.Isp_vac                = interp1d(Isp_matrix["time"],    Isp_matrix["Isp"],       bounds_error=False, fill_value=(0, 0))((t-p["Ignition"])*p["thrust_coef"]) * p["Isp_coef"]                    if isinstance(Isp_matrix, pd.DataFrame) else Isp_matrix
		s.thrust_vac             = interp1d(thrust_matrix["time"], thrust_matrix["thrust"], bounds_error=False, fill_value=(0, 0))((t-p["Ignition"])*p["thrust_coef"]) * p["Isp_coef"] * p["thrust_coef"] if isinstance(thrust_matrix, pd.DataFrame) else thrust_matrix
		s.nozzle_exaust_pressure = interp1d(thrust_matrix["time"], thrust_matrix["p"],      bounds_error=False, fill_value=(0, 0))((t-p["Ignition"])*p["thrust_coef"]) * p["Isp_coef"] * p["thrust_coef"] if isinstance(thrust_matrix, pd.DataFrame) else thrust_matrix
		self.state.flight_mode = FlightMode(settings["sim_settings"]["flight_mode"][0]) if self.conf["propulsion"]["motor"]["Ignition"] <= t and t <= self.conf["propulsion"]["motor"]["Burnout"] and self.state.propulsion.thrust_vac > 0 else FlightMode(settings["sim_settings"]["flight_mode"][1])
		if self.state.flight_mode == FlightMode(settings["sim_settings"]["flight_mode"][0]):
			self.state.m_dot = s.thrust_vac / s.Isp_vac / settings["constants"]["planet"]["Earth"]["g0"] # calculate mass flow rate
			s.thrust_momentum = s.thrust_vac - p["nozzle_exhaust_area"] * p["nozzle_exhaust_pressure"]
			s.thrust = s.thrust_vac - p["nozzle_exhaust_area"] * env_info.air.pressure
			s.Isp = s.thrust / self.state.m_dot / settings["constants"]["planet"]["Earth"]["g0"] if self.state.m_dot > 0.0001 else 0
		else:
			s.thrust, self.state.m_dot, s.Isp = 0.0, 0.0, 0.0


	def flight_simulation(self, t, x, env_info):
		"""Conduct flight analysis

		Args:
		Args:
			t (float): time [s]
			x (numpy.ndarray[7]): the following input variables.
				x[0]:   mass [kg]
				x[1:3]: position(x, y, z) in ECI coordinate system [m]
				x[4:6]: velocity(vx, vy, vz) in ECI coordinate system [m/s]
			env_info (EnvInfo): environmental information

		Returns:
			numpy.ndarray[7]: dx
		"""
		if self.state.flight_mode == FlightMode("3DOF"):
			self.state.force.axial = self.state.structure.CA * self.state.dynamic_pressure * self.conf["structure"]["body_area"]
			self.state.force.normal_yaw = self.state.structure.CN_yaw * self.state.dynamic_pressure * self.conf["structure"]["body_area"]
			self.state.force.normal_pitch = self.state.structure.CN_pitch * self.state.dynamic_pressure * self.conf["structure"]["body_area"]
			self.state.force.airBODY = np.array([-self.state.force.axial, -self.state.force.normal_yaw, -self.state.force.normal_pitch])
			# TODO:6DOF機能
			#if self.is_consider_neutrality:
			#	self.sin_of_gimbal_angle_yaw  = self.force_air_vector_BODYframe[1] / self.thrust * (self.pos_CP_yaw - self.pos_CG) / (self.pos_Controller - self.pos_CG)
			#	self.sin_of_gimbal_angle_pitch = self.force_air_vector_BODYframe[2] / self.thrust * (self.pos_CP_pitch - self.pos_CG) / (self.pos_Controller - self.pos_CG)
			#if self.is_consider_neutrality and self.sin_of_gimbal_angle_pitch < 1 \
			#and self.sin_of_gimbal_angle_pitch > -1 and self.sin_of_gimbal_angle_yaw < 1 and self.sin_of_gimbal_angle_yaw > -1:
			#	self.gimbal_angle_pitch = np.arcsin(self.sin_of_gimbal_angle_pitch)
			#	self.gimbal_angle_yaw   = np.arcsin(self.sin_of_gimbal_angle_yaw)
			#	self.force_thrust_vector = np.array([self.thrust * np.cos(self.gimbal_angle_yaw) * np.cos(self.gimbal_angle_pitch),
			#										-self.thrust * np.sin(self.gimbal_angle_yaw),
			#										-self.thrust * np.cos(self.gimbal_angle_yaw) * np.sin(self.gimbal_angle_pitch)])
			self.state.force.thrust_vector = np.array([self.state.propulsion.thrust, 0.0, 0.0]) # body coordinate
			# dv/dt
			self.state.ECI.acc = 1/x[0] * (self.state.dcm.BODY2ECI @ (self.state.force.thrust_vector + self.state.force.airBODY)) + self.state.dcm.NED2ECI @ env_info.g

		elif self.state.flight_mode == FlightMode("6DOF"):
			pass

		elif self.state.flight_mode == FlightMode("ballistic") or self.state.flight_mode == FlightMode("inertial_object"):
			self.state.force.axial = self.state.dynamic_pressure / self.conf["structure"]["BC"]
			self.state.force.airNED = self.state.force.axial * (-1) * (self.state.AIR.velNED / np.linalg.norm(self.state.AIR.velNED))
			self.state.force.normal_yaw = self.state.force.normal_pitch = 0.0
			self.state.force.airBODY = np.array([-self.state.force.axial, 0.0, 0.0]) # TODO:(NEDじゃない?)
			self.state.force.thrust_vector = np.array([0.0, 0.0, 0.0])
			# dv/dt
			self.state.ECI.acc = self.state.dcm.NED2ECI @ (self.state.force.airNED + env_info.g)

		else:
			raise AttributeError("Undefined FlightMode.")

		self.state.BODY.acc = self.state.dcm.ECI2BODY @ (self.state.ECI.acc - self.state.dcm.NED2ECI @ env_info.g)
		self.state.posLLH_IIP = cs.get_IIP(t=t, vel_ECEF_NEDframe=self.state.NED.vel, posECI=self.state.ECI.pos)
		self.state.kinematic_energy = 0.5 * x[0] * np.linalg.norm(self.state.NED.vel) ** 2
		self.state.downrange = cs.distance_surface(settings["sim_settings"]["launch"]["position"], self.state.LLH.pos)

		# calculate loss velocity
		if np.linalg.norm(self.state.force.thrust_vector) > 0.1 or self.state.flag.separated is None:
			v_xy = np.sqrt(self.state.NED.vel[0]**2 + self.state.NED.vel[1]**2)
			path_angle_rad = np.arctan2(-self.state.NED.vel[2], v_xy)
			self.state.loss.gravity = env_info.g[2] * np.sin(path_angle_rad)
		else:
			self.state.loss.gravity = 0
		#TODO: "motor"がハードコーディングに依存している
		self.state.loss.thrust = env_info.air.pressure * self.conf["propulsion"]["motor"]["nozzle_exhaust_area"] / x[0] if np.linalg.norm(self.state.force.thrust_vector) > 0.1 else 0
		self.state.loss.aerodynamics = self.state.force.axial / x[0]
		self.state.loss.total = self.state.loss.gravity + self.state.loss.aerodynamics + self.state.loss.thrust

		self.state.dx = np.array([-self.state.m_dot, x[4], x[5], x[6], self.state.ECI.acc[0], self.state.ECI.acc[1], self.state.ECI.acc[2]])

		return self.state.dx


	def update_status(self, t, x):
		"""Update rocket flight status, event progress, etc.

		Args:
			t (float): time [s]

		Returns:
			list: flying object if there is isolate
		"""
		res = []

		if t >= self.conf["staging"]["separation_time"] and self.state.flag.separated is None:
			self.state.flag.separated = [t, copy.deepcopy(self.state.ECI)]

		for k, v in self.conf["staging"]["deploy_object"].items():
			if t == v["separation_time"]:
				if not "separated" in v:
					s = copy.deepcopy(self.state.ECI)
					s.vel = s.vel + self.state.dcm.NED2ECI @ np.array(v["additional_velocity"])
					v["separated"] = s
					v["name"] = k
					v["mass"] = self.conf["mass"][1] if v["mass"] is None else v["mass"]
					if t < self.conf["staging"]["separation_time"]:
						self.state.flag.deployed = np.array([
							x[0] - self.state.m_dot - v["mass"],
							self.state.ECI.pos[0],
							self.state.ECI.pos[1],
							self.state.ECI.pos[2],
							self.state.ECI.vel[0],
							self.state.ECI.vel[1],
							self.state.ECI.vel[2]
						])
					if self.conf["staging"]["calc_deploy_object"] and settings["sim_settings"]["simulation_target_mode"] != "fast":
						res.append(construct_flying_object(self, k, v))

		return res


def construct_flying_object(rs, name, property):
	""" Constructor for making an instance of ballistic flight object that is a dumping product
		Create a new RocketStage instance from the original RocketStage and
		the position and speed at the time of dumping.

	Args:
		rs (RocketStage): RocketStage Object.
		posECI_init_args (numpy.ndarray[3]): initial position in ECI coordinate system.
		velECI_init_args (numpy.ndarray[3]): initial velocity in ECI coordinate system.

	Returns:
		RocketStage: an instance of ballistic flight object that is a dumping product.
	"""
	fo = copy.deepcopy(rs)
	fo.conf = copy.deepcopy(rs.conf)
	fo.property = property
	fo.state.name = name
	fo.state.flight_mode = FlightMode("inertial_object")

	#fo.log_name = "{0}_dynamics_{1}_dump.csv".format(settings["ProjectName"], property["name"])
	fo.conf["mass"] = property["mass"]
	fo.conf["propulsion"]["motor"]["thrust"] = 0.0
	fo.conf["propulsion"]["motor"]["Isp"] = 0.0
	fo.conf["propulsion"]["motor"]["Ignition"] = 0.0
	fo.conf["propulsion"]["motor"]["Burnout"] = 0.0
	fo.conf["structure"]["BC"] = property["BC"]
	fo.conf["staging"]["deploy_object"] = {} # TODO:強引な解決法になっている

	return fo
