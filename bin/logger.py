

import sys
import pandas as pd
import datetime
from pathlib import Path

from define import *
from rocket_state import *


class Logger():
	""" A class that manages log information, such as sequentially saving analysis results and outputting results to files.
	"""
	output_dir = None

	oldstyle = ["time", "mass", "thrust", "thrust_vac", "lat", "lon", "alt", "x(ECI)", "y(ECI)", "z(ECI)", "vx(ECI)", "vx(ECI)", "vz(ECI)", "vx(NED)", "vy(NED)", "vz(NED)",
			"ax(ECI)", "ay(ECI)", "az(ECI)", "ax(BODY)", "ay(BODY)", "az(BODY)", "Isp", "Mach", "PITCH", "YAW", "aoa(alpha)", "aoa(beta)", "aoa(gamma)",
			"Q", "airforce_x(BODY)", "airforce_y(BODY)", "airforce_z(BODY)", "thrust_x(BODY)", "thrust_y(BODY)", "thrust_z(BODY)", #"gimbal_angle_pitch", "gimbal_angle_yaw",
			"wind_speed", "wind_direction", "downrange", "latIIP", "lonIIP", "altIIP", "dcmBODY2ECI","inertial_velocity", "kinematic_energy(NED)", "loss_gravity", "loss_aerodynamics", "loss_thrust", "flight_mode", "separated"]

	symbol = {
		# symbol                        : correspondence value (can be extracted from the following regex)
		# find word   : (.*: *)"(.*)",\n
		# replace word: $1$2,\n
		"event"							: """[k for k, v in settings["SOE"].items() if v == t][0] if len([k for k, v in settings["SOE"].items() if v == t]) > 0 else """,
		"time" 							: "t",
		"mass"  						: "x[0]",
		"thrust"    					: "rs.propulsion.thrust",
		"thrust_vac"					: "rs.propulsion.thrust_vac",
		"lat" 							: "rs.LLH.pos[0]",
		"lon"							: "rs.LLH.pos[1]",
		"alt"							: "rs.LLH.pos[2]",
		"x(ECI)"						: "x[1]",
		"y(ECI)"						: "x[2]",
		"z(ECI)"						: "x[3]",
		"vx(ECI)"						: "x[4]",
		"vx(ECI)"						: "x[5]",
		"vz(ECI)"						: "x[6]",
		"ax(ECI)"						: "rs.ECI.acc[0]",
		"ay(ECI)"						: "rs.ECI.acc[1]",
		"az(ECI)"						: "rs.ECI.acc[2]",
		"x(ECEF)"						: "rs.ECEF.pos[0]",
		"y(ECEF)"						: "rs.ECEF.pos[1]",
		"z(ECEF)"						: "rs.ECEF.pos[2]",
		"vx(NED)"						: "rs.NED.vel[0]",
		"vy(NED)"						: "rs.NED.vel[1]",
		"vz(NED)"						: "rs.NED.vel[2]",
		"ax(BODY)"						: "rs.BODY.acc[0]",
		"ay(BODY)"						: "rs.BODY.acc[1]",
		"az(BODY)"						: "rs.BODY.acc[2]",
		"Isp"							: "rs.propulsion.Isp",
		"Isp_vac"						: "rs.propulsion.Isp_vac",
		"Mach"							: "rs.mach",
		"PITCH"							: "rs.NED.attitude[0]",
		"YAW"							: "rs.NED.attitude[1]",
		"ROLL"							: "rs.NED.attitude[2]",
		"aoa(alpha)"					: "np.rad2deg(rs.angle_of_attack[0])",
		"aoa(beta)"						: "np.rad2deg(rs.angle_of_attack[1])",
		"aoa(gamma)"					: "np.rad2deg(rs.angle_of_attack[2])",
		"Q"								: "rs.dynamic_pressure",
		"airforce_x(BODY)"				: "rs.force.airBODY[0]",
		"airforce_y(BODY)"				: "rs.force.airBODY[1]",
		"airforce_z(BODY)"				: "rs.force.airBODY[2]",
		"thrust_x(BODY)"				: "rs.propulsion.force.thrust_vector[0]",
		"thrust_y(BODY)"				: "rs.propulsion.force.thrust_vector[1]",
		"thrust_z(BODY)"				: "rs.propulsion.force.thrust_vector[2]",
		"wind_speed"					: "env_info.wind.speed",
		"wind_direction"				: "env_info.wind.direction",
		"downrange"						: "rs.downrange",
		"latIIP"						: "rs.posLLH_IIP[0]",
		"lonIIP"						: "rs.posLLH_IIP[1]",
		"altIIP"						: "rs.posLLH_IIP[2]",
		"dcmECI2ECEF"					: "rs.dcm.ECI2ECEF",
		"dcmECEF2NED"					: "rs.dcm.ECEF2NED",
		"dcmECI2NED"					: "rs.dcm.ECI2NED",
		"dcmNED2BODY"					: "rs.dcm.NED2BODY",
		"dcmECI2BODY"					: "rs.dcm.ECI2BODY",
		"dcmNED2ECEF"					: "rs.dcm.NED2ECEF",
		"dcmNED2ECI"					: "rs.dcm.NED2ECI",
		"dcmBODY2ECI"					: "rs.dcm.BODY2ECI",
		"dcmECEF2NED_init"				: "rs.dcm.ECEF2NED_init",
		"dcmECI2NED_init"				: "rs.dcm.ECI2NED_init",
		"inertial_velocity"				: "np.linalg.norm(rs.ECI.vel)",
		"kinematic_energy(NED)"			: "rs.kinematic_energy",
		"loss_gravity"					: "rs.loss.gravity",
		"loss_aerodynamics"				: "rs.loss.aerodynamics",
		"loss_thrust"					: "rs.loss.thrust",
		"loss_total"					: "rs.loss.total",
		"flight_mode"					: "rs.flight_mode",
		"separated"						: "rs.flag.separated",
		"is_flying"						: "rs.flag.is_flying",
	}

	def __init__(self, id="nominal", project_name=""):
		self.id = id
		self.df = {}
		self.dictionary = {}
		if Logger.output_dir is None and settings["system"]["output_log"]:
			t_delta = datetime.timedelta(hours=9)
			JST = datetime.timezone(t_delta, 'JST')
			now = datetime.datetime.now(JST)
			Logger.output_dir = "./output/{0}({1})/".format(project_name, now.strftime("%Y%m%d_%H%M%S"))
			p = None if Path(Logger.output_dir).exists() else Path(Logger.output_dir).mkdir(parents=True)
		self.header = ",".join(["time(s)", "mass(kg)", "thrust(N)", "lat(deg)", "lon(deg)", "altitude(m)", "pos_ECI_X(m)", "pos_ECI_Y(m)", "pos_ECI_Z(m)",
				"vel_ECI_X(m/s)", "vel_ECI_Y(m/s)", "vel_ECI_Z(m/s)", "vel_NED_X(m/s)", "vel_NED_Y(m/s)", "vel_NED_Z(m/s)", "acc_ECI_X(m/s2)", "acc_ECI_Y(m/s2)", "acc_ECI_Z(m/s2)",
				"acc_Body_X(m/s)", "acc_Body_Y(m/s)", "acc_Body_Z(m/s)", "Isp(s)", "Mach number", "attitude_azimth(deg)", "attitude_elevation(deg)",
				"attack of angle alpha(deg)", "attack of angle beta(deg)", "all attack of angle gamma(deg)", "dynamic pressure(Pa)", "airforce_Body_X[N]", "airforce_Body_Y[N]", "airforce_Body_Z[N]",
				"thrust_Body_X[N]", "thrust_Body_Y[N]", "thrust_Body_Z[N]", "gimbal_angle_pitch(deg)", "gimbal_angle_yaw(deg)", "wind speed(m/s)", "wind direction(deg)", "downrange(m)",
				"IIP_lat(deg)", "IIP_lon(deg)", "dcmBODY2ECI_11", "dcmBODY2ECI_12", "dcmBODY2ECI_13", "dcmBODY2ECI_21", "dcmBODY2ECI_22", "dcmBODY2ECI_23", "dcmBODY2ECI_31", "dcmBODY2ECI_32", "dcmBODY2ECI_33",
				"inertial velocity(m/s)", "kinematic_energy_NED(J)", "loss_gravity(m/s2)", "loss_aerodynamics(m/s2)", "loss_thrust(m/s2)", "is_powered(1=powered 0=free)", "is_separated(1=already 0=still)", "\n"])


	def save_all(self):
		"""save log file all
		"""
		for id, values in self.dictionary.items():
			self.df[id] = pd.DataFrame().from_dict(values, orient="index")
		for id, values in self.df.items():
			p = Logger.output_dir + "/" + id + "/"
			if not Path(p).exists():
				Path(p).mkdir(parents=True)
			for k, v in values.items():
				with open(p + "{0}.csv".format(k), mode="w", newline="", encoding="utf-8") as f:
					v.to_csv(f, index=False)
		args = sys.argv


	def save(self):
		"""save log file
		"""
		df = {}
		for k, v in self.dictionary[self.id].items():
			df[k] = pd.DataFrame().from_dict(v, orient="index")
		p = Path(Logger.output_dir + "/" + self.id + "/").mkdir(parents=True) if not Path(Logger.output_dir + "/" + self.id + "/").exists() else None
		for k, v in df.items():
			with open(Logger.output_dir + "/" + self.id + "/" +  k + ".csv", mode="w", newline="", encoding="utf-8") as f:
				v.to_csv(f, index=False)


	def is_dupulicated(self, name, t):
		"""Check if information with the same time has already been registered

		Args:
			name (str): data name
			t (float): time [s]

		Returns:
			bool: check result
		"""
		#return True if name in self.df.keys() and t in np.array(self.df[name]["time"]) else False
		return True if name in self.dictionary.keys() and t in np.array(list(self.dictionary[name].keys()), dtype=float) else False


	def register(self, key, t, x, rs:RocketState, env_info, id=None):
		"""Register information in the database

		Args:
			key (str): data key
			t (float): time [s]
			x (numpy.ndarray[7]): parameters of differential equations
			rs (RocketState): rocket status
			env_info (EnvInfo): environmental information
		"""
		if False: #TODO: 全体ログは広めに記録後、出力時に特定symbolに絞って出力する機能の実装
			self.df[key] = pd.DataFrame(columns = list(Logger.symbol.keys()))

		id = str(id) if id is not None else "nominal"

		if not id in self.dictionary.keys():
			self.dictionary[id] = {}
		if not key in self.dictionary[id].keys():
			self.dictionary[id][key] = {}

		# TODO:(暫定)出力を旧形式に合わせている
		self.dictionary[id][key][str(t)] = {
			"event"							: [k for k, v in settings["SOE"].items() if v == t][0] if len([k for k, v in settings["SOE"].items() if v == t]) > 0 else "",
			"time" 							: t,
			"mass"  						: x[0],
			"thrust"    					: rs.propulsion.thrust,
			#"thrust_vac"					: rs.propulsion.thrust_vac,
			"lat" 							: rs.LLH.pos[0],
			"lon"							: rs.LLH.pos[1],
			"alt"							: rs.LLH.pos[2],
			"x(ECI)"						: x[1],
			"y(ECI)"						: x[2],
			"z(ECI)"						: x[3],
			"vx(ECI)"						: x[4],
			"vy(ECI)"						: x[5],
			"vz(ECI)"						: x[6],
			#"x(ECEF)"						: rs.ECEF.pos[0],
			#"y(ECEF)"						: rs.ECEF.pos[1],
			#"z(ECEF)"						: rs.ECEF.pos[2],
			"vx(NED)"						: rs.NED.vel[0],
			"vy(NED)"						: rs.NED.vel[1],
			"vz(NED)"						: rs.NED.vel[2],
			"ax(ECI)"						: rs.ECI.acc[0],
			"ay(ECI)"						: rs.ECI.acc[1],
			"az(ECI)"						: rs.ECI.acc[2],
			"ax(BODY)"						: rs.BODY.acc[0],
			"ay(BODY)"						: rs.BODY.acc[1],
			"az(BODY)"						: rs.BODY.acc[2],
			"Isp"							: rs.propulsion.Isp,
			#"Isp_vac"						: rs.propulsion.Isp_vac,
			"Mach"							: rs.mach,
			"YAW"							: rs.NED.attitude[1],
			"PITCH"							: rs.NED.attitude[0],
			#"ROLL"							: rs.NED.attitude[2],
			"aoa(alpha)"					: np.rad2deg(rs.angle_of_attack[0]),
			"aoa(beta)"						: np.rad2deg(rs.angle_of_attack[1]),
			"aoa(gamma)"					: np.rad2deg(rs.angle_of_attack[2]),
			"Q"								: rs.dynamic_pressure,
			"airforce_x(BODY)"				: rs.force.airBODY[0],
			"airforce_y(BODY)"				: rs.force.airBODY[1],
			"airforce_z(BODY)"				: rs.force.airBODY[2],
			"thrust_x(BODY)"				: rs.force.thrust_vector[0],
			"thrust_y(BODY)"				: rs.force.thrust_vector[1],
			"thrust_z(BODY)"				: rs.force.thrust_vector[2],
			"gimbal_pitch"					: np.NaN,
			"gimbal_yaw"					: np.NaN,
			"wind_speed"					: env_info.wind.speed,
			"wind_direction"				: env_info.wind.direction,
			"downrange"						: rs.downrange,
			"latIIP"						: rs.posLLH_IIP[0],
			"lonIIP"						: rs.posLLH_IIP[1],
			#"altIIP"						: rs.posLLH_IIP[2],
			#"dcmECI2ECEF"					: rs.dcm.ECI2ECEF,
			#"dcmECEF2NED"					: rs.dcm.ECEF2NED,
			#"dcmECI2NED"					: rs.dcm.ECI2NED,
			#"dcmNED2BODY"					: rs.dcm.NED2BODY,
			#"dcmECI2BODY"					: rs.dcm.ECI2BODY,
			#"dcmNED2ECEF"					: rs.dcm.NED2ECEF,
			#"dcmNED2ECI"					: rs.dcm.NED2ECI,
			"dcmBODY2ECI_11"				: rs.dcm.BODY2ECI[0, 0],
			"dcmBODY2ECI_12"				: rs.dcm.BODY2ECI[0, 1],
			"dcmBODY2ECI_13"				: rs.dcm.BODY2ECI[0, 2],
			"dcmBODY2ECI_21"				: rs.dcm.BODY2ECI[1, 0],
			"dcmBODY2ECI_22"				: rs.dcm.BODY2ECI[1, 1],
			"dcmBODY2ECI_23"				: rs.dcm.BODY2ECI[1, 2],
			"dcmBODY2ECI_31"				: rs.dcm.BODY2ECI[2, 0],
			"dcmBODY2ECI_32"				: rs.dcm.BODY2ECI[2, 1],
			"dcmBODY2ECI_33"				: rs.dcm.BODY2ECI[2, 2],
			#"dcmECEF2NED_init"				: rs.dcm.ECEF2NED_init,
			#"dcmECI2NED_init"				: rs.dcm.ECI2NED_init,
			"inertial_velocity"				: np.linalg.norm(rs.ECI.vel),
			"kinematic_energy(NED)"			: rs.kinematic_energy,
			"loss_gravity"					: rs.loss.gravity,
			"loss_aerodynamics"				: rs.loss.aerodynamics,
			"loss_thrust"					: rs.loss.thrust,
			#"loss_total"					: rs.loss.total,
			"flight_mode"					: rs.flight_mode,
			"separated"						: rs.flag.separated,
			#"is_flying"					: rs.flag.is_flying,
		}