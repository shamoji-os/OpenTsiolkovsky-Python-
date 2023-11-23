# シミュレーション設定・定数の定義(Simulation settings/constant definition.)

import sys
from pathlib import Path
from types import MappingProxyType
import copy
import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline

import json5_wrapper


# When a configuration file (json5 format) is given as input,
# its contents are stored in an immutable dictionary type variable (MappingProxyType) and returned(variable name is "settings").
# Furthermore, if you import this code (from define import *), you will be able to access the "settings" variable without declaring it.
try:
	settings = json5_wrapper.load(Path(sys.argv[1]))

	settings["ProjectName"] = settings["ProjectName"] if settings["ProjectName"] is not None else Path(sys.argv[1]).stem
	settings["sim_settings"]["launch"]["attitude"] = np.array(settings["sim_settings"]["launch"]["attitude"]) if settings["sim_settings"]["launch"]["attitude"] is not None else None
	settings["sim_settings"]["launch"]["offset"] = np.array(settings["sim_settings"]["launch"]["offset"])
	settings["sim_settings"]["launch"]["position"] = np.array(settings["sim_settings"]["launch"]["position"])
	settings["sim_settings"]["launch"]["velocity"] = np.array(settings["sim_settings"]["launch"]["velocity"])
	if settings["trajectory"]["mode"] != 2:
		if type(settings["trajectory"]["data"]) == str:
			settings["trajectory"]["data"] = pd.read_csv(settings["trajectory"]["data"], header=0, names=["time", "PITCH", "YAW", "ROLL"])
			if settings["trajectory"]["mode"] == 1:
				settings["sim_settings"]["launch"]["attitude"] = np.array(settings["trajectory"]["data"].iloc[0])[1:] if settings["sim_settings"]["launch"]["attitude"] is None else np.array(settings["sim_settings"]["launch"]["attitude"])
				#settings["sim_settings"]["launch"]["attitude"] += np.array(settings["sim_settings"]["launch"]["offset"])
				d, dt = settings["trajectory"]["data"].diff(), settings["trajectory"]["data"].diff()["time"]
				new_df = pd.DataFrame({"time": settings["trajectory"]["data"]["time"], "PITCH": d["PITCH"]/dt, "YAW": d["YAW"]/dt, "ROLL": d["ROLL"]/dt})
				new_df = new_df.shift(-1, fill_value=0.0) # 姿勢角で指定した場合は最後のプログラムレートは0.0 [deg/s]
				new_df["time"] = settings["trajectory"]["data"]["time"]
				settings["trajectory"]["data"] = new_df
			elif settings["sim_settings"]["launch"]["attitude"] is None:
				raise AttributeError("attitude at launch not set.")
			settings["trajectory"]["data"] = [InterpolatedUnivariateSpline(settings["trajectory"]["data"]["time"], settings["trajectory"]["data"][c], k=1) for c in settings["trajectory"]["data"].columns[1:]]
		elif settings["trajectory"]["mode"] == 1:
			settings["sim_settings"]["launch"]["attitude"] = np.array(settings["trajectory"]["data"]) + settings["sim_settings"]["launch"]["offset"]
			settings["trajectory"]["data"] = 0.0
	else:
		pass #TODO:FSW機能は未実装
	#TODO:6DOF機能は未実装
	#self.CGXt_matrix = pd.read_csv("./" + self.CGXt_file_name, header=0, names=["time", "azi", "ele"]) if self.is_consider_neutrality else None
	#self.Xcp_matrix = pd.read_csv("./" + self.CP_file_name, header=0, names=["time", "CG_pos_STA", "Controller_pos_STA"]) if self.is_consider_neutrality else None


	# Load data that may be in matrix form
	for stage in settings["stage"].values():
		for key, values in list(stage["propulsion"].items()):
			for k, v in values.items():
				if k == "Isp":
					stage["propulsion"][key][k] = pd.read_csv(v, header=0, names=["time", "Isp"]) if type(v) == str else v
				if k == "thrust":
					stage["propulsion"][key][k] = pd.read_csv(v, header=0, names=["time", "thrust", "p"]) if type(v) == str else v
			# add calculating data
			stage["propulsion"][key]["throat_area"] = stage["propulsion"][key]["throat_diameter"]**2 * np.pi / 4.0
			stage["propulsion"][key]["nozzle_exhaust_area"] = stage["propulsion"][key]["throat_area"] * stage["propulsion"][key]["nozzle_expansion_ratio"]
		for k, v in list(stage["structure"].items()):
			if k == "CN":
				stage["structure"][k] = pd.read_csv(v, header=0) if type(v) == str else v
			if k == "CA":
				stage["structure"][k] = pd.read_csv(v, header=0, names=["M", "CA"]) if type(v) == str else v
		# add calculating data
		stage["structure"]["body_area"] = stage["structure"]["diameter"]**2 * np.pi / 4.0


	# TODO:実装予定
	#if self.is_consider_neutrality:
	#	self.pos_CG    = interp1d(self.CGXt_matrix["time"], self.CGXt_matrix["azi"], bounds_error=False, fill_value=(self.CGXt_matrix["azi"].iloc[0], self.CGXt_matrix["azi"].iloc[-1]))(time)
	#	self.pos_Controller = interp1d(self.CGXt_matrix["time"], self.CGXt_matrix["ele"], bounds_error=False, fill_value=(self.CGXt_matrix["azi"].iloc[0], self.CGXt_matrix["azi"].iloc[-1]))(time)
	#if time >= self.later_stage_separation_time and self.is_separated == False:
	#self.gimbal_angle_pitch = np.NaN
	#self.gimbal_angle_yaw = np.NaN

except Exception as e:
	print(e)
	raise KeyError("Configuration file setting is something wrong.")

nominal_staging = copy.deepcopy(settings["stage"])
settings = MappingProxyType(settings) # 変更不可の辞書型オブジェクトを返す
