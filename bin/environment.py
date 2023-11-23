#　Manage environmental information (gravity, atmosphere, wind, etc.)

import abc
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from define import *
import air
import gravity


class EnvInfo():
	def __init__(self, g, air_pressure, air_speed, air_density, wind_speed, wind_direction):
		"""Environmental information.

		Args:
			g (numpy.ndarray[3]): gravity vector [m/s^2]
			air_pressure (float): air pressure [Pa]
			air_speed (float): air speed [m/s]
			air_density (float): air density []
			wind_speed (float): wind speed [m/s]
			wind_direction (float): wind direction [deg]
		"""
		self.g = g
		self.air = type("AirInfo", (object,), {})
		self.air.pressure = air_pressure
		self.air.speed = air_speed
		self.air.density = air_density
		self.wind = type("WindInfo", (object,), {})
		self.wind.speed = wind_speed
		self.wind.direction = wind_direction


class IEnvironmentalInformation(metaclass=abc.ABCMeta):
	@abc.abstractclassmethod
	def get_EnvInfo(self, altitude, latitude) -> EnvInfo:
		raise NotImplementedError()


class Environment(IEnvironmentalInformation):
	def __init__(self) -> None:
		super().__init__()
		self.air = air.Air()

		# Load wind data
		if type(settings["env_params"]["wind_data"]) == str:
			settings["env_params"]["wind_data"] = pd.read_csv(settings["env_params"]["wind_data"], header=0, names=["alt", "speed", "dir"])
			settings["env_params"]["wind_data"].assign(u = - settings["env_params"]["wind_data"]["speed"] * np.sin(np.deg2rad(settings["env_params"]["wind_data"]["dir"])))
			settings["env_params"]["wind_data"] = settings["env_params"]["wind_data"].assign(v = - settings["env_params"]["wind_data"]["speed"] * np.cos(np.deg2rad(settings["env_params"]["wind_data"]["dir"])))
		else:
			settings["env_params"]["wind_data"] = np.array(settings["env_params"]["wind_data"])


	def get_EnvInfo(self, altitude, latitude):
		"""Get environmental information.

		Args:
			altitude (float): altitude [m]
			latitude (float): latitude [deg]

		Returns:
			EnvInfo: Environmental information. (EnvInfo -> g, air_pressure, air_speed, air_density, wind_speed, wind_direction)
		"""
		g = np.array([0.0, 0.0, - gravity.gravity(altitude, latitude)])
		self.air.calc_with_variation(altitude, settings["sim_settings"]["calculate_condition"]["atomospheric_dispersion"])

		if isinstance(settings["env_params"]["wind_data"], pd.DataFrame):
			wind_u = interp1d(settings["env_params"]["wind_data"]["alt"], settings["env_params"]["wind_data"]["u"])(altitude)
			wind_v = interp1d(settings["env_params"]["wind_data"]["alt"], settings["env_params"]["wind_data"]["v"])(altitude)
			wind_speed = np.sqrt(wind_u**2 + wind_v**2)
			wind_direction = np.rad2deg(np.arctan2(wind_u, wind_v)) + 180.0
		else:
			wind_speed, wind_direction = settings["env_params"]["wind_data"]

		return EnvInfo(g, self.air.pressure, self.air.air_speed, self.air.density, wind_speed, wind_direction)


if __name__ == "__main__":
	env = Environment()

	def test():
		with open("./output/env.csv", "w") as f:
			f.write("altitude(km),latitude(deg),gravity(m/s2),air_pressure[Pa],air_speed[m/s],air_density\n")
			for lat in range(0, 91, 30):
				for alt in range(0, 1000001, 1000):
					info = env.get_EnvInfo(alt, lat)
					print(info.values())
					f.write("{0},{1},{2},{3},{4},{5}\n".format(alt/1000, lat, info.g[2], info.air.pressure, info.air_speed, info.air_density))
	test()
