
# import modules
import numpy as np

from define import *


class Air:
	""" Define atmospheric models etc.
	"""
	def __init__(self, k=0, temperature=300, air_speed=1200, pressure=101300, density=1.2):
		self.k = k
		self.temperature = temperature
		self.air_speed = air_speed
		self.pressure = pressure
		self.density = density


	def calc(self, altitude):
		""" Calculate various parameters from the current altitude.

		Args:
			altitude (float): altitude [m]

		Raises:
			ValueError: Information corresponding to the current altitude does not exist in the database.
		"""
		HAL = settings["constants"]["planet"]["Earth"]["atomosphere"]["HAL"]
		gamma = settings["constants"]["planet"]["Earth"]["atomosphere"]["gamma"]
		R = settings["constants"]["planet"]["Earth"]["atomosphere"]["R"]
		LR = settings["constants"]["planet"]["Earth"]["atomosphere"]["LR"]
		T0 = settings["constants"]["planet"]["Earth"]["atomosphere"]["T0"]
		P0 = settings["constants"]["planet"]["Earth"]["atomosphere"]["P0"]

		#TODO: 特異点(速度0, 特定姿勢?)でtrue devide errorが出る
		for i in range(len(HAL)):
			if i == len(HAL)-1:
				if HAL[i] <= altitude:
					self.k = i
				else:
					raise ValueError("something was occured in Air.altitude().")
			elif HAL[i] <= altitude and altitude < HAL[i+1]:
				self.k = i
				break

		self.temperature = T0[self.k] + LR[self.k] * (altitude - HAL[self.k])
		self.air_speed = np.sqrt(self.temperature * gamma * R)
		if LR[self.k] != 0:
			self.pressure = P0[self.k] * pow(((T0[self.k] + LR[self.k] * (altitude - HAL[self.k])) / T0[self.k]), (settings["constants"]["planet"]["Earth"]["g0"] / -LR[self.k] / R))
		else:
			self.pressure = P0[self.k] * np.exp(settings["constants"]["planet"]["Earth"]["g0"] / R * (HAL[self.k] - altitude) / T0[self.k])
		self.density = self.pressure / R / self.temperature


	# altitude : [m]
    # percent : [%](-100 ~ 100), enter 0 when nominal air density
    # @output coefficient of density variance : [-1.0 ~ 1.0]
    # cf. U.S. standard atmosphere PART2 Atmospheric Model 2.1.4 Density Variations
	def coef_density_variance(self, altitude, input_percent):
		if input_percent == 0:
			return 0
		else:
			percent_of_density_with_alt = None
			minux_x = np.array([-12.8, -7.9, -1.3, -14.3, -15.9, -18.6, -32.1, -38.6, -50.0, -55.3, -65.0, -68.1, -76.7, -42.2])
			minus_y = np.array([1010, 4300, 8030, 10220, 16360, 20300, 26220, 29950, 40250, 50110, 59970, 70270, 80140, 90220])
			plus_x =  np.array([21.6, 7.4, 1.5, 5.3, 26.7, 20.2, 14.3, 18.2, 33.6, 47.4, 59.5, 72.2, 58.7, 41.4])
			plus_y =  np.array([1230, 4300, 8030, 10000, 16360, 20300, 26220, 29950, 40250, 50110, 59970, 70270, 80360, 90880])
			if input_percent < 0:
				percent_of_density_with_alt = np.interp(altitude, minux_x, minus_y)
			else:
				percent_of_density_with_alt = np.interp(altitude, plus_x, plus_y)
			return percent_of_density_with_alt / 100 * abs(input_percent) / 100


    # Pseudo constructor.
    # Enter altitude and variation ratio air density with altitude to calculate air density
	def calc_with_variation(self, altitude, input_percent):
		self.calc(altitude)
		coef = self.coef_density_variance(altitude, input_percent)
		self.density = self.density * (1.0 + coef)


if __name__ == "__main__":
	air = Air()
	with open("./output/air.csv", "w") as f:
		f.write("altitude(m),temperature(K),air_speed(m/s),density(kg/m3)\n")
		for alt in range(0, 100000, 100):
			air.calc(alt)
			f.write("{0},{1},{2},{3}\n".format(alt, air.temperature, air.air_speed, air.density))


"""
//function [T, a, P, rho] = atmosphere_Rocket( h )
//% ATMOSPHERE_ROCKET 標準大気モデルを用いた、高度による温度、音速、大気圧、空気密度の関数
//% 高度は基準ジオポテンシャル高度を元にしている。
//% 標準大気の各層ごとの気温減率から定義式を用いて計算している。
//% Standard Atmosphere 1976　ISO 2533:1975
//% 中間圏高度86kmまでの気温に対応している。それ以上は国際標準大気に当てはまらないので注意。
//% cf. http://www.pdas.com/hydro.pdf
//% @param h 高度[m]
//% @return T 温度[K]
//% @return a 音速[m/s]
//% @return P 気圧[Pa]
//% @return rho 空気密度[kg/m3]
//% 1:	対流圏		高度0m
//% 2:	対流圏界面	高度11000m
//% 3:	成層圏  		高度20000m
//% 4:	成層圏　 		高度32000m
//% 5:	成層圏界面　	高度47000m
//% 6:	中間圏　 		高度51000m
//% 7:	中間圏　 		高度71000m
//% 8:	中間圏界面　	高度84852m
//
//% ----
//% Future Works:
//% ATOMOSPHERIC and SPACE FLIGHT DYNAMICSより
//% Standard ATOMOSPHEREのスクリプトに変更して高度2000kmまで対応にする。
//% 主に温度上昇と重力加速度とガス状数が変化することに対応すること。
//% ----
"""