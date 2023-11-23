
# import modules
import numpy as np

from define import *


def gravity(altitude, latitude):
	""" Calculate gravitational acceleration.

	Args:
		altitude (numpy.float64): altitude [m] in LLH(緯度経度高度座標系)
		latitude (numpy.float64): latitude [deg] in LLH(緯度経度高度座標系)

	Returns:
		numpy.float64: g [m/s2]
	"""
	r = altitude + settings["constants"]["planet"]["Earth"]["Re"] * (1 - settings["constants"]["planet"]["Earth"]["f"] * np.sin(latitude)**2)
	gc = -settings["constants"]["planet"]["Earth"]["mu"] / r / r * (1 - 1.5 * settings["constants"]["planet"]["Earth"]["J2"] * (settings["constants"]["planet"]["Earth"]["Re"]/r)**2 * (3*np.sin(latitude)**2 - 1))
	return gc


if __name__ == "__main__":
	def test_gravity():
		with open("./output/gravity.csv", "w") as f:
			f.write("altitude(km),latitude(deg),gravity(m/s2)\n")
			for lat in range(0, 91, 30):
				for alt in range(0, 1000001, 1000):
					g = gravity(alt, lat)
					print(g)
					f.write("{0},{1},{2}\n".format(alt/1000, lat, g))
	test_gravity()
