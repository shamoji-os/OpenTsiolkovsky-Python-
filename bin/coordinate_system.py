# Define a coordinate system

# import modules
import numpy as np
from enum import Enum

from define import *


class CoordinateList(Enum):
	"""座標系のリスト(Enum型)
		ECI = "I:ECI(地球中心慣性座標系)"
		ECEF = "E:ECEF(地球中心回転座標系)"
		LLH = "L:LLH(緯度経度高度座標系)"
		NED = "H:NED(局所平面座標系)"
		AIR = "A:AIR(速度座標系)"
		BODY = "B:BODY(機体座標系)"
	"""
	ECI = "I:ECI(地球中心慣性座標系)"
	ECEF = "E:ECEF(地球中心回転座標系)"
	LLH = "L:LLH(緯度経度高度座標系)"
	NED = "H:NED(局所平面座標系)"
	AIR = "A:AIR(速度座標系)"
	BODY = "B:BODY(機体座標系)"


def get_DCM_ECI2ECEF(sec):
	"""ECI(地球中心慣性座標系)からECEF(地球中心回転座標系)への変換行列(DCM)を取得する

	Args:
		sec (double): time [s]

	Returns:
		numpy.ndarray[3]: DCM from ECI to ECEF
	"""
	theta = settings["constants"]["planet"]["Earth"]["omega"] * sec
	return np.array([[ np.cos(theta), np.sin(theta), 0.0],
					 [-np.sin(theta), np.cos(theta), 0.0],
					 [           0.0,           0.0, 1.0]])


def get_DCM_ECEF2NED(posLLH):
	"""ECEF(地球中心回転座標系)からNED(局所平面座標系)への変換行列(DCM)を取得する

	Args:
		posLLH (numpy.ndarray[3]): position(lat, lon, alt) in LLH coordinate system

	Returns:
		numpy.ndarray[3]: DCM from ECEF to NED
	"""
	lat = np.deg2rad(posLLH[0])
	lon = np.deg2rad(posLLH[1])
	return np.array([[-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon),  np.cos(lat)],
					 [            -np.sin(lon),              np.cos(lon),          0.0],
					 [-np.cos(lat)*np.cos(lon), -np.cos(lat)*np.sin(lon), -np.sin(lat)]])


def get_DCM_ECI2NED(dcmECEF2NED=None, dcmECI2ECEF=None, posLLH=None, sec=None):
	"""ECI(地球中心慣性座標系)からNED(局所平面座標系)への変換行列(DCM)を取得する

	Args:
		dcmECEF2NED (numpy_ndarray[3], optional): DCM from ECEF to NED. Defaults to None but either this or posLLH is required.
		dcmECI2ECEF (numpy_ndarray[3], optional): DCM from ECI to ECEF. Defaults to None but either this or sec is required.
		posLLH (numpy_ndarray[3], optional): position(x, y, z) in LLH coordinate system. Defaults to None but either this or dcmECEF2NED is required.
		sec (float, optional): time [s]. Defaults to None but either this or dcmECI2ECEF is required.

	Raises:
		ValueError: return an error if there are not enough arguments.

	Returns:
		numpy.ndarray[3]: DCM from ECI to NED
	"""
	if dcmECI2ECEF is None:
		if sec is not None:
			dcmECI2ECEF = get_DCM_ECI2ECEF(sec)
		else:
			raise ValueError("Please set value -> dcmECI2ECEF or sec")
	if dcmECEF2NED is None:
		if posLLH is not None:
			dcmECEF2NED = get_DCM_ECEF2NED(posLLH)
		else:
			raise ValueError("Please set value -> dcmECEF2NED or posLLH")
	return dcmECEF2NED @ dcmECI2ECEF


def get_DCM_NED2BODY(azi, ele):
	"""NED(局所平面座標系)からBODY(機体座標系)への変換行列(DCM)を取得する

	Args:
		azi (float): azimuth [rad]
		ele (float): elevation [rad]

	Returns:
		numpy.ndarray[3]: DCM from NED to BODY
	"""
	return np.array([[ np.cos(ele)*np.cos(azi), np.cos(ele)*np.sin(azi), -np.sin(ele)],
					 [            -np.sin(azi),             np.cos(azi),          0.0],
					 [ np.sin(ele)*np.cos(azi), np.sin(ele)*np.sin(azi),  np.cos(ele)]])


def get_DCM_ECI2BODY(dcmNED2BODY=None, dcmECI2NED=None, azi=None, ele=None, dcmECEF2NED=None, dcmECI2ECEF=None, posLLH=None, sec=None):
	"""ECI(地球中心慣性座標系)からBODY(機体座標系)への変換行列(DCM)を取得する

	Args:
		dcmNED2BODY (numpy_ndarray[3], optional): DCM from NED to BODY. Defaults to None but either this or azi, ele is required.
		dcmECI2NED (numpy_ndarray[3], optional): DCM from ECI to NED. Defaults to None but either this or dcmECEF2NED, dcmECI2ECEF is required.
		azi (float, optional): azimuth [rad]. Defaults to None.
		ele (float, optional): elevation [rad]. Defaults to None.
		dcmECEF2NED (numpy_ndarray[3], optional): DCM from ECEF to NED. Defaults to None but either this or posLLH is required.
		dcmECI2ECEF (numpy_ndarray[3], optional): DCM from ECI to ECEF. Defaults to None but either this or sec is required.
		posLLH (numpy_ndarray[3], optional): position(x, y, z) in LLH coordinate system. Defaults to None but either this or dcmECEF2NED is required.
		sec (float, optional): time [s]. Defaults to None but either this or dcmECI2ECEF is required.


	Raises:
		ValueError: return an error if there are not enough arguments.

	Returns:
		numpy.ndarray[3]: DCM from ECI to BODY
	"""
	if dcmNED2BODY is None:
		if azi is not None and ele is not None:
			dcmNED2BODY = get_DCM_NED2BODY(azi, ele)
		else:
			raise ValueError("Please set value -> dcmNED2BODY or (azi, ele)")
	if dcmECI2NED is None:
		dcmECI2NED = get_DCM_ECI2NED(dcmECI2ECEF=dcmECI2ECEF, dcmECEF2NED=dcmECEF2NED, sec=sec, posLLH=posLLH)
	return dcmNED2BODY @ dcmECI2NED


def get_pos_ECI2ECEF(posECI, dcmECI2ECEF=None, sec=None):
	"""位置の座標変換を行う(from ECI(地球中心慣性座標系) to ECEF(機体座標系))

	Args:
		posECI (numpy.ndarray[3]): position(x, y, z) in ECI coordinate system.
		dcmECI2ECEF (numpy_ndarray[3], optional): DCM from ECI to ECEF. Defaults to None but either this or sec is required.
		sec (float, optional): time [s]. Defaults to None but either this or dcmECI2ECEF is required.

	Raises:
		ValueError: return an error if there are not enough arguments.

	Returns:
		numpy.ndarray[3]: position(x, y, z) from ECI to ECEF
	"""
	if dcmECI2ECEF is None and sec is None:
		raise ValueError("Please set one or the other -> dcmECI2ECEF, sec")
	return dcmECI2ECEF @ posECI if dcmECI2ECEF is not None else get_DCM_ECI2ECEF(sec) @ posECI


def get_pos_ECEF2ECI(posECEF, sec):
	"""位置の座標変換を行う(from ECEF(機体座標系) to ECI(地球中心慣性座標系))

	Args:
		posECEF (numpy.ndarray[3]): position(x, y, z) in ECEF coordinate system.
		sec (float): time [s]

	Returns:
		numpy.ndarray[3]: position(x, y, z) from ECEF to ECI
	"""
	return get_DCM_ECI2ECEF(sec).T @ posECEF


def n_posECEF2LLH(phi_n_deg, a, e2):
	"""_summary_

	Args:
		phi_n_deg (_type_): _description_
		a (_type_): _description_
		e2 (_type_): _description_

	Returns:
		_type_: _description_
	"""
	return a / np.sqrt(1.0 - e2 * np.sin(np.deg2rad(phi_n_deg)) * np.sin(np.deg2rad(phi_n_deg)))


def get_pos_ECEF2LLH(posECEF):
	"""位置の座標変換を行う(from ECEF(機体座標系) to LLH(緯度経度高度座標系))

	Args:
		posECEF (numpy.ndarray[3]): position(x, y, z) in ECEF coordinate system.

	Returns:
		numpy.ndarray[3]: position(x, y, z) from ECEF to LLH
	"""
	p = np.sqrt(posECEF[0]**2 + posECEF[1]**2) #現在位置での地球回転軸からの距離[m]
	theta = np.arctan2(posECEF[2]*settings["constants"]["planet"]["Earth"]["a"], p*settings["constants"]["planet"]["Earth"]["b"]) # rad
	t = np.rad2deg(np.arctan2(posECEF[2]+ settings["constants"]["planet"]["Earth"]["ed2"] * settings["constants"]["planet"]["Earth"]["b"] * pow(np.sin(theta), 3), p - settings["constants"]["planet"]["Earth"]["e2"] * settings["constants"]["planet"]["Earth"]["a"] * pow(np.cos(theta), 3)))
	# deg返
	return np.array([t,
					 np.rad2deg(np.arctan2(posECEF[1], posECEF[0])),
					 p / np.cos(np.deg2rad(t)) - n_posECEF2LLH(t, settings["constants"]["planet"]["Earth"]["a"], settings["constants"]["planet"]["Earth"]["e2"])])


def get_pos_LLH2ECEF(posLLH):
	"""位置の座標変換を行う(from LLH(緯度経度高度座標系) to ECEF(機体座標系))

	Args:
		posLLH (numpy.ndarray[3]): position(x, y, z) in LLH coordinate system.

	Returns:
		numpy.ndarray[3]: position(x, y, z) from LLH to ECEF
	"""
	lat = np.deg2rad(posLLH[0])
	lon = np.deg2rad(posLLH[1])
	alt = posLLH[2]
	W = np.sqrt(1.0 - settings["constants"]["planet"]["Earth"]["e2"] * np.sin(lat)**2)
	N = settings["constants"]["planet"]["Earth"]["a"] / W # 卯酉線曲率半径
	return np.array([(N + alt) * np.cos(lat) * np.cos(lon),
					 (N + alt) * np.cos(lat) * np.sin(lon),
					 (N * (1 - settings["constants"]["planet"]["Earth"]["e2"]) + alt) * np.sin(lat)])


def vel_ECI_ECIframe(dcmNED2ECI, vel_ECEF_NEDframe, posECI):
	"""NED(局所平面座標系)におけるECEF(地球中心回転座標系)から見た相対速度(vel_ECEF_NEDframe)から、
		ECI(地球中心慣性座標系)におけるECIから見た慣性速度(vel_ECI_ECIframe)を求める

	Args:
		dcmNED2ECI (numpy.ndarray[3]): DCM from NED to ECI.
		vel_ECEF_NEDframe (numpy.ndarray[3]): NED(局所平面座標系)における相対速度(vel, 回転座標系上での速度)
		posECI (numpy.ndarray[3]): position(x, y, z) in ECI coordinate system.

	Returns:
		numpy.ndarray[3]: velocity(vx, vy, vz)
	"""
	omegaECI2ECEF = np.array([[                                              0.0, -settings["constants"]["planet"]["Earth"]["omega"], 0.0],
							  [settings["constants"]["planet"]["Earth"]["omega"],                                                0.0, 0.0],
							  [                                              0.0,                                                0.0, 0.0]])

	# 相対速度 + 現在位置における地球の自転速度(v = r * ω)
	return dcmNED2ECI@vel_ECEF_NEDframe + omegaECI2ECEF@posECI


def vel_ECEF_NEDframe(dcmECI2NED, vel_ECI_ECIframe, posECI):
	"""ECI(地球中心慣性座標系)におけるECIから見た慣性速度(vel_ECI_ECIframe)から、
		NED(局所平面座標系)におけるECEF(地球中心回転座標系)から見た相対速度(vel_ECEF_NEDframe)を求める

	Args:
		dcmECI2NED (numpy.ndarray[3]): DCM from ECI to NED.
		vel_ECI_ECIframe (numpy.ndarray[3]): ECI(地球中心慣性座標系)における慣性速度(vel)
		posECI (numpy.ndarray[3]): position(x, y, z) in ECI coordinate system.

	Returns:
		numpy.ndarray[3]: velocity(vx, vy, vz)
	"""
	omegaECI2ECEF = np.array([[                                              0.0, -settings["constants"]["planet"]["Earth"]["omega"], 0.0],
							  [settings["constants"]["planet"]["Earth"]["omega"],                                                0.0, 0.0],
							  [                                              0.0,                                                0.0, 0.0]])
	# DCM(ECI2NED) @ (慣性速度 - 現在位置における地球の自転速度(v = r * ω))
	return dcmECI2NED @ (vel_ECI_ECIframe - omegaECI2ECEF @ posECI)


def vel_wind_NEDframe(wind_speed, wind_direction):
	"""風向, 風速の情報からNED(局所平面座標系)における風を求める

	Args:
		wind_speed (float): wind speed [m/s]
		wind_direction (float): wind direction [deg] (north:0, east:90, south:180, west:270)

	Returns:
		numpy.ndarray[3]: wind velocity(vx, vy, vz)
	"""
	return np.array([-wind_speed * np.cos(np.deg2rad(wind_direction)),
					 -wind_speed * np.sin(np.deg2rad(wind_direction)),
					 0.0])


def aoa(vel_AIR_BODYframe):
	"""迎角(attack of angle)を求める

	Args:
		vel_AIR_BODYframe (numpy.ndarray[3]): 大気相対速度

	Returns:
		numpy.ndarray[3]: attack of angle(alpha, beta, gamma) [rad, rad, rad].
	"""
	vel_norm = np.linalg.norm(vel_AIR_BODYframe)
	if(abs(vel_AIR_BODYframe[0]) < 0.001 or vel_norm < 0.01):
		return np.array([0.0, 0.0, 0.0])
	else:
		return np.array([np.arctan2(vel_AIR_BODYframe[2], vel_AIR_BODYframe[0]), # alpha
						 np.arctan2(vel_AIR_BODYframe[1], vel_AIR_BODYframe[0]), # beta
						 np.arctan2(np.sqrt(vel_AIR_BODYframe[1]**2 + vel_AIR_BODYframe[2]**2), vel_AIR_BODYframe[0])]) # gamma


def azi_ele(vel_BODY_NEDframe):
	"""方位角(azimuth), 上下角(elevation)を求める

	Args:
		vel_BODY_NEDframe (numpy.ndarray[3]): NED(局所平面座標系)における相対速度(vel, 機体座標系上での速度)

	Returns:
		numpy.ndarray[2]: azimuth [rad], elevation [rad]
	"""
	north, east, down = vel_BODY_NEDframe[0], vel_BODY_NEDframe[1], vel_BODY_NEDframe[2]
	return np.array([np.pi/2.0 - np.arctan2(north, east), 			  # azi
					 np.arctan2(-down, np.sqrt(north**2 + east**2))]) # ele


def distance_surface(pos1_LLH, pos2_LLH):
	""" 緯度経度高度で記述された距離2地点間の地球表面の距離を算出
		ダウンレンジの計算などに使用
		LLH→ECEFを算出し、直交座標系での地球中心からの角度を求め、角度と地球半径から計算
		http://www.ic.daito.ac.jp/~mizutani/gps/measuring_earth.html

	Args:
		pos1_LLH (numpy.ndarray[3]): position1 (x, y, z) in LLH coordinate system.
		pos2_LLH (numpy.ndarray[3]): position2 (x, y, z) in LLH coordinate system.

	Returns:
		float: distance surface from position1 to position2
	"""
	pos1_ECEF = get_pos_LLH2ECEF(pos1_LLH)
	pos2_ECEF = get_pos_LLH2ECEF(pos2_LLH)
	x = np.dot(pos1_ECEF, pos2_ECEF) / np.linalg.norm(pos1_ECEF) / np.linalg.norm(pos2_ECEF)
	theta = np.arccos(1.0) if 1.0 < x and x < 1.0 + 1.0e-10 else np.arccos(x) # 計算精度によりarccosへの入力が1を極僅かに超える場合に修正する
	return settings["constants"]["planet"]["Earth"]["Re"] * theta


def get_IIP(t, vel_ECEF_NEDframe, posECI):
	"""ECI(地球中心慣性座標系)の位置から真空中予測落下点(IIP)を計算しLLH(緯度経度高度座標系)で返す

	Args:
		t (float): time [s]
		vel_ECEF_NEDframe (numpy.ndarray[3]): NED(局所平面座標系)における相対速度(vel, 回転座標系上での速度)
		posECI (numpy.ndarray[3]): position1 (x, y, z) in ECI coordinate system.

	Returns:
		numpy.ndarray[3]: IIP position(x, y, z) in LLH coordinate system.
	"""
	dcmECI2ECEF = get_DCM_ECI2ECEF(t)
	posLLH = get_pos_ECEF2LLH(get_pos_ECI2ECEF(posECI, dcmECI2ECEF))
	dcmNED2ECI = get_DCM_ECI2NED(dcmECEF2NED=get_DCM_ECEF2NED(posLLH=posLLH), dcmECI2ECEF=dcmECI2ECEF).T
	v_north, v_east, v_up = vel_ECEF_NEDframe[0], vel_ECEF_NEDframe[1], -vel_ECEF_NEDframe[2]
	h = posLLH[2]
	tau = 1.0 / settings["constants"]["planet"]["Earth"]["g0"] * (v_up + np.sqrt(v_up**2 + 2*h*settings["constants"]["planet"]["Earth"]["g0"]))
	dist_IIP_from_now_NED = np.array([v_north * tau, v_east * tau, h])
	posECI_IIP = posECI + dcmNED2ECI @ dist_IIP_from_now_NED
	posECEF_IIP = get_pos_ECI2ECEF(posECI_IIP, dcmECI2ECEF)
	return get_pos_ECEF2LLH(posECEF_IIP)
