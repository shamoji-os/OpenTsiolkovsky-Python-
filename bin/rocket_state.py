# -*- coding:utf-8 -*-

import numpy as np
from enum import Enum


class FlightMode(Enum):
	Power_3DOF = "3DOF"
	Power_6DOF= "6DOF"
	Free_Ballistic = "ballistic"
	Inertial_Object = "inertial_object"
	default = "default"


class StateQuantity():
	def __init__(self, pos=None, vel=None, acc=None, altitude=None):
		""" 状態量を定義する  また各座標系がここで定義する状態量を保持する
			(Define the state quantity Also, each coordinate system holds the state quantity defined here)

		Args:
			pos (numpy.ndarray[3], optional): position. Defaults to None.
			vel (numpy.ndarray[3], optional): velocity. Defaults to None.
			acc (numpy.ndarray[3], optional): acceleration. Defaults to None.
			altitude (numpy.ndarray[3], optional): altitude. Defaults to None.
		"""
		self.pos = np.zeros(3)
		self.vel = np.zeros(3)
		self.acc = np.zeros(3)
		self.attitude = np.zeros(3)


class StatusFlag():
	def __init__(self) -> None:
		self.separated = None    # None or RocketState.ECI in separated
		self.is_flying = True    # bool
		self.deployed = None     # None or x(mass, x, y, z, vx, vy, vz(ECI)) in deployed object


class RocketState():
	"""	ロケットの状態を定義する  座標系の略称は以下の通り
		(The abbreviations of the coordinate systems that define the state of the rocket are as follows:)
			I:ECI(地球中心慣性座標系)
			E:ECEF(地球中心回転座標系)
			L:LLH(緯度経度高度座標系)
			H:NED(局所平面座標系)
			A:AIR(速度座標系)
			B:BODY(機体座標系)
	"""
	def __init__(self, name=""):
		self.name = name
		self.m_dot = 0.0
		self.flag = StatusFlag()
		self.ECI = StateQuantity()
		self.ECEF = StateQuantity()
		self.LLH = StateQuantity()
		self.NED = StateQuantity()
		self.AIR = StateQuantity()
		self.BODY = StateQuantity()
		self.angle_of_attack = np.zeros(3)
		self.mach = 0.0
		self.dynamic_pressure = 0.0
		self.posLLH_IIP = None
		self.kinematic_energy = 0.0
		self.downrange = 0.0
		self.propulsion = type("Propulsion", (object,), {})
		self.structure = type("Structure", (object,), {})
		self.force = type("Force", (object,), {})
		self.loss = type("Loss", (object,), {})
		self.dcm = type("DCM", (object,), {})
		self.flight_mode = FlightMode("default")
		self.impact_point = None
