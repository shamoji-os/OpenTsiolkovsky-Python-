# Define functions for various calculations

# import modules
import numpy as np
import copy

from define import *


def count_digit(value):
	"""Count the number of digits in a number.

	Args:
		value (int or float): number

	Returns:
		int: the number of digits
	"""
	return sum(c.isdigit() for c in str(value))


def interp_matrix2d(x, y, matrix, bounds_error=True, fill_value=None):
	"""2変数関数として線形補間をする。
	   (Perform linear interpolation as a function of two variables.)

	Args:
		x (float): 取得したいデータ縦方向(Vertical direction of the data you want to obtain.)
		y (float): 取得したいデータ横方向(Horizontal direction of the data you want to obtain.)
		matrix (numpy.ndarray): matrix. shape = (2, 2)
		bounds_error (bool): If True, forced termination if data outside the range is entered.
		# TODO: fill_valueが未実装
		fill_value (tuple or numpy.ndarray[2]): 範囲外の場合にどの値で埋めるか(default(None): データの先頭/末尾でfill)

	Returns:
		_type_: _description_
	"""
	#TODO: 全体的に処理が遅い
	if x < matrix[1, 0] or x > matrix[-1, 0]:
		if bounds_error:
			raise AttributeError("ERROR : interp_matrix2d. First argument(x) is out of the boundary of matrix.")
		else:
			if x < matrix[1, 0]:
				index_x = 1
				delta_x = 0
			elif x > matrix[-1, 0]:
				index_x = matrix.shape[0] - 2
				delta_x = 1
	else:
		for i in range(2, matrix.shape[0]):
			if x < matrix[i, 0]:
				index_x = i-1
				x_lower  = matrix[i-1, 0]
				x_higher = matrix[i,   0]
				delta_x = (x - x_lower)/(x_higher - x_lower)
				break

	if y < matrix[0, 1] or y > matrix[0, -1]:
		if bounds_error:
			raise AttributeError("ERROR : interp_matrix2d. Second argument(y) is out of the boundary of matrix.")
		else:
			if y < matrix[0, 1]:
				index_y = 1
				delta_y = 0
			elif y > matrix[0, -1]:
				index_y = matrix.shape[1] - 2
				delta_y = 1
	else:
		for i in range(2, matrix.shape[1]):
			if y < matrix[0, i]:
				index_y = i-1
				y_lower  = matrix[0, i-1]
				y_higher = matrix[0, i]
				delta_y = (y - y_lower)/(y_higher - y_lower)
				break

	if delta_x < 0.5:
		if delta_y < 0.5:
			res = matrix[index_x, index_y] \
				  + (matrix[index_x+1, index_y  ] - matrix[index_x, index_y]) * delta_x \
				  + (matrix[index_x,   index_y+1] - matrix[index_x, index_y]) * delta_y
		else:
			res = matrix[index_x, index_y+1] \
				  + (matrix[index_x+1, index_y+1] - matrix[index_x, index_y+1]) * delta_x \
				  - (matrix[index_x,   index_y+1] - matrix[index_x, index_y  ]) * (1-delta_y)
	else:
		if delta_y < 0.5:
			res = matrix[index_x+1, index_y] \
				  - (matrix[index_x+1, index_y  ] - matrix[index_x  , index_y]) * (1-delta_x) \
				  + (matrix[index_x+1, index_y+1] - matrix[index_x+1, index_y]) * delta_y
		else:
			res = matrix[index_x+1, index_y+1] \
				  - (matrix[index_x+1, index_y+1] - matrix[index_x,   index_y+1]) * (1-delta_x) \
				  - (matrix[index_x+1, index_y+1] - matrix[index_x+1, index_y  ]) * (1-delta_y)

	return res


def integrate_program_rate(matrix, t, start_time=0.0):
	return [i.integral(start_time, t) for i in matrix]


def integrate_program_rate_ref(matrix, t, start_time=0.0):
	""" プログラムレートを積算して指定した時刻での姿勢角の変化量を求める
		(Calculate the amount of change in attitude angle at a specified time by integrating the program rate.)

	Args:
		matrix (pandas.DataFrame): the data table of program rate
		t (float or tuple): (start_time, end_time) [s] - start_time is optional.

	Returns:
			numpy.ndarray[3]: delta attitude (PITCH, YAW, ROLL) [deg, deg, deg]
	"""
	time = np.array(matrix["time"])
	matches = np.where(start_time <= time, True, False) * np.where(time <= t, True, False)
	matched_time = time[matches]
	dt = (matched_time - np.insert(matched_time, 0, 0)[0:len(matched_time)])[1:]
	pitch = np.array(matrix["PITCH"])[matches]
	yaw = np.array(matrix["YAW"])[matches]
	roll = np.array(matrix["ROLL"])[matches]

	return np.array([
		np.sum(pitch[:len(matched_time)-1] * dt) + pitch[-1] * (t - matched_time[-1]),
		np.sum(yaw[:len(matched_time)-1]   * dt) + yaw[-1]   * (t - matched_time[-1]),
		np.sum(roll[:len(matched_time)-1]  * dt) + roll[-1]  * (t - matched_time[-1]),
	])


def update_dict(d, now, value=None, keys=None, _d2=None):
	""" Update the value of one key in a nested dictionary with the value of another key.

	Args:
		d (dict): dictionary data to be updated
		now (list[str, int]): current position
		keys (list[str, int]): a key indicating the value you want to update
		d2 (dict, optional): for internal storage of variables. Defaults to None.
	"""
	_d2 = copy.deepcopy(d) if _d2 is None else _d2

	# valueが設定されている場合、d(辞書)のnow(現在位置)の値をvalueの値に更新する
	if value is not None:
		if keys is not None:
			raise AttributeError("Specify only one of value or keys")
		if len(now) == 1:
			d[now[0]] = value
		else:
			update_dict(d[now[0]], now[1:], value=value, _d2=_d2)

	# keysが設定されている場合、d(辞書)のnow(現在位置)の値をd(辞書)のkeysの値で更新する
	elif keys is not None:
		if len(now) == 1:
			if len(keys) == 1:
				d[now[0]] = _d2[keys[0]]
			else:
				update_dict(d, now, keys=keys[1:], _d2=_d2[keys[0]])
		else:
			update_dict(d[now[0]], now[1:], keys=keys, _d2=_d2)

	else:
		raise AttributeError("please set argument in update_dict().")


def runge_kutta(f, t, x0, t_eval=None, observer=None, show_message=False):
	x = x0
	step = np.diff(t)
	t = t[0]
	result = {str(t): x}

	f(t, x)
	observer(t, x)


	for dt in step:
		k1 = f(t, x)
		k2 = f(t + dt, x + dt*k1/2)
		k3 = f(t + dt/2, x + dt*k2/2)
		k4 = f(t + dt, x + dt*k3)
		x = x + dt*(k1 + 2*k2 + 2*k3 + k4)/6
		t = t + dt
		result[str(t)] = x

		if show_message:
			print(t, x[0], x[1], x[2], x[3], x[4], x[5], x[6])

		if observer is not None:
			if t_eval is None or t in t_eval:
				obs = observer(t, x)
				if obs[0] == "deployed":
					x = obs[2]
				elif obs[0] == "exit":
					return result
				elif obs[0] == "continue":
					pass
				else:
					raise AttributeError("observer result is error.")

	return result
