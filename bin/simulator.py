
# import modules
import numpy as np
from scipy.integrate import odeint, solve_ivp
from itertools import count
import json

from define import *
from rocket import *
from calculation import *


class Simulator():
	"""Conduct a rocket flight simulation (called directly under main).
	"""
	def __init__(self, rocket_conf, id=None, output_dir=None):
		self.id = id
		Logger.output_dir = output_dir if output_dir is not None else None
		self.rocket = Rocket(rocket_conf, id=id)

	def simulation(self):
		if settings["system"]["show_message"] > 1:
			print("time[s]\tmass[kg]\tx[m](ECI)\ty[m](ECI)\tz[m](ECI)\tvx[m/s](ECI)\tvy[m/s](ECI)\tvz[m/x](ECI)")

		calc_condition = settings["sim_settings"]["calculate_condition"]
		step = settings["sim_settings"]["calculate_condition"]["step"]

		for i in count():
			status = self.rocket.get_status(SimulationStatus(status=0, num=i))
			if status.is_continue:
				time_array = list(np.arange(status.calc_start_time, calc_condition["end"]+step, step))
				time_array += list(settings["SOE"].values())
				time_array =[t for t in sorted(list(set(time_array))) if status.calc_start_time <= t and t <= calc_condition["end"]]
				#solver = odeint(self.rocket, status.y0, time_array)
				res = runge_kutta(self.rocket, time_array, status.y0, show_message=True if settings["system"]["show_message"]>1 else False, observer=self.rocket.observer)
				#sol = solve_ivp(self.rocket, (status.calc_start_time, calc_condition["end"]), status.y0, method="RK45", rtol=calc_condition["rtol"], atol=calc_condition["atol"])
				#sol = pd.DataFrame(, columns=["mass", "x", "y", "z", "vx", "vy", "vz"]).to_csv(o, index=False)
				#for t in range(len(solver)):
				#	self.rocket(solver[t], time_array[t], is_repeat_phase=True)
				#with open(Logger.output_dir + "/" + str(i) + ".csv", "w") as o:
				#	pd.DataFrame(solver, columns=["mass", "x", "y", "z", "vx", "vy", "vz"]).to_csv(o, index=False)

				"""
				time_array = list(settings["SOE"].values())
				time_array.insert(0, status.calc_start_time)
				time_array.append(settings["sim_settings"]["calculate_condition"]["end"])
				time_array = [t for t in sorted(list(set(time_array)))  if status.calc_start_time <= t and t <= settings["sim_settings"]["calculate_condition"]["end"]]

				calc_condition = settings["sim_settings"]["calculate_condition"]
				solver = ode(self.rocket).set_integrator(calc_condition["method"], rtol=calc_condition["rtol"], atol=calc_condition["atol"])
				solver.set_solout(self.rocket.observer)
				solver.set_initial_value(status.y0, time_array[0])

				# flight simulation loop TODO: mass reduce 未実装
				for j in range(0, len(time_array)-1):
					while solver.successful() and solver.t < time_array[j+1] and self.rocket.get_status(SimulationStatus(status=1)).is_flying:
						# Continue the calculation until the event time. Maintains the calculation interval even if the event time is specified in seconds that are finer than the step time.
						if solver.t + step < time_array[j+1]:
							t = solver.t + step if not round(solver.t%step, count_digit(step)) > 0 else (solver.t//step * step) + step
						else:
							t = time_array[j+1]
						self.rocket.get_status(SimulationStatus(status=1, target_time=t))
						sol = solver.integrate(t)
						if settings["system"]["show_message"] > 1:
							print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(t, sol[0], sol[1], sol[2], sol[3], sol[4], sol[5], sol[6]))

				if settings["system"]["show_message"] > 0:
					res = self.rocket.get_status(SimulationStatus(status=2))
					res.impact_point = ["is", "flying"] if res.impact_point is None else res.impact_point
					print("\n{0} impact point [deg]:\t{1}\t{2}".format(res.stage_name, res.impact_point[0], res.impact_point[1]))
				"""

			else:
				break

		# show result
		if settings["system"]["show_message"] > 0:
			res = self.rocket.get_status(SimulationStatus(status=3))
			if settings["sim_settings"]["montecarlo_simulation"] != False:
				print("\nid: {0} is finished.".format(res.id), end="")
			print("\nmax altitude[m] :\t{0}".format(res.max_values["alt"]))
			print("max downrange[m]:\t{0}".format(res.max_values["downrange"]))

		if settings["system"]["output_log"]:
			self.rocket.log.save()
			with open(Logger.output_dir + "/" + self.rocket.id + "/" + "conf.json5", "w") as f:
				json.dump(self.rocket.rf, f, cls=MyEncoder)


class MyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		elif isinstance(obj, pd.DataFrame):
			return obj.to_json()
		elif isinstance(obj, StateQuantity):
			return {"pos": obj.pos.tolist(), "vel": obj.vel.tolist(), "acc": obj.acc.tolist(), "attitude": obj.attitude.tolist()}
		else:
			return super(MyEncoder, self).default(obj)