# OpenTsiolkovsky(Python)
#
# 本プログラムは、インターステラテクノロジズ株式会社殿が開発された
#「OpenTsiolkovsky」(https://github.com/istellartech/OpenTsiolkovsky)を基に、
# コアプログラムのPython化、リファクタリング等の修正を行ったプログラムです。
# 現在のところ、動作は「OpenTsiolkovsky」のVer.0.41に準拠した動作を行います。
# 本プログラムのライセンスはMITライセンスに準拠します。
#
# This program is based on "OpenTsiolkovsky" (https://github.com/istellartech/OpenTsiolkovsky)
# developed by Interstellar Technologies, Inc., with modifications such as converting
# the core program to Python and refactoring.
# Currently, the operation complies with "OpenTsiolkovsky" Ver.0.41.
# The license for this program is based on the MIT License.
#
# 2023.11.17: shamoji

# import modules
import copy
import numpy as np
import pandas as pd
import os
import sys
import multiprocessing
import datetime
import time
import cProfile
from pathlib import Path
import shutil
from types import MappingProxyType
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), './bin/'))
from define import *
from simulator import Simulator


class SimulationManager():
	""" Simulation Management Class
	"""
	def __init__(self, output_dir=None) -> None:
		""" Initialize and create output folder.

		Args:
			output_dir (string, optional): Path to output folder. Defaults to None.
		"""
		self.process_list = {}

		if output_dir is None:
			t_delta = datetime.timedelta(hours=9)
			JST = datetime.timezone(t_delta, 'JST')
			now = datetime.datetime.now(JST)
			try:
				self.output_dir = sys.argv[2] + "/{0}({1})/".format(settings["ProjectName"], now.strftime("%Y%m%d_%H%M%S"))
			except:
				self.output_dir = "./output/{0}({1})/".format(settings["ProjectName"], now.strftime("%Y%m%d_%H%M%S"))
			p = None if Path(self.output_dir).exists() else Path(self.output_dir).mkdir(parents=True)
		else:
			self.output_dir = output_dir


	def process(self, id=None, conf=None):
		""" Run the simulation.
			(If no execution conditions are specified, the flight will be executed using the nominal flight path conditions.)

		Args:
			id (string or int, optional): Specify the id of the process to run. Defaults to None.
			conf (dict, optional): Specify execution conditions. Defaults to None.
		"""
		id = str(id) if id is not None else "nominal"
		conf = conf if conf is not None else nominal_staging
		self.process_list[id] = Simulator(rocket_conf=copy.deepcopy(conf), id=id, output_dir=self.output_dir)
		self.process_list[id].simulation()


	def montecarlo(self, nominal_output=True):
		""" Run the montecarlo simulation.

		Args:
			nominal_output (bool, optional): Outputs the nominal flight path. Defaults to True.
		"""
		# Initializing Monte Carlo simulation execution conditions
		ms = settings["sim_settings"]["montecarlo_simulation"]
		np.random.seed(ms["seed"])
		thread = ms["num_thread"] if ms["num_thread"] != 0 else max(multiprocessing.cpu_count()-1, 1)
		threads = thread
		nom = 1 if nominal_output else 0

		# Run Monte Carlo simulation for specified number of cases
		loop = (ms["num_cases"]) // thread + 1
		mod = (ms["num_cases"]) - ((loop-1)*thread)
		for loop_index in tqdm(range(loop)):
			processes = []
			if loop_index == loop - 1:
				if nominal_output:
					proc = multiprocessing.Process(target=self.process, args=("nominal", nominal_staging))
					processes.append(proc)
					proc.start()
				thread = mod
			for i in range(thread):
				id = loop_index*threads + i + 1
				#TODO: 姿勢誤差とかcommonな設定もモンテカルロできるように修正したい
				proc = multiprocessing.Process(target=self.process, args=(id, self.generate_input_condition()["stage"]))
				processes.append(proc)
				proc.start()
			for p in processes:
				p.join()


	def generate_input_condition(self):
		""" Generate execution conditions (error cases) used in Monte Carlo simulation.

		Returns:
			dict: simulation's configuration
		"""
		ms_errordict = settings["sim_settings"]["montecarlo_simulation"]["error"]
		df = {}

		def get_scale(value, error, error_type):
			nominal = np.array(value) if not isinstance(value, pd.DataFrame) else value.iloc[:, 1:]
			if error_type == '%':
				scale = nominal*(error/100)/3
			elif error_type == '*':
				scale = nominal*error/3
			elif error_type == 'add':
				scale = error/3 if not isinstance(nominal, pd.DataFrame) else nominal*0+(error/3)
			if isinstance(scale, pd.DataFrame):
				scale.insert(0, "time", value["time"])
			return scale

		def recursive(base_dict, output_dict, error_dict, _now=None):
			_now = [] if _now is None else _now
			if type(base_dict) == dict or type(base_dict) == MappingProxyType:
				for k, v in base_dict.items():
					prev_now = copy.deepcopy(_now)
					output_dict[k] = copy.deepcopy(v)
					_now.append(k)
					if _now in [x.split('.') for x in error_dict.keys()]:
						e = '.'.join(_now)
						scale = get_scale(output_dict[k], error_dict[e][0], error_dict[e][1])
						if isinstance(scale, pd.DataFrame):
							o = pd.DataFrame(columns=scale.columns)
							for col in scale.columns:
								if col == "time":
									o[col] = scale[col]
								else:
									o[col] = np.random.normal(loc=base_dict[k][col], scale=scale[col])
							output_dict[k] = o
						else:
							output_dict[k] = np.random.normal(loc=base_dict[k], scale=scale)
					recursive(v, output_dict[k], error_dict, _now)
					_now = prev_now

		recursive(settings, df, ms_errordict)

		return df


if __name__ == "__main__":
	print("Hello, OpenTsiolkovsky(Python)!")
	start = time.time()

	def f():
		sm = SimulationManager()
		if settings["sim_settings"]["montecarlo_simulation"] != False:
			sm.montecarlo(nominal_output=settings["sim_settings"]["montecarlo_simulation"]["nominal_output"])
		else:
			sm.process()
		shutil.copy(Path(sys.argv[1]), sm.output_dir)
		shutil.copy(Path("./profiling.stats"), sm.output_dir)

	cProfile.run('f()', "profiling.stats") if settings["system"]["profiling"] else f()

	stop = time.time()
	print("Processing time: {0} [sec]\n".format(stop - start))
