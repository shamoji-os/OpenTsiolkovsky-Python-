
# import modules
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import interp1d
import pickle


def calc_three_sigma(path, output_dir=None, target_columns="all", stage="merged", ext="csv", base="time", step=1.0, threshold=0.9973, resume=True):
	""" Perform variance analysis from the Monte Carlo simulation results and calculate the ±3σ value.

	Args:
		path (pathlib object or str): Path to the directory where Monte Carlo simulation results are saved.
		output_dir (pathlib object or str, optional): Specify where to save analysis results. Defaults to None.
		target_columns (str, optional): Specify the symbols to be analyzed for dispersion (e.g. altitude, downrange distance, etc.). Defaults to "all".
		stage (str, optional): Specify the stage (stage1, stage2, stage3, etc.) for variance analysis. Defaults to "merged".
		ext (str, optional): Specify the save format (extension) for Monte Carlo simulation results. Defaults to "csv".
		base (str, optional): Specify the reference x axis symbol when performing dispersion analysis (e.g. time, downrange distance, etc.). Defaults to "time".
		step (float, optional): time step. Defaults to 1.0.
		threshold (float, optional): threshold. Defaults to 0.9973(3σ).
		resume (bool, optional): If there are any results that are still being analyzed, resume from where you left off. Defaults to True.
	"""
	df = {}
	db = {}
	path = Path(path) if type(path) == str else path
	output_dir = Path(output_dir) if type(output_dir) == str else path
	cases = [p for p in path.iterdir() if p.is_dir() and p.name != "nominal"]

	with open(path / Path("nominal/{0}.{1}".format(stage, ext)), "r") as f:
		nominal = pd.read_csv(f)
		columns = []
		for col in nominal.columns:
			if col != base and col != "event":
				columns.append("d+" + col)
				columns.append("d-" + col)
		threshold = int(len(cases) - (threshold * len(cases)))
		if target_columns == "all":
			target_columns = [i for i in nominal.columns if i != "event"]
		else:
			pass

	if resume:
		try:
			with open(output_dir / Path("db.pickle"), "rb") as f:
				db = pickle.load(f)
		except:
			for i in tqdm(cases):
				with open(i / Path(stage + "." + ext), "r") as f:
					db[str(i.name)] = pd.read_csv(f)
			with open(output_dir / Path("db.pickle"), "wb") as o:
				pickle.dump(db, o)

	#TODO:
	base_start = -1.0E+30
	base_end = 1.0E+30
	for v in db.values():
		b = np.array(v[base])
		base_start = b[0] if base_start < b[0] else base_start
		base_end = b[-1] if b[-1] < base_end else base_end
	x = np.arange(base_start, base_end+step, step)

	for col in tqdm(target_columns):
		try:
			y = []
			df[col] = {}
			df["d+" + col] = {}
			df["d-" + col] = {}

			nom_base = np.array(nominal[base])
			nom_target = np.array(nominal[col])
			n = interp1d(nom_base, nom_target)(x)

			for k, v in db.items():
				t_base = np.array(v[base])
				t_target = np.array(v[col])
				y.append(interp1d(t_base, t_target)(x))

			for i in range(len(x)):
				r = [r[i] for r in y]
				r.sort()
				df[col].update({str(x[i]): n[i]})
				df["d+" + col].update({str(x[i]): r[len(y)-threshold-1]})
				df["d-" + col].update({str(x[i]): r[threshold]})
		except:
			continue

	with open(output_dir / Path("three_sigma.csv"), "w", newline="", encoding="utf-8") as o:
		o.write("{0}".format(base))
		for col in df.keys():
			o.write(",{0}".format(col))
		o.write("\n")

		for i in range(len(x)):
			o.write("{0}".format(x[i]))
			for k, v in df.items():
				try:
					o.write(",{0}".format(v[str(x[i])]))
				except:
					o.write(",{0}".format(np.nan))
			o.write("\n")


if __name__ == "__main__":
	from ui import UI
	import time
	ui = UI(title='select result directory.', multi_select=False, stype='dir').get_path()

	start = time.time()
	calc_three_sigma(ui)
	end = time.time()

	print("finished! {0} sec".format(end-start))
