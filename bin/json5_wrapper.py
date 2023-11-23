# Define a wrapper function that adds functionality that supports unique notation to the json5 parser.

import json5

from calculation import *

def load(path):
	"""json5 wrapper supports following addtional notation.
	     * In addition, the following functions are supported by extending the json5 notation.
         *     refer to other items   : Refer to the value of the corresponding key by specifying the key name with "&" at the beginning as follows.
         *                              "[key]": "&[parent_key.referantial_key]" (specify a key from the global scope)
         *     open other setting file: Read data from different configuration files with the following syntax.
         *                              "[key]": "&f&[file_path]" ← ToDo

	Args:
		path (str or pathlib): json5 file path

	Returns:
		dict: load data
	"""
	with open(path, mode="r", encoding="utf-8") as f:
		s = json5.load(f)

		def recursive_find(obj, now=[], d=None):
			"""Recursively search dictionary values ​​and process items written in proprietary notation.

			Args:
				obj (dict, or other type value): When first called, it takes a dictionary type object as an argument.
													Takes several type value as an argument during recursive processing.
				now (list, optional): current position. Defaults to [].
				d (dict, optional): for internal storage of variables. Defaults to None.
			"""
			d = obj if d is None else d
			if type(obj) == dict:
				for k, v in obj.items():
					prev_now = copy.copy(now)
					recursive_find(k, now, d)
					now.append(k)
					recursive_find(v, now, d)
					now = prev_now
			if type(obj) == list:
				for i, v in enumerate(obj):
					prev_now = copy.copy(now)
					now.append(i)
					recursive_find(v, now, d)
					now = prev_now
			if type(obj) == str:
				if obj.startswith('&'):
					l = obj[1:].split('.')
					update_dict(d, now, keys=l)

		recursive_find(s)

		return s


if __name__ == "__main__":
	from pathlib import Path

	s = load(Path("./M-V-1.json5"))
	print()
	print(s)
