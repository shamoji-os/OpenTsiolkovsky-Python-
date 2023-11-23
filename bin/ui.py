# -*- coding:utf-8 -*-

import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# UIクラスを定義
# title: Window Titleを設定
# multi_select: 複数選択モードのON/OFF
# stype: 'file' or 'dir'
# show_message: default 'False'
class UI:
	def __init__(self, title = None, multi_select = False, stype = 'file',show_message = False):
		# 選択したファイル情報を保存する変数を定義
		self.get = None
		self.title = title
		self.multi_select = multi_select
		self.show_message = show_message
		self.stype = stype

		# tkinterパッケージを呼び出し初期化する
		self.root = tk.Tk()
		self.root.withdraw()

	# ダイアログを表示しユーザが選択したファイルのパスを取得する
	def get_path(self, title=None, multi_select=None, stype=None, show_message=None):
		title = title if title != None else self.title
		multi_select = multi_select if multi_select != None else self.multi_select
		stype = stype if stype != None else self.stype
		show_message = show_message if show_message != None else self.show_message

		if stype == 'file':
			file_type = [('', '*')]
			path = filedialog.askopenfilenames(filetypes = file_type, title = title)
			print(path)
			print(len(path))
			if multi_select is False and len(path) != 1:
				raise Exception("ファイルは１つだけ選択してください。")
		elif stype == 'dir' or 'directory' or 'folder':
			path = filedialog.askdirectory()
			if multi_select is True:
				print("typeがdirectoryの場合はmulti_select機能に対応していません.")
		else:
			raise Exception('UI get_get Error: stype input error.')

		# 選択したファイルパス（文字列）をPathオブジェクトに変換して返す（複数選択時はリスト化）
		if multi_select and stype == 'file':
			self.get =[]
			for i in path:
				self.get.append(Path(i))
		elif stype == 'file':
			self.get = Path(path[0])
		elif stype == 'dir' or 'directory' or 'folder':
			self.get = Path(path)

		if show_message:
			print("選択: {0}".format(self.get))
			print(type(path))

		return self.get


# test code
if __name__ == '__main__':
	if True:
		# ファイルを選択する(single)
		ui = UI(title='select input data(single).', multi_select=False, stype='file', show_message=True)
		ui.get_path()
		print("\n選択したファイル:\n{0}\n".format(ui.get))
		print("file type: {0}".format(type(ui.get)))

	if True:
		# ファイルを選択する(multi)
		ui = UI(title='select input data(multi).', multi_select=True, stype='file', show_message=True)
		ui.get_path()
		print("\n選択したファイル:\n{0}\n".format(ui.get))
		print("file type: {0}".format(type(ui.get)))
