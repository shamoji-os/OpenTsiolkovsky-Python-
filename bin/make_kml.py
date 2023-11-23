# -*- coding: utf-8 -*-
# Copyright (c) 2017 Interstellar Technologies Inc. All Rights Reserved.
# Authors : Takahiro Inagawa
# ==============================================================================

# 「OpenTsiolkovsky」の同名ファイルの移植版

import sys
import os
import simplekml
import numpy as np
import json5
import pandas as pd
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from define import *
from ui import UI

stage_literal = ["", "M", "S", "T", "F"]  # ex. MECO SECO etc...
kml = simplekml.Kml(open=1)

def make_kml(name, div, file_path):
    """
    Args:
        name (str) : ロケット名(jsonファイルのname)
        div (int) : 間引き数
        file_path : csvファイルへのパス
    """
    try:
        df = pd.read_csv(file_path, index_col=False)
        time = df["time"]
        lat = df["lat"]
        lon = df["lon"]
        altitude = df["alt"]
        lat_IIP = df["latIIP"]
        lon_IIP = df["lonIIP"]
        #is_powered = df[""]
        #is_separated = df["is_separated(1=already 0=still)"]
        # イベント毎に点を打つために前の値からの変化を監視して変化していればcoord_pointに追加
        #is_powered_prev = is_powered[0]
        #is_separated_prev = is_separated[0]
        name = file_path.stem
        """
        for i in range(len(time)):
            if (is_powered_prev == 1 and is_powered[i] == 0):
                pnt = kml.newpoint(name=stage_literal[stage] + "ECO")
                pnt.coords = [(lon.iloc[i], lat.iloc[i], altitude.iloc[i])]
                pnt.description = "T+" + str(time.iloc[i]) + "[sec]"
                pnt.style.iconstyle.icon.href = "http://earth.google.com/images/kml-icons/track-directional/track-none.png"
                pnt.altitudemode = simplekml.AltitudeMode.absolute
            if (is_powered_prev == 0 and is_powered[i] == 1):
                pnt = kml.newpoint(name=stage_literal[stage] + "EIG")
                pnt.coords = [(lon.iloc[i], lat.iloc[i], altitude.iloc[i])]
                pnt.description = "T+" + str(time.iloc[i]) + "[sec]"
                pnt.style.iconstyle.icon.href = "http://earth.google.com/images/kml-icons/track-directional/track-none.png"
                pnt.altitudemode = simplekml.AltitudeMode.absolute
            if (is_separated_prev == 0 and is_separated[i] == 1):
                pnt = kml.newpoint(name=stage_literal[stage] + "SEP")
                pnt.coords = [(lon.iloc[i], lat.iloc[i], altitude.iloc[i])]
                pnt.description = "T+" + str(time.iloc[i]) + "[sec]"
                pnt.style.iconstyle.icon.href = "http://earth.google.com/images/kml-icons/track-directional/track-none.png"
                pnt.altitudemode = simplekml.AltitudeMode.absolute
            is_powered_prev = is_powered[i]
            is_separated_prev = is_separated[i]
        """
        # 間引いた時点ごとに線を引く
        coord_line = []
        for i in range(len(time)//div):
            index = i * div
            coord_line.append((lon.iloc[index], lat.iloc[index], altitude.iloc[index]))
        coord_line.append((lon.iloc[-1], lat.iloc[-1], altitude.iloc[-1]))
        ls = kml.newlinestring(name="%s" % (name))
        ls.style.linestyle.width = 3
        ls.extrude = 0  # 高度方向の線を無くしたいときはここを変更
        ls.altitudemode = simplekml.AltitudeMode.absolute
        ls.coords = coord_line
        ls.style.linestyle.color = simplekml.Color.aliceblue
        ls.style.linestyle.colormode = simplekml.ColorMode.normal
        ls.lookat.latitude = lat.iloc[0]
        ls.lookat.longitude = lon.iloc[0]
        ls.lookat.range = 200000
        # IIP線を引く
        coord_IIP = []
        for i in range(len(time)//div):
            index = i * div
            coord_IIP.append((lon_IIP.iloc[index], lat_IIP.iloc[index]))
        coord_line.append((lon.iloc[-1], lat.iloc[-1]))
        ls_IIP = kml.newlinestring(name="%s IIP" % (name))
        ls_IIP.style.linestyle.width = 3
        ls_IIP.coords = coord_IIP
        ls_IIP.style.linestyle.colormode = simplekml.ColorMode.normal
        ls_IIP.style.linestyle.color = simplekml.Color.changealphaint(150, simplekml.Color.antiquewhite)
        print("created kml file:" + name)
    except:
        print("Error: {0} CANNNOT be maked kml.".format(name))

if __name__ == '__main__':
    file_name = UI(title='select setting file(json5).', multi_select=False, stype='file').get_path()
    result_list = UI(title='select result files.', multi_select=True, stype='file').get_path()
    data = json5.load(open(file_name, encoding="utf-8"))
    name = data["ProjectName"] if data["ProjectName"] is not None else file_name.stem

    time_step_output = 1  # KML出力の時間ステップ[sec]
    time_step = data["sim_settings"]["calculate_condition"]["step"]
    reduce_interval = int(time_step_output // time_step)

    print("INPUT FILE: %s" % (file_name))
    for v in result_list:
        make_kml(name, reduce_interval, v)

    kml.save("output/" + name + "_GoogleEarth.kml")
    print("Done...")
