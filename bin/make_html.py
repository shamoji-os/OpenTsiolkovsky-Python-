# -*- coding: utf-8 -*-
# Copyright (c) 2017 Interstellar Technologies Inc. All Rights Reserved.
# Authors : Takahiro Inagawa
#
# Lisence : MIT Lisence
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

# 「OpenTsiolkovsky」の同名ファイルの移植版

import sys
import os
import io
import numpy as np
import pandas as pd
import json5
from pathlib import Path

from jinja2 import Template

from bokeh.embed import components
from bokeh.models import Range1d
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.util.browser import view
from bokeh.palettes import d3
from bokeh.layouts import gridplot
from bokeh.io import output_file, show
from bokeh.models import PrintfTickFormatter, HoverTool

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from define import *
from ui import UI

g = settings["constants"]["planet"]["Earth"]["g0"]

input_json = UI(title='select setting file(json5).', multi_select=False, stype='file').get_path()
result_file = UI(title='select output file(csv).', multi_select=False, stype='file').get_path()

# ==== Read input file ====
f = open(input_json, 'r', encoding="utf-8")
json_dict = json5.load(f)
# rocket_name = json_dict['name']
rocket_name = json_dict['ProjectName'] if json_dict['ProjectName'] is not None else input_json.stem

# ==== Bokeh setup ====
TOOLS="pan,wheel_zoom,box_zoom,reset,save,hover"
PLOT_OPTIONS = dict(tools=TOOLS, plot_width=550, plot_height=300)
HOVER_SET = [("date (x,y)", "($x{0,0}, $y{0,0})")]
HOVER_SET_F = [("date (x,y)", "($x{0,0}, $y{0,0.00})")]
C = d3["Category10"][10]

# ==== Plot each stages ====
for stage_str in ['1', '2', '3']:
    st = stage_str + ' stage: ' # stage string for title
    file_name = result_file
    if (os.path.exists(file_name) != True):continue
    df1 = pd.read_csv(file_name, index_col=False)
    # ==== 燃焼終了 or 遠地点までのプロットの場合コメントオンオフ ====
    time_burnout = df1[df1["thrust"] == 0]["time"][1:].min()
    time_apogee = df1[df1["alt"] == df1["alt"].max()]["time"]
    # import pdb; pdb.set_trace()
    if not isinstance(time_apogee, float):
        time_apogee = time_apogee.iloc[0]
    # df1 = df1[df1["time(s)"] < float(time_burnout)]
    df1 = df1[df1["time"] < float(time_apogee)]
    # ==== 燃焼終了 or 遠地点までのプロットの場合コメントオンオフ ====

    xr1 = Range1d(start=0, end=df1["time"].max())

    p_mass = figure(title=st+"質量", x_axis_label="時刻 [sec]", y_axis_label="質量 [kg]",
                    x_range=xr1, **PLOT_OPTIONS)
    p_mass.line(df1["time"], df1["mass"], color=C[0])
    p_mass.select_one(HoverTool).tooltips = HOVER_SET


    p_thrust = figure(title=st+"推力", x_axis_label="時刻 [sec]", y_axis_label="推力 [N]",
                      x_range=xr1, **PLOT_OPTIONS)
    p_thrust.line(df1["time"], df1["thrust"], color=C[1])
    p_thrust.yaxis[0].formatter = PrintfTickFormatter(format="%f")
    p_thrust.select_one(HoverTool).tooltips = HOVER_SET


    p_alt = figure(title=st+"高度", x_axis_label="時刻 [sec]", y_axis_label="高度 [m]",
                   x_range=xr1, **PLOT_OPTIONS)
    p_alt.line(df1["time"], df1["alt"], color=C[2])
    p_alt.yaxis[0].formatter = PrintfTickFormatter(format="%f")
    p_alt.select_one(HoverTool).tooltips = HOVER_SET


    p_Isp = figure(title=st+"比推力", x_axis_label="時刻 [sec]", y_axis_label="比推力 [秒]",
                   x_range=xr1, **PLOT_OPTIONS)
    p_Isp.line(df1["time"], df1["Isp"], color=C[3])
    p_Isp.select_one(HoverTool).tooltips = HOVER_SET


    p_downrange = figure(title=st+"ダウンレンジ", x_axis_label="時刻 [sec]", y_axis_label="ダウンレンジ [m]",
                         x_range=xr1, **PLOT_OPTIONS)
    p_downrange.line(df1["time"], df1["downrange"], color=C[4])
    p_downrange.yaxis[0].formatter = PrintfTickFormatter(format="%f")
    p_downrange.select_one(HoverTool).tooltips = HOVER_SET


    p_profile = figure(title=st+"飛翔プロファイル", x_axis_label="ダウンレンジ [m]", y_axis_label="高度 [m]",
                       **PLOT_OPTIONS)
    p_profile.line(df1["downrange"], df1["alt"], color=C[5])
    p_profile.xaxis[0].formatter = PrintfTickFormatter(format="%f")
    p_profile.yaxis[0].formatter = PrintfTickFormatter(format="%f")
    p_profile.select_one(HoverTool).tooltips = HOVER_SET


    p_velh = figure(title=st+"水平面速度", x_axis_label="時刻 [sec]", y_axis_label="速度 [m/s]",
                         x_range=xr1, **PLOT_OPTIONS)
    vel_horizontal = np.sqrt(df1["vx(NED)"] ** 2 + df1["vy(NED)"] ** 2)
    p_velh.line(df1["time"], df1["vx(NED)"], legend="North", color=C[6])
    p_velh.line(df1["time"], df1["vy(NED)"], legend="East", color=C[7])
    p_velh.line(df1["time"], vel_horizontal, legend="Horizon", color=C[8])
    p_velh.select_one(HoverTool).tooltips = HOVER_SET


    p_velv = figure(title=st+"垂直速度", x_axis_label="時刻 [sec]", y_axis_label="速度 [m/s]",
                         x_range=xr1, **PLOT_OPTIONS)
    p_velv.line(df1["time"], -df1["vz(NED)"], legend="Up", color=C[8])
    p_velv.select_one(HoverTool).tooltips = HOVER_SET


    p_q = figure(title=st+"動圧", x_axis_label="時刻 [sec]", y_axis_label="動圧 [kPa]",
                         x_range=xr1, **PLOT_OPTIONS)
    p_q.line(df1["time"], df1["Q"] / 1000, color=C[9])
    p_q.select_one(HoverTool).tooltips = HOVER_SET


    p_mach = figure(title=st+"機体のその場高度でのマッハ数", x_axis_label="時刻 [sec]", y_axis_label="マッハ数 [-]",
                         x_range=xr1, **PLOT_OPTIONS)
    p_mach.line(df1["time"], df1["Mach"], color=C[0])
    p_mach.select_one(HoverTool).tooltips = HOVER_SET_F


    p_acc = figure(title=st+"加速度", x_axis_label="時刻 [sec]", y_axis_label="加速度 [G]",
                   x_range=xr1, **PLOT_OPTIONS)
    acc = np.sqrt(df1["ax(BODY)"] **2 + df1["ax(BODY)"] ** 2 + df1["ax(BODY)"] ** 2)
    p_acc.line(df1["time"], acc / g, color=C[1])
    p_acc.select_one(HoverTool).tooltips = HOVER_SET_F


    p_acc3 = figure(title=st+"機体の各軸にかかる加速度", x_axis_label="時刻 [sec]", y_axis_label="加速度 [G]",
                    x_range=xr1, **PLOT_OPTIONS)
    p_acc3.line(df1["time"], df1["ax(BODY)"] / g, legend="X", color=C[2])
    p_acc3.line(df1["time"], df1["ay(BODY)"] / g, legend="Y", color=C[3])
    p_acc3.line(df1["time"], df1["az(BODY)"] / g, legend="Z", color=C[4])
    p_acc3.select_one(HoverTool).tooltips = HOVER_SET_F


    p_az = figure(title=st+"姿勢：方位角", x_axis_label="時刻 [sec]", y_axis_label="角度 [deg]",
                   x_range=xr1, **PLOT_OPTIONS)
    p_az.line(df1["time"], df1["YAW"], color=C[5])
    p_az.select_one(HoverTool).tooltips = HOVER_SET


    p_el = figure(title=st+"姿勢：仰角", x_axis_label="時刻 [sec]", y_axis_label="角度 [deg]",
                   x_range=xr1, **PLOT_OPTIONS)
    p_el.line(df1["time"], df1["PITCH"], color=C[6])
    p_el.select_one(HoverTool).tooltips = HOVER_SET


    p_AoAa = figure(title=st+"迎角：α", x_axis_label="時刻 [sec]", y_axis_label="迎角 [deg]",
                   x_range=xr1, **PLOT_OPTIONS)
    p_AoAa.line(df1["time"], df1["aoa(alpha)"], color=C[7])
    p_AoAa.select_one(HoverTool).tooltips = HOVER_SET_F


    p_AoAb = figure(title=st+"迎角：β", x_axis_label="時刻 [sec]", y_axis_label="迎角 [deg]",
                   x_range=xr1, **PLOT_OPTIONS)
    p_AoAb.line(df1["time"], df1["aoa(beta)"], color=C[8])
    p_AoAb.select_one(HoverTool).tooltips = HOVER_SET_F

    p_AoAg = figure(title=st+"全迎角：γ", x_axis_label="時刻 [sec]", y_axis_label="迎角 [deg]",
                   x_range=xr1, **PLOT_OPTIONS)
    p_AoAg.line(df1["time"], df1["aoa(gamma)"], color=C[7])
    p_AoAg.select_one(HoverTool).tooltips = HOVER_SET_F

    """
    p_drag = figure(title=st+"抗力", x_axis_label="時刻 [sec]", y_axis_label="抗力 [N]",
                   x_range=xr1, **PLOT_OPTIONS)
    p_drag.line(df1["time(s)"], df1["aero Drag(N)"], color=C[9])
    p_drag.select_one(HoverTool).tooltips = HOVER_SET


    p_lift = figure(title=st+"揚力", x_axis_label="時刻 [sec]", y_axis_label="揚力 [N]",
                   x_range=xr1, **PLOT_OPTIONS)
    p_lift.line(df1["time(s)"], df1["aero Lift(N)"], color=C[0])
    p_lift.select_one(HoverTool).tooltips = HOVER_SET
    """

    # plots can be a single Bokeh model, a list/tuple, or even a dictionary
    # plots = {'mass': p_mass, 'thrust': p_thrust, 'Blue': p2, 'Green': p3}

    plots = gridplot([[p_mass, p_thrust], [p_alt, p_Isp],
                      [p_downrange, p_profile], [p_velh, p_velv],
                      [p_q, p_mach], [p_acc, p_acc3],
                      [p_az, p_el], [p_AoAa, p_AoAb]])

    if stage_str == '1':
        script1, div1 = components(plots)
        script2, div2 = "", ""
        script3, div3 = "", ""
    elif stage_str == '2':
        script2, div2 = components(plots)
    elif stage_str == '3':
        script3, div3 = components(plots)

# ==== Output HTML ====
filename = "output/" + rocket_name + "_output.html"

template = Template('''<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Bokeh Scatter Plots</title>
        {{ js_resources }}
        {{ css_resources }}
        {{ script1 }}
        {{ script2 }}
        {{ script3 }}
        <style>
            .embed-wrapper {
                width: 95%;
                height: 2800px;
                margin: auto;
            }
        </style>
    </head>
    <body>
        <H1>1st stage</H1>
        <div class="embed-wrapper">
            {{ div1 }}
        </div>
        <HR>
        <H1>2nd stage</H1>
        <div class="embed-wrapper">
            {{ div2 }}
        </div>
        <HR>
        <H1>3rd stage</H1>
        <div class="embed-wrapper">
            {{ div3 }}
        </div>
    </body>
</html>
''')

js_resources = INLINE.render_js()
css_resources = INLINE.render_css()
html = template.render(js_resources=js_resources,
                       css_resources=css_resources,
                       script1=script1,
                       script2=script2,
                       script3=script3,
                       div1=div1,
                       div2=div2,
                       div3=div3)

with io.open(filename, mode='w', encoding='utf-8') as f:
    f.write(html)

view(filename)
