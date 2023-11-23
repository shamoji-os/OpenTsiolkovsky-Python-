# OpenTsiolkovsky(Python)

# Japanese Section
## はじめに
本プログラムは、インターステラテクノロジズ株式会社殿が開発された「[OpenTsiolkovsky](https://github.com/istellartech/OpenTsiolkovsky)」をベースに、
C++でコーディングされたコアプログラムのPythonへの移植、リファクタリング等の変更を加えたプログラムです。

現在のところ、基本的に「OpenTsiolkovsky」のVer.0.41に準拠した動作を行います。
ただし、移植にあたり、クラス設計、プログラムの実行順序、データ形式等に大きな変更を加えているため、実行結果は完全に一致はしません。
M-Vロケット1号機をモデルとし、公開情報等から作成したインプットファイルをサンプルとして実行できるよう同梱してあります。
ベースとしたプログラム「OpenTsiolkovsky Ver0.41」との実行結果の比較も出来るよう、
飛翔解析の結果をoutputフォルダ以下の「【sample】M-V-1(for ver0.41 compared configuration)」に保管いたしましたので、あわせてご参照ください。
（Google Earth等にkmlファイルを読み込ませることによって、解析結果(飛行経路)の比較を行うことが出来ます。）

## 使い方
1. 本プログラムをGitよりクローン(複製)し、ローカル環境にダウンロードします。
2. コンソールを開き、本プログラムの保存されたディレクトリ直下(「OpenTsiolkovskyPy.py」が存在するディレクトリ)に移動します。
3. 同梱したサンプル(「M-V-1.json5」)を参考に、インプットファイルを作成します。パラメータの詳細等は同設定ファイルのコメントをご参照ください。
4. コンソールにて以下のコマンドを実行します。
	`python ./OpenTsiolkovskyPy.py "<作成したインプットファイルの名称>.json5"`
5. 支障なくプログラムの動作が完了した場合、実行結果がoutputフォルダ内に出力されます。

※ 正常に動作しない場合は以下をお試しください。
 * コンソール画面にてモジュールが不足している等の警告が出る場合は当該モジュールのインストールを実行してください。
 * 出力結果が何も出ない場合、いずれかのインプットファイルに構文違いやスペルミス等の誤りがある可能性があります。
   再度サンプルファイルを参考に入力ミス等がないかご確認ください。

## 解析結果からkmlファイル(GoogleEarthに読み込む用)およびサマリ(htmlファイル)を作成する方法について
1. コンソールを開き、本プログラムの保存されたディレクトリ直下(「OpenTsiolkovskyPy.py」が存在するディレクトリ)に移動します。
2.  コンソールにて以下のいずれかのコマンドを実行します。
	`python ./bin/make_html.py "<作成したインプットファイルの名称>.json5"`
	`python ./bin/make_kml.py "<作成したインプットファイルの名称>.json5"`
3. プログラムが正常に動作した場合、ファイルを選択するダイアログが２回開きます。以下の順序でファイルを選択してください。
	1回目の選択： 解析を実行する際に使用したインプットファイル(上記のプロンプトで指定したものと同じもの)を選択します。
				 (outputフォルダ内の解析結果を保存したフォルダ内にも実行時のインプットファイルのコピーが保存されています。)
	2回目の選択： 解析結果（「merged.csv」等）を選択します。
4. プログラムが正常に動作した場合、outputフォルダ内に作成されたkmlファイル、htmlファイルが出力されます。

## License
本プログラムのライセンスはMITライセンスに準拠します。


# English Translation(Google Translate)

## Introduction
This program is based on [OpenTsiolkovsky](https://github.com/istellartech/OpenTsiolkovsky) developed by Interstellar Technologies, Inc.,
and is a program in which the core program coded in C++ has been ported to Python, refactored, and other changes have been made.

Currently, the operation basically complies with "OpenTsiolkovsky" Ver.0.41.
However, due to the major changes made to the class design, program execution order,
data format, etc. during porting, the execution results will not match completely.
Modeled after M-V Rocket No. 1, an input file created from publicly available information is included so that it can be executed as a sample.
We have saved the flight analysis results in "【sample】 M-V-1 (for ver0.41 compared configuration)" under the output folder
so that you can compare the execution results with the base program "OpenTsiolkovsky Ver0.41". Therefore, please refer to it as well.
(By loading the kml file into Google Earth, etc., you can compare the analysis results (flight paths).)

## How to use
1. Clone (duplicate) this program from Git and download it to your local environment.
2. Open the console and move to the directory where this program is saved (the directory where "OpenTsiolkovskyPy.py" is located).
3. Create an input file using the included sample ("M-V-1.json5") as a reference. Please refer to the comments in the configuration file for details on the parameters.
4. Execute the following command on the console.
	`python ./OpenTsiolkovskyPy.py "<name of the input file you created>.json5"`
5. If the program completes without any problems, the execution results will be output to the output folder.

*If it does not work properly, please try the following.
  * If a warning appears on the console screen that a module is missing, etc., please install the module in question.
  * If there is no output result, there may be an error in one of the input files, such as a syntax error or a spelling error.
    Please refer to the sample file again and check for input errors.


## About how to create a kml file (for loading into Google Earth) and summary (html file) from the analysis results
1. Open the console and move to the directory where this program is saved (the directory where "OpenTsiolkovskyPy.py" is located).
2. Execute one of the following commands on the console.
	`python ./bin/make_html.py "<name of the input file you created>.json5"`
	`python ./bin/make_kml.py "<name of the input file you created>.json5"`
3. If the program runs normally, the file selection dialog will open twice. Please select the files in the following order.
	First selection: Select the input file used when running the analysis (the same one specified in the prompt above).
					 (A copy of the input file at runtime is also saved in the folder where the analysis results are saved in the output folder.)
	Second selection: Select the analysis result (such as "merged.csv").
4. If the program runs normally, the kml file and html file created in the output folder will be output.


## License
This program is an Open Source project licensed under the MIT License.

## 参考文献(References)
1) OpenTsiolkovsky: https://github.com/istellartech/OpenTsiolkovsky
2) M-V rocket F1 referenced data(folowing):
	2-1) Attitude: https://jaxa.repo.nii.ac.jp/record/33355/files/SA0200128.pdf
	2-2) Propulsion system: https://www.isas.jaxa.jp/publications/hokokuSP/hokokuSP47/85-116.pdf
	2-3) SOE: https://www.isas.jaxa.jp/j/enterp/rockets/vehicles/m-v/seq.shtml#
	2-4) Epoch, initial azimuth:https://www.isas.jaxa.jp/publications/hokokuSP/hokokuSP47/579-592.pdf
 * The sample data included as an input file for this program
   uses data digitized using Graphcel(https://www.vector.co.jp/soft/win95/business/se247204.html)
   from the above references.
