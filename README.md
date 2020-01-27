# o-glassesX


o-glassesX is an intuitive binary classification and visualization tool with machine learning.

## Requirement

* An OS that can run Python 2.7.3 or later
* [Chainer](https://github.com/chainer/chainer) 4.0 or later
* matplotlib
* pillow
* distorm3

## Installation
Ubuntu 16.04
```
>sudo apt-get update
>sudo apt install python-pip
>pip install chainer
>pip install matplotlib
>pip install pillow
>pip install distorm3
```

## Usage
####  Mode1(data set validation:データセットの検証)
The following command shows an example of validating data set you prepare.
o-glassesX divides each file into L-instruction-block  and validate using k-fold cross validation.
```
>python o-glassesX.py -d path-to-data-set 
```
You can use the following option when running.
* The `-b` option specify batch-size[100].
* The `-e` option specify number of epoch[20].
* The `-k` option specify k-fold cross validation[3].
* The `-l` option specify the length of input instructions[16].
* The `-s` option specify the limitation of sample number of each label. [-1] (Negative value indicates no-limitation)
* The `-g` option specify GPU_id[-1]. (Negative value indicates using CPU)

e.g.,
When you want to validate the dataset using GPU (id:0), the command is following.
```
>python o-glassesX.py -d './dataset/compiler/' -k 4 -s 1000 -g 0 -e 5- -b 1000 -l 64
```

#### Mode2(building trained model:学習済モデルの生成)
In this case, o-glassesX build a trained model.
When it finished, (model name).json and (model_name).npz will be created.
```
>python o-glassesX.py -om model_name -i path-to-data-set
```
* The `-om` spcify output name of trained model.

#### Mode3(file estimation:ファイル推定)
The following is the example of cpu architecture estimation in accordance with trained model you specify.

学習済みモデルを使用し，ペイロードのCPUアーキテクチャの推定をする場合

```
> python o-glasses.py -im compiler -i ./payload.bin
[0, 20, 0, 2] win_gnu_gcc_32bit
```
* The `-im` option specify trained model(cpu.npz and cpu.json).
* The `-i` option specify a file for estimation check.
* The `--output_image` option outputs the estimation result as images (containing Structural Entropy)

* 「-im compiler」で学習済みモデル(compiler.npz, compiler.json)を読み込み
* 「-i ./payload.bin」でチェック対象File(./payload.bin）を読み込み


o-glassesX split input file into L-instruction-block and classify every block.
In default, it does every 1 byte slide.

[0, 20, 0, 2] shown in output the result of classification for every block.

出力結果の[0, 20, 0, 2]は各ブロックのクラス分類結果を示す．

In this case(この場合の判定結果は以下のとおり．), 

* No blocks are estimated as the first label.
* 20 blocks are estimated as the second label 'win_gnu_gcc_32bit'.
* No blocks are estimated as the third label.
* 2  blocks are estimated as the fourth label.

The most estimated label 'win_gnu_gcc_32bit' is displayed as a result of file estimation.

最終的に最も多く判定されたラベル「win_gnu_gcc_32bit」が表示される．

## Licence
Released under the MIT license  
http://opensource.org/licenses/mit-license.php

## Author

[yotsubo](https://github.com/yotsubo)
