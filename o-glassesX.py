#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017-2020 Yuhei Otsubo
Released under the MIT license
http://opensource.org/licenses/mit-license.php
"""
#環境構築(インストール直後のUbuntu 16.04の場合)
# sudo apt-get update
# sudo apt install python-pip
# pip install chainer
# pip install matplotlib
# pip install pillow
# pip install distorm3
#
#基本的な使い方
#オプション無し：datasetディレクトリ(-d)のデータを使ってk-分割交差検証の実験
#-omオプション：学習済みモデルを出力
#-imオプション&-iオプション：学習済みモデルを使用し(-im)、入力ファイル(-i)の推定
#

check_dataset = False
output_image = True

import os
import sys
import commands
import json
import random
from chainer.datasets import tuple_dataset
from chainer import Variable
from chainer import serializers
import numpy as np
from PIL import Image
from distorm3 import Decode, Decode32Bits, Decode64Bits
import binascii

from operator import itemgetter
try:
    import matplotlib
    import sys
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import MyClassifier
import compressed_pickle as pickle
from multiprocessing import Pool

from sklearn.cluster import KMeans


        
# Simple Attention
class Attention(chainer.Chain):
        def __init__(self, length, depth):
                super(Attention, self).__init__()
                with self.init_scope():
                        self.l_q = L.ConvolutionND(1,depth, depth*4, 1)
                        self.l_k = L.ConvolutionND(1,depth, depth*4, 1)
                        self.l_v = L.ConvolutionND(1,depth, depth*4, 1)
                        self.l_q2 = L.ConvolutionND(1,depth*4, depth, 1)
                        self.l_k2 = L.ConvolutionND(1,depth*4, depth, 1)
                        self.l_v2 = L.ConvolutionND(1,depth*4, depth, 1)
                self.depth = depth
                self.length = length
                
        def __call__(self, Input, hidden = False):
                Memory = Input
                i_shape = Input.shape
                query = F.relu(self.l_q(Input))
                key = F.relu(self.l_k(Memory))
                value = F.relu(self.l_v(Memory))
                
                query = F.relu(self.l_q2(query))
                key = F.relu(self.l_k2(key))
                value = F.relu(self.l_v2(value))
                
                logit = F.matmul(key, query, transa=True)
                attention_weight = F.softmax(logit)
                
                attention_output = F.matmul(value, attention_weight)
                net_out = attention_output
                
                if hidden:
                        return net_out, attention_weight, query, key, value, attention_output
                                             
                
                return net_out
                
                
# Network definition
class MLP(chainer.Chain):

    def __init__(self, op_len, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.conv1=L.ConvolutionND(1,1, 96, 16*8, stride=16*8, pad=0)
            self.l3=L.Linear(None, n_out)  # n_units -> n_out
            self.bnorm1=L.BatchNormalization(96)
            self.Att1 = Attention(op_len,96)
        xp = self.xp
        self.pos_block = xp.array([])
    def __call__(self, x, hidden = False):
        h1 = F.relu(self.conv1(x))


        #Positional Encoding
        if self.pos_block.shape != h1.shape:
                xp = self.xp
                batch_size, depth, length = h1.shape
                
                channels = depth
                position = xp.arange(length, dtype='f')
                num_timescales = channels // 2
                log_timescale_increment = (
                    xp.log(10000. / 1.) /
                    (float(num_timescales) - 1))
                inv_timescales = 1. * xp.exp(
                    xp.arange(num_timescales).astype('f') * -log_timescale_increment)
                scaled_time = \
                    xp.expand_dims(position, 1) * \
                    xp.expand_dims(inv_timescales, 0)
                signal = xp.concatenate(
                    [xp.sin(scaled_time), xp.cos(scaled_time)], axis=1)
                signal = xp.reshape(signal, [1, length, channels])
                position_encoding_block = xp.transpose(signal, (0, 2, 1))
                self.pos_block = xp.tile(position_encoding_block, (batch_size,1,1))
        h1_ = h1 + self.pos_block*0.01

        if hidden:
                result = self.Att1(h1_,hidden)
                h3 = result[0]
                h = result[1:]
        else:
                h3 = self.Att1(h1_)
        h3 = self.bnorm1(h3)
        out_n = self.l3(h3)
        
        if hidden:
                return out_n, h ,h1
        else:
                return out_n
                



def fild_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            path = os.path.join(root, file)
            if os.path.islink(path):
                continue
            yield path


def get_result(result):
        max_i=0
        for i in range(len(result)):
                if result[max_i]<result[i]:
                        max_i=i
        return max_i

def bitmap_view(b):
        return b
        if b==0:
                r=0
        elif b<0x20:
                r=0x20
        elif b<0x80:
                r=0x80
        else:
                r=0xFF
        return r

def entropy(data):
    result = []
    s = len(data)
    for x in range(256):
        n = 0
        for i in data:
            if i == x:
                n+=1
        p_i = float(n)/s
        if p_i != 0:
                result.append(p_i * np.log2(p_i))
    r = 0.0
    for i in result:
        if i == i:
            #NaNでないときの処理
            r += i
    return np.int32((-r)/8.0*255.0)

def show_info_dataset(dataset):
        n = len(dataset)
        l = {}
        for t in dataset:
                if not l.has_key(t[2]):
                        l[t[2]] = 1
                else:
                        l[t[2]] += 1
        print n,l

def cos_sim(xp, a, b):
        #return xp.linalg.norm((a-b).data)
        return np.dot(a,b).data / (xp.linalg.norm(a.data)*xp.linalg.norm(b.data))
        
def main():
        parser = argparse.ArgumentParser(description='Chainer: eye-grep test')
        parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
        parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
        parser.add_argument('--k', '-k', type=int, default=3,
                        help='Number of folds (k-fold cross validation')
        parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
        parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
        parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
        parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
        parser.add_argument('--unit', '-u', type=int, default=400,
                        help='Number of units')
        parser.add_argument('--length', '-l', type=int, default=16,
                        help='Number of instruction')
        parser.add_argument('--dataset', '-d', type=str, default="dataset",
                        help='path of dataset')
        parser.add_argument('--input', '-i', type=str, default="",
                        help='checked file name')
        parser.add_argument('--input_mode', '-imode', type=int, default=0,
                        help='check file mode, 0:all, 1:head,2:middle,3:last')
        parser.add_argument('--output_model', '-om', type=str, default="",
                        help='model file path')
        parser.add_argument('--input_model', '-im', type=str, default="",
                        help='model file name')
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--disasm_x86', action='store_true')
        group.add_argument('--no-disasm_x86', action='store_false')
        parser.set_defaults(disasm_x86=True)
        parser.add_argument('--s_limit', '-s', type=int, default=-1,
                        help='Limitation of Sample Number  (negative value indicates no-limitation)')

        #for output image
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--output_image', action='store_true')
        parser.set_defaults(output_image=False)

        args = parser.parse_args()
        output_image = args.output_image

        #入力オペコードの数
        op_num = args.length#16
        block_size = 16*op_num
        #SGD,MomentumSGD,AdaGrad,RMSprop,AdaDelta,Adam
        #selected_optimizers = chainer.optimizers.Adam()
        selected_optimizers = chainer.optimizers.SGD(lr=0.01)
        

        if not args.input_model:
                #datasetディレクトリから学習モデルを作成

                path = args.dataset
                print path

                #ファイル一覧の取得

                files_file = [f for f in fild_all_files(path) if os.path.isfile(os.path.join(f))]
                files_file.sort()

                #ファイルタイプのナンバリング
                file_types = {}
                file_types_ = []
                num_of_file_types = {}
                num_of_types = 0
                for f in files_file:
                        #ディレクトリ名でファイルタイプ分類
                        file_type = f.replace(path,"").replace(os.path.basename(f),"").split("/",1)[0]
                        #print(file_type)
                        if file_type in file_types:
                                num_of_file_types[file_type] += 1
                        else:
                                file_types[file_type]=num_of_types
                                file_types_.append(file_type)
                                num_of_file_types[file_type] = 1
                                print num_of_types,file_type
                                num_of_types+=1

                #データセットの作成
                print "make dataset"
                BitArray = [[int(x) for x in format(y,'08b')] for y in range(256)]
                num_of_dataset = {}
                master_dataset = []
                master_dataset_b = []
                order_l = [[0 for i in range(32)] for j in range(num_of_types)]
                random.shuffle(files_file)
                for f in files_file:
                        ft = f.replace(path,"").replace(os.path.basename(f),"").split("/",1)[0]
                        if ft not in num_of_dataset:
                                num_of_dataset[ft] = 0
                        if args.s_limit > 0 and num_of_dataset[ft] >= args.s_limit:
                                continue
                        ftype = np.int32(file_types[ft])
                        fin = open(f,"rb")
                        bdata = fin.read()
                        if args.disasm_x86:
                                l = Decode(0x4000000, bdata, Decode64Bits)
                                #16バイトで命令を切る
                                lengths = [i[1] for i in l]
                                pos = 0
                                b = b''
                                for l in lengths:
                                        if l>16:
                                                b += bdata[pos:pos+16]
                                        else:
                                                b += bdata[pos:pos+l]+b'\0'*(16-l)
                                        order_l[ftype][l]+=1
                                        pos += l

                                #l = Decode(0x4000000, bdata, Decode32Bits)
                                ##16バイトで命令を切る
                                #lengths = [i[1] for i in l]
                                #pos = 0
                                #for l in lengths:
                                #        if l>16:
                                #                b += bdata[pos:pos+16]
                                #        else:
                                #                b += bdata[pos:pos+l]+b'\0'*(16-l)
                                #        order_l[ftype][l]+=1
                                #        pos += l

                                bdata = b
                        fsize = len(bdata)
                        if fsize < block_size:
                                continue

                        #block_size(256バイト)区切りでデータセット作成
                        for c in range(0,fsize-block_size,block_size):
                                if args.s_limit > 0 and num_of_dataset[ft] >= args.s_limit:
                                        break
                                offset = c*1.0/fsize
                                block = bdata[c:c+block_size]
                                train = []
                                #1 Byte to 8 bit-array
                                for x in block:
                                        train.extend(BitArray[ord(x)])
                                train = np.asarray([train],dtype=np.float32)
                                train = (train,ftype)
                                master_dataset.append(train)
                                master_dataset_b.append((block,ftype))
                                num_of_dataset[ft]+=1


                #データセットの情報を表示
                total_samples = 0
                total_files = 0
                total_types = 0
                print "label","File","Code","1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
                for t in file_types_:
                        print t,
                        print num_of_file_types[t],
                        print num_of_dataset[t],
                        total_types+=1
                        total_files+=num_of_file_types[t]
                        total_samples+=num_of_dataset[t]
                        if args.disasm_x86:
                                for j in range(1,16+1):
                                        print order_l[file_types[t]][j],
                        print ""
                        
                print "total types", total_types
                print "total files", total_files
                print "total samples", total_samples

                #データセットのダブリをチェック
                if check_dataset:
                        print "Dataset Duplication"
                        master_dataset_b.sort(key=lambda x: x[0])
                        checked_list = [False for i in range(total_samples)]
                        Duplication_list = [[0 for i in range(total_types)] for j in range(total_types)]
                        for i in range(total_samples):
                                if checked_list[i]:
                                        continue
                                d_list = [False]*total_types
                                (train1,ftype1) = master_dataset_b[i]
                                d_list[ftype1] = True
                                d = 0
                                for j in range(i,total_samples):
                                        (train2, ftype2) = master_dataset_b[j]
                                        if train1 == train2:
                                                d_list[ftype2] = True
                                                d += 1
                                        else:
                                                break
                                d_num = 0
                                for t in d_list:
                                        if t:
                                                d_num += 1
                                for j in range(d):
                                        (train2, ftype2) = master_dataset_b[i+j]
                                        Duplication_list[ftype2][d_num-1] += 1
                                        checked_list[i+j] = True
                                        
                                                
                        for t in file_types_:
                                print t,
                                for j in range(total_types):
                                        print Duplication_list[file_types[t]][j],
                                print ""

                print('GPU: {}'.format(args.gpu))
                print('# unit: {}'.format(args.unit))
                print('# Minibatch-size: {}'.format(args.batchsize))
                print('# epoch: {}'.format(args.epoch))
                print('')
                        
        else:
                #学習済みモデルの入力
                f = open(args.input_model+".json","r")
                d = json.load(f)
                file_types_ = d['file_types_']
                num_of_types = d['num_of_types']
                #model = MyClassifier.MyClassifier(MLP(d['unit'], num_of_types))
                model = MyClassifier.MyClassifier(MLP(op_num, num_of_types))
                serializers.load_npz(args.input_model+".npz", model)
                if args.gpu >= 0:
                        chainer.cuda.get_device_from_id(args.gpu).use()  # Make a specified GPU current
                        model.to_gpu()  # Copy the model to the GPU
        if args.output_model and master_dataset:
                #master_datasetが作成されていない場合、学習済みモデルは出力されない
                #学習済みモデルの作成
                # Set up a neural network to train
                # Classifier reports softmax cross entropy loss and accuracy at every
                # iteration, which will be used by the PrintReport extension below.
                #model = MyClassifier.MyClassifier(MLP(args.unit, num_of_types))
                model = MyClassifier.MyClassifier(MLP(op_num, num_of_types))
                if args.gpu >= 0:
                        chainer.cuda.get_device_from_id(args.gpu).use()  # Make a specified GPU current
                        model.to_gpu()  # Copy the model to the GPU

                # Setup an optimizer
                optimizer = selected_optimizers
                optimizer.setup(model)

                train_iter = chainer.iterators.SerialIterator(master_dataset, args.batchsize)
                updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
                trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

                # Dump a computational graph from 'loss' variable at the first iteration
                # The "main" refers to the target link of the "main" optimizer.
                trainer.extend(extensions.dump_graph('main/loss'))

                # Write a log of evaluation statistics for each epoch
                trainer.extend(extensions.LogReport())

                # Save two plot images to the result dir
                if extensions.PlotReport.available():
                        trainer.extend(
                            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                                  'epoch', file_name='loss.png'))
                        trainer.extend(
                            extensions.PlotReport(
                                ['main/accuracy', 'validation/main/accuracy'],
                                'epoch', file_name='accuracy.png'))

                # Print selected entries of the log to stdout
                # Here "main" refers to the target link of the "main" optimizer again, and
                # "validation" refers to the default name of the Evaluator extension.
                # Entries other than 'epoch' are reported by the Classifier link, called by
                # either the updater or the evaluator.
                trainer.extend(extensions.PrintReport(
                ['epoch', 'main/loss', 'validation/main/loss',
                 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

                # Print a progress bar to stdout
                trainer.extend(extensions.ProgressBar())

                # Run the training
                trainer.run()

                #学習済みモデルの出力
                d={}
                d['file_types_'] = file_types_
                d['unit'] = args.unit
                d['num_of_types'] = num_of_types
                f = open(args.output_model+".json","w")
                json.dump(d,f)
                model.to_cpu()
                serializers.save_npz(args.output_model+".npz",model)

        elif args.input:
                if not args.input_model:
                        #学習済みデータセットが指定されていない場合
                        return
                #解析対象のデータセットの作成
                BitArray = [[int(x) for x in format(y,'08b')] for y in range(256)]
                checked_dataset = []
                f=args.input
                basename = os.path.basename(f)
                fin = open(f,"rb")
                bdata = fin.read()
                if args.input_mode == 1:
                        bdata = bdata[:4096]
                elif args.input_mode == 2:
                        middle = int(len(bdata)/2)
                        bdata = bdata[middle-2048:middle+2048]
                elif args.input_mode == 3:
                        bdata = bdata[-4096:]
                fsize = len(bdata)
                h=(fsize+127)/128
                max_h = 1024
                img = Image.new('RGB', (128, h))
                for i in range(0,fsize):
                        b = ord(bdata[i])
                        if b == 0x00:
                                c=(255,255,255)
                        elif b < 0x20:
                                c=(0,255,255)
                        elif b<0x80:
                                c=(255,0,0)
                        else:
                                c=(0,0,0)
                        img.putpixel((i%128,i/128),c)
                if output_image:
                    for num in range(0,(h-1)/max_h+1):
                            box = (0,num*max_h,128,num*max_h+max_h)
                            img.crop(box).save(basename+"_bitmap_"+"{0:04d}".format(num)+".png")
                    box = (0,num*max_h,128,h)
                    img.crop(box).save(basename+"_bitmap_"+"{0:04d}".format(num)+".png")
                    img.save(basename+"_bitmap.png")
                    #img.show()


                #256バイト区切りでデータセット作成
                #print args.input
                col = [[#for 19 classification
                        (255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),#VC
                        (0,255,0),(0,255,0),(0,255,0),(0,255,0),#gcc
                        (0,0,255),(0,0,255),(0,0,255),(0,0,255),#clang
                        (255,0,255),(255,0,255),(255,0,255),(255,0,255),#icc
                        (255,255,0),(255,0,255),(0,255,255)],
                        [
                        (255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),#VC
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        [
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (0,255,0),(0,255,0),(0,255,0),(0,255,0),#gcc
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        [
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (0,0,255),(0,0,255),(0,0,255),(0,0,255),(255,255,255),#clang
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        [
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,0,255),(255,0,255),(255,0,255),(255,0,255),(255,255,255),#icc
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        [
                        (255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),#VC
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        [
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,0,0),(0,255,0),(0,0,255),(255,255,0),#gcc
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        [
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,0,0),(0,255,0),(0,0,255),(255,255,0),#clang
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        [
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,0,0),(0,255,0),(0,0,255),(255,255,0),#icc
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        [
                        (255,0,0),(255,0,0),(0,255,0),(0,255,0),(255,255,255),(255,255,255),#VC for 32bit
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255),
                        (255,255,255),(255,255,255),(255,255,255),(255,255,255)],
                        ]
                img_ = Image.new('RGB', (128, h))
                #解析対象のファイルの分類結果を表示
                img = [Image.new('RGB', (128, h)) for i in range(len(col))]
                l=1
                results = [0 for i in range(num_of_types)]
                i_=0
                num=0
                asm = {}
                for c in range(0,fsize-block_size+1,l):
                        offset = c*1.0/fsize
                        block = bdata[c:c+block_size]
                        block_ = [ord(x) for x in block]
                        e = entropy(block_)
                        for j in range(0,l):
                                img_.putpixel(((c+j)%128,(c+j)/128),(e,e,e))
                        if args.disasm_x86:
                                m = Decode(0x4000000+c, block, Decode64Bits)
                                block = b''
                                for i in m:
                                        b = b''
                                        for c_ in range(16):#16バイトで命令を切る
                                                if c_ < len(i[3])/2:
                                                        b += chr(int(i[3][c_*2:c_*2+2],16))
                                                else:
                                                        b += b'\0'
                                        block += b
                                block = block[:block_size]

                        train = []
                        for x in block:
                                train.extend(BitArray[ord(x)])
                        train = np.asarray([train],dtype=np.float32)
                        if args.gpu >= 0:
                                xp = chainer.cuda.cupy
                        else:
                                xp = np
                        with chainer.using_config('train', False):
                                result = model.predictor(xp.array([train]).astype(xp.float32),hidden = True)
                                result2 = int(result[0].data.argmax(axis=1)[0])
                                result3 = F.softmax(result[0])[0][result2].data

                                results[result2]+=1
                                if False and result3 > 0.99 and file_types_[result2] in args.input:
                                        results[result2]+=1
                                        
                                        attention_weight = result[1][0][0]
                                        l2 = F.batch_l2_norm_squared(attention_weight)
                                        result4 = int(xp.argmax(l2.data))
                                        ai = result4
                                        if m[ai][2] in asm:
                                                asm[m[ai][2]]+=1
                                        else:
                                                asm[m[ai][2]]=1
                        for j in range(0,l):
                                for i in range(len(col)):
                                        img[i].putpixel(((i_*l+j)%128,(i_*l+j)/128),col[i][result2])
                        i_+=1
                        if output_image:
                            if (i_%128) == 0:
                                    box = (0,num*max_h,128,num*max_h+max_h)
                                    img_.crop(box).save(basename+"_entropy_"+"{0:04d}".format(num)+".png")
                                    for i in range(len(col)):
                                            img[i].crop(box).save(basename+"_v_"+"{0:02d}_".format(i)+"{0:04d}".format(num)+".png")
                            if (i_*l)%(128*max_h) == 0:
                                    print i_,"/",fsize
                                    box = (0,num*max_h,128,num*max_h+max_h)
                                    img_.crop(box).save(basename+"_entropy_"+"{0:04d}".format(num)+".png")
                                    for i in range(len(col)):
                                            img[i].crop(box).save(basename+"_v_"+"{0:02d}_".format(i)+"{0:04d}".format(num)+".png")
                                    num+=1
                print results,file_types_[get_result(results)]
                for k, v in sorted(asm.items(), key = lambda x: -x[1]):
                        print '"'+str(k)+'" '+str(v)
                if output_image:
                    box = (0,num*max_h,128,h)
                    img_.crop(box).save(basename+"_entropy_"+"{0:04d}".format(num)+".png")
                    for i in range(len(col)):
                            img[i].crop(box).save(basename+"_v_"+"{0:02d}_".format(i)+"{0:04d}".format(num)+".png")
                            img[i].save(basename+"_v_"+"{0:02d}_".format(i)+".png")
                    img_.save(basename+"_entropy.png")
                    #img.show()
        else:

                #k-分割交差検証
                random.shuffle(master_dataset)
                k=args.k
                mtp = [0 for j in range(num_of_types)]
                mfp = [0 for j in range(num_of_types)]
                mfn = [0 for j in range(num_of_types)]
                mtn = [0 for j in range(num_of_types)]
                mftn = [0 for j in range(num_of_types)]
                mrs = [[0 for i in range(num_of_types)] for j in range(num_of_types)]
                for i in range(k):
                        pretrain_dataset = []
                        train_dataset = []
                        test_dataset = []
                        flag = True
                        #各クラスの比率を維持
                        c = [0 for j in range(num_of_types)]
                        for train in master_dataset:
			        ft = train[1]
			        totalsamples = num_of_dataset[file_types_[ft]]
                                if c[ft]<totalsamples*i/k:
                                        train_dataset.append(train)
                                elif c[ft]>=totalsamples*(i+1)/k:
                                        train_dataset.append(train)
                                else:
                                        test_dataset.append(train)
                                c[ft]+=1
                        c2 = [0 for j in range(num_of_types)]
                        for train in train_dataset:
                                ft = train[1]
                                if c2[ft] < c[ft]/2:
                                        pretrain_dataset.append(train)
                                c2[ft]+=1

                        random.shuffle(train_dataset)

                        model = MyClassifier.MyClassifier(MLP(op_num, num_of_types))
                        if args.gpu >= 0:
                                chainer.cuda.get_device_from_id(args.gpu).use()  # Make a specified GPU current
                                model.to_gpu()  # Copy the model to the GPU

                        # Setup an optimizer
                        optimizer = selected_optimizers
                        optimizer.setup(model)

                        if args.gpu >= 0:
                                xp = chainer.cuda.cupy
                        else:
                                xp = np


                        train_iter = chainer.iterators.SerialIterator(pretrain_dataset, args.batchsize)
                        test_iter = chainer.iterators.SerialIterator(test_dataset, args.batchsize,
                                                                 repeat=False, shuffle=False)
                        updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
                        trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out+"{0:02d}".format(i))
                        trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
                        trainer.extend(extensions.dump_graph('main/loss'))
                        trainer.extend(extensions.LogReport())
                        # Save two plot images to the result dir
                        if extensions.PlotReport.available():
                                trainer.extend(
                                    extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                                          'epoch', file_name='loss.png'))
                                trainer.extend(
                                    extensions.PlotReport(
                                        ['main/accuracy', 'validation/main/accuracy'],
                                        'epoch', file_name='accuracy.png'))

                        trainer.extend(extensions.PrintReport(
                        ['epoch',
                         'main/loss', 'validation/main/loss',
                         'main/accuracy', 'validation/main/accuracy',
                         'elapsed_time']))
                        trainer.extend(extensions.ProgressBar())
                        trainer.run()

                                
                              
                        #結果を集計

                        if args.gpu >= 0:
                                xp = chainer.cuda.cupy
                        else:
                                xp = np
                        tp = [0 for j in range(num_of_types)]
                        fp = [0 for j in range(num_of_types)]
                        fn = [0 for j in range(num_of_types)]
                        tn = [0 for j in range(num_of_types)]
                        ftn = [0 for j in range(num_of_types)]
                        rs = [[0 for j2 in range(num_of_types)] for j in range(num_of_types)]
                        for train in test_dataset:
	                        ft = train[1]
	                        totalsamples = num_of_dataset[file_types_[ft]]
                                with chainer.using_config('train', False):
                                        result = int(model.predictor(xp.array([train[0]]).astype(xp.float32)).data.argmax(axis=1)[0])
                                if ft == result:
                                        tp[ft] += 1
                                        tn[result] += 1
                                        mtp[ft] += 1
                                        mtn[result] += 1
                                else:
                                        fp[ft] += 1
                                        fn[result] += 1
                                        mfp[ft] += 1
                                        mfn[result] += 1
                                ftn[ft] += 1
                                rs[ft][result]+=1
                                mftn[ft] += 1
                                mrs[ft][result]+=1
                                        
                                #print ft,result
                        print "",
                        for t in file_types_:
                                print t,
                        print
                        for t in file_types_:
                                print t,
                                for j in range(num_of_types):
                                        print rs[file_types[t]][j],
                                print
                        print "no label Num TP FP FN TN R P F1 Acc."
                        for t in file_types_:
                                ft = file_types[t]
                                print ft,
                                print t,
                                print ftn[ft],
                                print tp[ft],fp[ft],fn[ft],tn[ft],
                                if tp[ft]+fn[ft] != 0:
                                        r = float(tp[ft])/(tp[ft]+fn[ft])
                                else:
                                        r = 0.0
                                print r,
                                if tp[ft]+fp[ft] != 0:
                                        p = float(tp[ft])/(tp[ft]+fp[ft])
                                else:
                                        p = 0.0                        
                                print p,
                                if r+p != 0:
                                        f1 = 2*r*p/(r+p)
                                else:
                                        f1 = 0.0
                                print f1,
                                acc = float(tp[ft]+tn[ft])/(tp[ft]+fp[ft]+fn[ft]+tn[ft])
                                print acc
                for t in file_types_:
                        print t,
                print
                for t in file_types_:
                        print t,
                        for j in range(num_of_types):
                                print mrs[file_types[t]][j],
                        print
                print "no label Num TP FP FN TN R P F1 Acc."
                for t in file_types_:
                        ft = file_types[t]
                        print ft,
                        print t,
                        print mftn[ft],
                        print mtp[ft],mfp[ft],mfn[ft],mtn[ft],
                        if mtp[ft]+mfn[ft] != 0:
                                r = float(mtp[ft])/(mtp[ft]+mfn[ft])
                        else:
                                r = 0.0
                        print r,
                        if mtp[ft]+mfp[ft] != 0:
                                p = float(mtp[ft])/(mtp[ft]+mfp[ft])
                        else:
                                p = 0.0                        
                        print p,
                        if r+p != 0:
                                f1 = 2*r*p/(r+p)
                        else:
                                f1 = 0.0
                        print f1,
                        acc = float(mtp[ft]+mtn[ft])/(mtp[ft]+mfp[ft]+mfn[ft]+mtn[ft])
                        print acc
                sum_mftn = sum(mftn)
                sum_mtp = sum(mtp)
                sum_mfp = sum(mfp)
                sum_mfn = sum(mfn)
                sum_mtn = sum(mtn)
                print '','',sum_mftn,sum_mtp,sum_mfp,sum_mfn,sum_mtn,
                if sum_mtp+sum_mfn != 0:
                        r = float(sum_mtp)/(sum_mtp+sum_mfn)
                else:
                        r = 0.0
                print r,
                if sum_mtp+sum_mfp != 0:
                        p = float(sum_mtp)/(sum_mtp+sum_mfp)
                else:
                        p = 0.0                        
                print p,
                if r+p != 0:
                        f1 = 2*r*p/(r+p)
                else:
                        f1 = 0.0
                print f1,
                acc = float(sum_mtp+sum_mtn)/(sum_mtp+sum_mfp+sum_mfn+sum_mtn)
                print acc

if __name__ == '__main__':
    main()
