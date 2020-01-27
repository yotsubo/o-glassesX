# -*- coding: utf-8 -*-i
#20191108 ソースのディレクトリ構造を再現してデータセットを作成する用に修正
#20200120 exe形式のx64用コードをマージ

import sys
import os
import struct
import platform

path = ""
exportpath = ""

if len(sys.argv) != 3:
    print("[!]Error:引数として、オブジェクトファイルが存在するフォルダと出力先のフォルダのパスを入力してください。\n<How to use>")
    print("python3 elf_coff2bin.py [source code dirpath] [output dirpath]")
    print("Ex)python3 elf_coff2bin.py C:\\test C:\\export")
    exit()
else:
    path = sys.argv[1]
    exportpath = sys.argv[2]

if not os.path.exists(sys.argv[2]):
    os.mkdir(sys.argv[2])
    print("[*]CREATE-DIR:" + sys.argv[2])

files = []

splitter = ""

if platform.system() == "Windows":
    splitter = "\\"
else:
    splitter = "/"

if os.path.exists(exportpath) == False:
    os.mkdir(exportpath)

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)

for file in find_all_files(path):
    #ファイル以外はPASS
    if os.path.isdir(file):
        continue
    #filename = file.split(splitter)
    #filename = filename[-1]
    filename = os.path.basename(file)
    dirname = os.path.dirname(file).split(path)[-1]
    wfilename = os.path.join(exportpath,dirname,filename+".bin")
    #EXEの場合 32bitのみで試験済み
    if file[-4:] == ".exe":
        ##EXE(PE)##_____________________________________________
        f = open(file, "rb")
        sign = f.read(2)
        if sign !=  b"MZ":
            f.close()
            continue
        f.seek(0x04)
        signi = struct.unpack("<b",f.read(1))[0]

        #x64
        if signi == 0x02:
            f.seek(0x3C)
            e_lfanew = struct.unpack("<I", f.read(4))[0]
            f.seek(e_lfanew+6)
            num_of_sections = struct.unpack("<H", f.read(2))[0]
            f.seek(e_lfanew+20)
            size_of_optionalheader = struct.unpack("<H", f.read(2))[0]
            f.seek(e_lfanew+24+size_of_optionalheader)
            sectionpos = 0
            size_of_rawdata = 0
            #Extract NativeCode(.text)_________________________________________
            for i in range(0,num_of_sections):
                sectionname = f.read(5)
                if(sectionname == b'.text' or sectionname == b'.TEXT'):
                    f.read(3)
                    size_of_rawdata = struct.unpack("<I", f.read(4))[0]
                    f.read(8)
                    sectionpos = struct.unpack("<I", f.read(4))[0]
                    break
                f.read(35)
            if sectionpos != 0 and size_of_rawdata != 0:
                f.seek(sectionpos)
                nativecode = f.read(size_of_rawdata)
                if not os.path.isdir(os.path.dirname(wfilename)):
                    os.makedirs(os.path.dirname(wfilename))
                wf = open(wfilename,"wb")
                wf.write(nativecode)
                wf.close()
            else:
                print("[!]ERROR_.text_section_not_found_:"+file)
        else:
            f.seek(0x3C)
            e_lfanew = struct.unpack("<I", f.read(4))[0]
            f.seek(e_lfanew+6)
            num_of_sections = struct.unpack("<H", f.read(2))[0]
            f.seek(e_lfanew+20)
            size_of_optionalheader = struct.unpack("<H", f.read(2))[0]
            f.seek(e_lfanew+24+size_of_optionalheader)
            sectionpos = 0
            size_of_rawdata = 0
            #Extract NativeCode(.text)_________________________________________
            for i in range(0,num_of_sections):
                sectionname = f.read(5)
                if(sectionname == b'.text' or sectionname == b'.TEXT'):
                    f.read(3)
                    size_of_rawdata = struct.unpack("<I", f.read(4))[0]
                    f.read(8)
                    sectionpos = struct.unpack("<I", f.read(4))[0]
                    break
                f.read(35)
            if sectionpos != 0 and size_of_rawdata != 0:
                f.seek(sectionpos)
                nativecode = f.read(size_of_rawdata)
                wf = open(exportpath + splitter + filename +".bin","wb")
                wf.write(nativecode)
                wf.close()
            else:
                print("[!]ERROR_.text_section_not_found_:"+file)
            f.close()

    #(windows)OBJ FILE
    elif file[-4:] == ".obj":
        print("[*]LOG:"+file)
        f = open(file,"rb")
        f.seek(2)
        num_of_sections = struct.unpack("<H", f.read(2))[0]
        sectionpos = 0
        size_of_rawdata = 0
        if not os.path.isdir(os.path.dirname(wfilename)):
            os.makedirs(os.path.dirname(wfilename))
        fw = open(wfilename,"wb")
        fw.close()
        f.seek(20)
        #Extract NativeCode(.text)_________________________________________
        for i in range(0,num_of_sections):
            ifpos = f.tell()
            table_str = f.read(9)
            sectionname = table_str[0:8]
            sectionname2 = sectionname[0:5]
            f.seek(ifpos+8)
            #@Todo:f.readをseekのように使っているのは後で修正
            if ((sectionname == b'.text$mn' or sectionname == b'.TEXT$mn') and (table_str[8] == 0x00)):
                f.read(8)
                size_of_rawdata = struct.unpack("<I",f.read(4))[0]
                sectionpos = struct.unpack("<I",f.read(4))[0]
                ifpos = f.tell()
                f.seek(sectionpos)
                nativecode = f.read(size_of_rawdata)
                if not os.path.isdir(os.path.dirname(wfilename)):
                    os.makedirs(os.path.dirname(wfilename))
                wf = open(wfilename,"ab")
                wf.write(nativecode)
                wf.close()
                f.seek(ifpos + 16)
            elif ((sectionname2 == b'.text' or sectionname2 == b'.TEXT') and (table_str[8] == 0x00 )):
                f.read(8)
                size_of_rawdata = struct.unpack("<I",f.read(4))[0]
                sectionpos = struct.unpack("<I",f.read(4))[0]
                ifpos = f.tell()
                f.seek(sectionpos)
                nativecode = f.read(size_of_rawdata)
                if not os.path.isdir(os.path.dirname(wfilename)):
                    os.makedirs(os.path.dirname(wfilename))
                wf = open(wfilename,"ab")
                wf.write(nativecode)
                wf.close()
                f.seek(ifpos + 16)
            else:
                f.seek(ifpos+40)
        if sectionpos != 0 and size_of_rawdata != 0:
            print("[*]EXTRACT->OK:"+file)
        else:
            print("[!]ERROR_.text_section_not_found_:"+file)
        f.close()

    ##ELF_FILE(LINUX)2019/01/31##
    elif file[-2:] == ".o":
        try:
            f = open(file, "rb")
        except:
            continue
        print("[*]LOG:" + file)
        f.seek(1)
        sign = f.read(3)
        if sign != b"ELF":
            f.close()
            continue
        signi = struct.unpack("<b",f.read(1))[0]
        #32bit
        if signi == 0x01:
            sectionpos = 0
            size_of_rawdata = 0
            f.seek(0x20)
            section_header_table_offset = struct.unpack("<I", f.read(4))[0]
            f.seek(0x2E)
            sec_header_size = struct.unpack("<H", f.read(2))[0]
            f.seek(0x30)
            sec_header_count = struct.unpack("<H", f.read(2))[0]
            f.seek(0x32)
            sec_string_header_index = struct.unpack("<H", f.read(2))[0]
            x_string_header_index_offset = section_header_table_offset + (sec_header_size * sec_string_header_index)
            f.seek(x_string_header_index_offset)
            f.read(0x10)
            sec_string_header_offset = struct.unpack("<I",f.read(4))[0]
            sec_string_header_size = struct.unpack("<I",f.read(4))[0]
            s_counter = 0
            for z in range(0,sec_string_header_size):
                f.seek(sec_string_header_offset+z)
                #sign = f.read(5)
                nativecode_index = 0
                ifpos = f.tell()
                table_str = f.read(6)
                sectionname = table_str[0:5]
                f.seek(ifpos+5)

                if (sectionname == b".text" or sectionname == b".TEXT") and (table_str[5] == 0x00) :
                    nativecode_index = z
                    rtn_file_position = f.tell()
                    nativecode_size = 0
                    nativecode_offset = 0
                    for j in range(0,sec_header_count):
                        f.seek(section_header_table_offset + (sec_header_size * j))
                        str_sign = struct.unpack("<I",f.read(4))[0]
                        if nativecode_index == str_sign:
                            #EXTRACT_DATA
                            f.read(12)
                            nativecode_offset = struct.unpack("<I",f.read(4))[0]
                            nativecode_size = struct.unpack("<I",f.read(4))[0]
                            if nativecode_offset != 0 and nativecode_size != 0:
                                if not os.path.isdir(os.path.dirname(wfilename)):
                                    os.makedirs(os.path.dirname(wfilename))
                                wf = open(wfilename,"ab")
                                f.seek(nativecode_offset)
                                nativecode = f.read(nativecode_size)
                                wf.write(nativecode)
                                wf.close()
                                break
                            else:
                                print("[!]ERROR:" + file)
        #64bit
        elif signi == 0x02:
            sectionpos = 0
            size_of_rawdata = 0
            # Section Header Offset
            f.seek(0x28)
            section_header_table_offset = struct.unpack("<Q", f.read(8))[0]
            f.seek(0x3A)
            sec_header_size = struct.unpack("<H", f.read(2))[0]
            f.seek(0x3C)
            sec_header_count = struct.unpack("<H", f.read(2))[0]
            f.seek(0x3E)
            sec_string_header_index = struct.unpack("<H", f.read(2))[0]
            x_string_header_index_offset = section_header_table_offset + (sec_header_size * sec_string_header_index)
            f.seek(x_string_header_index_offset)
            f.read(0x18)
            sec_string_header_offset = struct.unpack("<Q",f.read(8))[0]
            sec_string_header_size = struct.unpack("<Q",f.read(8))[0]
            s_counter = 0
            for z in range(0,sec_string_header_size):
                f.seek(sec_string_header_offset+z)
                #sign = f.read(5)
                nativecode_index = 0
                ifpos = f.tell()
                table_str = f.read(6)
                sectionname = table_str[0:5]
                f.seek(ifpos+5)
                if  (sectionname == b".text" or sectionname == b".TEXT") and (table_str[5] == 0x00) :
                    nativecode_index = z
                    rtn_file_position = f.tell()
                    nativecode_size = 0
                    nativecode_offset = 0
                    for j in range(0,sec_header_count):
                        f.seek(section_header_table_offset + (sec_header_size * j))
                        str_sign = struct.unpack("<I",f.read(4))[0]
                        if nativecode_index == str_sign:
                            #EXTRACT_DATA
                            f.read(0x14)
                            nativecode_offset = struct.unpack("<Q",f.read(8))[0]
                            nativecode_size = struct.unpack("<Q",f.read(8))[0]
                            if nativecode_offset != 0 and nativecode_size != 0:
                                if not os.path.isdir(os.path.dirname(wfilename)):
                                    os.makedirs(os.path.dirname(wfilename))
                                wf = open(wfilename,"ab")
                                f.seek(nativecode_offset)
                                nativecode = f.read(nativecode_size)
                                wf.write(nativecode)
                                wf.close()
                                break
                            else:
                                print("[!]ERROR:" + file)
                                #print(nativecode_offset,nativecode_size,j,sec_header_count)
        f.close()
    else:
        try:
            f = open(file, "rb")
        except:
            continue
        print("[*]LOG:" + file)
        f.seek(1)
        sign = f.read(3)
        if sign != b"ELF":
            f.close()
            continue
        signi = struct.unpack("<b",f.read(1))[0]

        if signi == 0x01:
            sectionpos = 0
            size_of_rawdata = 0
            # Section Header Offset
            f.seek(0x20)
            section_header_table_offset = struct.unpack("<I", f.read(4))[0]
            f.seek(0x2E)
            sec_header_size = struct.unpack("<H", f.read(2))[0]
            f.seek(0x30)
            sec_header_count = struct.unpack("<H", f.read(2))[0]
            f.seek(0x32)
            sec_string_header_index = struct.unpack("<H", f.read(2))[0]

            for i in range(0,sec_header_count):
                f.seek(section_header_table_offset + (sec_header_size * i))
                s_name = struct.unpack("<I",f.read(4))[0]
                s_type = struct.unpack("<I",f.read(4))[0]
                if s_name == 1 and s_type == 3:
                    f.read(8)
                    sec_string_header_offset = struct.unpack("<I",f.read(4))[0]
                    sec_string_header_size = struct.unpack("<I",f.read(4))[0]
                    s_counter = 0
                    for z in range(0,sec_string_header_size):
                        f.seek(sec_string_header_offset+z)
                        sign = f.read(5)
                        nativecode_index = 0
                        if sign == b".text" or sign == b".TEXT":
                            nativecode_index = z
                            rtn_file_position = f.tell()
                            nativecode_size = 0
                            nativecode_offset = 0
                            for j in range(0,sec_header_count):
                                f.seek(section_header_table_offset + (sec_header_size * j))
                                str_sign = struct.unpack("<I",f.read(4))[0]
                                if nativecode_index == str_sign:
                                    #EXTRACT_DATA
                                    f.read(12)
                                    nativecode_offset = struct.unpack("<I",f.read(4))[0]
                                    nativecode_size = struct.unpack("<I",f.read(4))[0]
                                    if nativecode_offset != 0 and nativecode_size != 0:
                                        wf = open(exportpath + splitter + filename + ".bin","ab")
                                        f.seek(nativecode_offset)
                                        nativecode = f.read(nativecode_size)
                                        wf.write(nativecode)
                                        wf.close()
                                        break
                                    else:
                                        print("[!]ERROR:" + file)
        elif signi == 0x02:
            sectionpos = 0
            size_of_rawdata = 0
            # Section Header Offset
            f.seek(0x28)
            section_header_table_offset = struct.unpack("<Q", f.read(8))[0]
            f.seek(0x3A)
            sec_header_size = struct.unpack("<H", f.read(2))[0]
            f.seek(0x3C)
            sec_header_count = struct.unpack("<H", f.read(2))[0]
            f.seek(0x3E)
            sec_string_header_index = struct.unpack("<H", f.read(2))[0]

            for i in range(0,sec_header_count):
                f.seek(section_header_table_offset + (sec_header_size * i))
                s_name = struct.unpack("<I",f.read(4))[0]
                s_type = struct.unpack("<I",f.read(4))[0]
                if s_name == 1 and s_type == 3:
                    f.read(16)
                    sec_string_header_offset = struct.unpack("<Q",f.read(8))[0]
                    sec_string_header_size = struct.unpack("<Q",f.read(8))[0]
                    s_counter = 0
                    for z in range(0,sec_string_header_size):
                        f.seek(sec_string_header_offset+z)
                        sign = f.read(5)
                        nativecode_index = 0
                        if sign == b".text" or sign == b".TEXT":
                            nativecode_index = z
                            rtn_file_position = f.tell()
                            nativecode_size = 0
                            nativecode_offset = 0
                            for j in range(0,sec_header_count):
                                f.seek(section_header_table_offset + (sec_header_size * j))
                                str_sign = struct.unpack("<I",f.read(4))[0]
                                if nativecode_index == str_sign:
                                    f.read(20)
                                    nativecode_offset = struct.unpack("<Q",f.read(8))[0]
                                    nativecode_size = struct.unpack("<Q",f.read(8))[0]
                                    if nativecode_offset != 0 and nativecode_size != 0:
                                        wf = open(exportpath + splitter + filename + ".bin","ab")
                                        f.seek(nativecode_offset)
                                        nativecode = f.read(nativecode_size)
                                        wf.write(nativecode)
                                        wf.close()
                                        break
                                    else:
                                        print("[!]ERROR:" + file)
            f.close()
print("[*]END------")
