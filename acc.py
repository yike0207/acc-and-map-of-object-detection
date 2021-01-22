import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil

wd = getcwd()  # 获取当前文件的路径
wd = wd.replace('\\', '/')

gt_num=0
pred_num=0

CLS=["……,……"]

for cls in CLS:
    locals()[str(cls)] =0
    locals()['right_'+str(cls)] = 0
    locals()['wrong_' + str(cls)] = 0#判断为knief or pollute，实际上是fire


wrong=0
wrong_cls=0
wrong_cls_temp=0

no_cls=0
no_test_labels=0
acc=[]#准确：1，错误：0，test_labels里没有：2
f = os.listdir(wd+'/test_labels/')



with open(wd+'/result.txt', 'r',encoding='utf8') as f1:
    line = f1.readline()  # 整行读取数
    i=1
    while i==1:
        temp=[]#每个图片测试结果的类别情况
        temp2=[]#每个图片原标注的类别情况
        wrong_cls_temp=0
        if ".jpg" in line:
            try:
                in_file = open(wd+'/test_labels/%s.xml'%line[44:49])
                tree = ET.parse(in_file)  # 导入xml数据
                root = tree.getroot()  # 得到跟节点
                while not "Enter Image Path" in line:#多找几行，把测试结果下的类别都统计进去
                    line=next(f1)
                    for cls in CLS:
                        if cls in line:
                            temp.append(cls)
                for num, obj in enumerate(root.iter('object')):  # 对根节点下面的'object’节点进行遍历
                    cls = obj.find('name').text
                    temp2.append(cls)
                gt_num=gt_num+num+1
                exec(str(temp2[0])+'='+str(temp2[0])+'+num+1')#统计各类数目
                pred_num=pred_num+len(temp)
                temp1=temp.copy()
                if temp:#如果测试结果有类别结果的话
                    for cls in temp2:
                        if cls in temp:
                            acc.append(1)#每个xml下有多个标签的话，每个标签算作一次，去匹配
                            temp.remove(cls)#匹配到了就删掉测试结果对应中的的类别
                            exec('right_'+str(cls)+'=right_'+str(cls)+'+1')
                        else:
                            acc.append(0)
                            # exec('wrong=wrong+1')
                            if temp:
                                exec('wrong=wrong+1')
                                exec('wrong_'+str(cls)+'=wrong_'+str(cls)+'+1')
                                temp.pop()
                            else:
                                no_cls=no_cls+1
                else:
                    line=next(f1)
                    no_cls=no_cls+num+1#测试集中没有标注的计入no_cls
                    acc.extend([0]*(num+1))#没有标记出来的也计入acc
            except:
                line=next(f1)
                acc.append(2)#测试结果中的文件在原标注汇总找不到
                no_test_labels=no_test_labels+1
        else:
            try:
                line = next(f1)
            except:
                i=0

print("gt_num:",gt_num)
print("pred_num:",pred_num)
print("no_test_labels:",no_test_labels)
print("no_cls:",no_cls)
print('\n')

for cls in CLS:
    print(cls,":",eval(cls))
    print("right_"+cls+"_rate", eval("right_"+cls)/eval(cls))
    print("wrong_" + cls + "_rate", eval("wrong_" + cls) / eval(cls))
    print('\n')


print("wrong:",wrong/gt_num)
true=acc.count(1)
flase=acc.count(0)
print("acc:",true/(true+flase))
print('分母：gt个数')


