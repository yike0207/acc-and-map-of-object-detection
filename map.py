import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil
import re
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt




class AP():

    def __init__(self,annofiles,detefile,classname):
        """
        :param annofiles: 存储true标签的文件夹
        :param detefile: 存储detection新的文件
        :param classname: 需要计算TP，FP，FN类别
        """
        self.TP=[]#记录每个bbox的信息
        self.FP=[]#同上，相同位置的self.TP+self.FP=1
        self.imagefile=[]#统计annofile中的图片名
        self.annofiles=annofiles
        self.detefile=detefile
        self.classname=classname
        self.GT_sort=0


    def evaluationTPFP(self,iouthresh=0.3):
        """
        :param conthresh: 置信度的阈值
        """


        # 创建空det数据框，存储数据
        list = ["file","num_det","confidence", "max_IOU", "argmax_index_true"]
        df_det_all = pd.DataFrame(columns=list)


        #遍历detefile文件，找到".jpg"之后，打开对应的annotation文件，进行计算与存储
        with open(self.detefile) as f1:
            line=f1.readline()
            try:#line是有更新的
                while True:
                    if ".jpg" not in line:
                        line=next(f1)
                    else:
                        if not os.path.exists(self.annofiles+line[44:49]+".xml"):
                            # print("%s.jpg文件之前未标注"%line[44:49])
                            line=next(f1)
                        else:
                            self.imagefile.append(line[44:49])
                            # true数据框的存储
                            list1 = [ "xmin", "ymin", "xmax", "ymax"]
                            df_true = pd.DataFrame(columns=list1)
                            true_file = open(self.annofiles + self.imagefile[-1] + '.xml')
                            tree = ET.parse(true_file)
                            root = tree.getroot()
                            for num, obj in enumerate(root.iter('object')):  # 对根节点下面的'object’节点进行遍历
                                cls = obj.find('name').text
                                if cls == self.classname:#需要存储classname这一类别下的
                                    xmlbox = obj.find('bndbox')
                                    df3 = pd.DataFrame(
                                        [[float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                                          float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text)]], columns=list1)
                                    df_true = df_true.append(df3, ignore_index=True)

                            #记录大小，用于修正det数据，有些数据超出边界了
                            obj=root.find('size')
                            x_w=int(obj.find('width').text)
                            y_h=int(obj.find('height').text)

                            #det数据框的存储
                            line=next(f1)#这一步之后进入到标注行，或者enter
                            list2 = [ "confidence", "xmin", "ymin", "xmax", "ymax"]
                            df_det = pd.DataFrame(columns=list2)
                            while not "Enter Image Path" in line:#多找几行，把测试结果下的类别都统计进去
                                temp=re.split(r'[%:,),\s]\s*',line)
                                if temp[0] == self.classname:
                                    xmin=int(temp[3])
                                    ymin=int(temp[5])
                                    w=int(temp[7])
                                    h=int(temp[9])
                                    df2 = pd.DataFrame([[int(temp[1]),max(xmin,0),max(ymin,0),min(xmin+w,x_w),min(ymin+h,y_h)]], columns=list2)
                                    df_det=df_det.append(df2,ignore_index=True)
                                line=next(f1)


                            #已经得到两个数据框了，接下来记录IOU的信息
                            #实际没有，但是检测到了
                            if df_det.shape[0]!=0 and df_true.shape[0]==0:
                                for i in range(df_det.shape[0]):
                                    df=pd.DataFrame([[self.imagefile[-1],i,df_det.loc[i,'confidence'],0,-1]],columns=list)#检测出该类别，但是实际上不是，所以直接iou记为0，用-1做gt的标记
                                    df_det_all=df_det_all.append(df,ignore_index=True)
                            #实际有，也检测到了
                            elif df_det.shape[0]!=0 and df_true.shape[0]!=0:
                                for i in range(df_det.shape[0]):
                                    gts=np.array(df_true)
                                    bbox=np.array(df_det.loc[i,["xmin", "ymin", "xmax", "ymax"]])
                                    [max_IOU,argmax_index]=self.IOU(bbox,gts)
                                    # 会不会有一种情况，置信度很高，但是IOU=0，所以第一个gt被标记了，所以有了这个ifelse
                                    if max_IOU>0:
                                        df=pd.DataFrame([[self.imagefile[-1],i,df_det.loc[i,'confidence'],max_IOU,argmax_index]],columns=list)
                                    else:
                                        df = pd.DataFrame(
                                            [[self.imagefile[-1], i, df_det.loc[i, 'confidence'], max_IOU, -1]],columns=list)  # iou为0，用-1做gt的标记
                                    df_det_all=df_det_all.append(df,ignore_index=True)



            except Exception as e:
                print(str(e))
        f1.close()
        df_det_all = df_det_all.sort_values(by="confidence", ascending=False)#按照置信度降序排序
        df_det_all.reset_index(drop=True, inplace=True)
        for j in range(df_det_all.shape[0]):
            #默认FP
            self.TP.append(0)
            self.FP.append(1)
            if df_det_all.loc[j,"argmax_index_true"]!=-1:#IOU不是0
                if df_det_all.loc[j,"max_IOU"]>iouthresh:#IOU大于阈值
                    #把这个图片对应的行截取出来，截取至这一行，如果他的最后一列的元素只出现了一次，就说明还没配对
                    df=df_det_all[(df_det_all.confidence>=df_det_all.loc[j,"confidence"])&(df_det_all.file==df_det_all.loc[j,"file"])]
                    temp=df["argmax_index_true"].values.tolist()
                    if temp.count(temp[-1])==1:
                        self.TP[-1]=1
                        self.FP[-1]=0


    #统计gt总个数
    def gt_sort(self):
        f=os.listdir(self.annofiles)
        for file in f:
            if file[:-4] in self.imagefile:
                with open(self.annofiles+file) as f1:
                    tree = ET.parse(f1)  # 导入xml数据
                    root = tree.getroot()  # 得到跟节点
                    for obj in root.iter('object'):  # 对根节点下面的'object’节点进行遍历
                        cls = obj.find('name').text
                        if cls == self.classname:
                            self.GT_sort=self.GT_sort+1




    def IOU(self,bbox,gts):
        """
        :param bbox: detection的bbox，只有一个数据 ，输入一维数组
        :param gts: 要比对的gts，可能有很多个，多维数组
        :return: 元组，第一个数字：最大IOU，第二个数字，gts的第几个位置得到了最大IOU
        """


        ixmin = np.maximum(gts[:, 0], bbox[0])
        iymin = np.maximum(gts[:, 1], bbox[1])
        ixmax = np.minimum(gts[:, 2], bbox[2])
        iymax = np.minimum(gts[:, 3], bbox[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bbox[2] - bbox[0] + 1.) * (bbox[3] - bbox[1] + 1.) +
               (gts[:, 2] - gts[:, 0] + 1.) *
               (gts[:, 3] - gts[:, 1] + 1.) - inters)

        overlaps = inters / uni  # 计算交并比
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)  # 找最大的那个
        return (ovmax,jmax)

    def calculate_ap(self):
        tp=np.cumsum(self.TP)#累积TP
        fp=np.cumsum(self.FP)#累积FP
        self.precision=[tp[i]/(tp[i]+fp[i]) for i in range(len(tp))]
        self.recall=[tp[i]/self.GT_sort for i in range(len(tp))]

    def pic(self):
        plt.plot(self.recall,self.precision)
        plt.show()

    def voc_ap(self, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(self.recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(self.precision[self.recall >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], self.recall, [1.]))
            mpre = np.concatenate(([0.], self.precision, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])#找这个区间两个端点中较高的点

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]#判断后一个是不是等于前一个，mrec表示去掉第一个之后的数组，mrec[:-1]表示去掉最后一个之后的数组

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap







if __name__=='__main__':
    wd = getcwd()  # 获取当前文件的路径
    wd = wd.replace('\\', '/')
    CLS=[……，……"]

    for iouthresh in range(30,90,5):
        ap = []
        for c in CLS:
            A=AP(wd+'/test_labels/',wd+'/result.txt',c)
            A.evaluationTPFP(iouthresh/100)
            A.gt_sort()
            A.calculate_ap()
            A.pic()
            ap.append(A.voc_ap())
        print(iouthresh,":",ap)
        print(iouthresh,":",np.mean(ap))










