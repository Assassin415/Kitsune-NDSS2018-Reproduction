import math
import numpy as np


class incStat:
    def __init__(self, Lambda, ID, init_time=0, isTypeDiff=False):  # init_time 是这个统计对象的创建时间
        self.ID = ID
        self.CF1 = 0  # 一次方累加和（论文里的 LS）
        self.CF2 = 0  # 平方累加和（论文里的 SS）
        self.w = 1e-20  # 权重（论文里的 w），从极小值起步是为了避免除以 0
        self.isTypeDiff = isTypeDiff
        self.Lambda = Lambda  # 衰减因子 λ
        self.lastTimestamp = init_time
        self.cur_mean = np.nan
        self.cur_var = np.nan
        self.cur_std = np.nan
        self.covs = [] # 与本统计对象相关的协方差对象列表（存的是引用）

    def insert(self, v, t=0):  # v 是一个数值（例如包长），t 是这个值到达的时间戳
        if self.isTypeDiff:
            dif = t - self.lastTimestamp
            if dif > 0:
                v = dif
            else:
                v = 0
        self.processDecay(t)

        # 用新值 v 更新三个累加量
        self.CF1 += v
        self.CF2 += math.pow(v, 2)
        self.w += 1
        self.cur_mean = np.nan  # 标记缓存失效：下次取用时重新计算
        self.cur_var = np.nan
        self.cur_std = np.nan

        # 顺便更新所有与本流相关的协方差对象
        for cov in self.covs:
            cov.update_cov(self.ID, v, t)

    def processDecay(self, timestamp):
        factor=1
        # 计算从上次更新到现在，应该衰减多少
        timeDiff = timestamp - self.lastTimestamp
        if timeDiff > 0:
            factor = math.pow(2, (-self.Lambda * timeDiff))
            self.CF1 = self.CF1 * factor
            self.CF2 = self.CF2 * factor
            self.w = self.w * factor
            self.lastTimestamp = timestamp
        return factor

    def weight(self):
        return self.w

    def mean(self):
        if math.isnan(self.cur_mean):  # 只在需要时算一次，算完缓存起来
            self.cur_mean = self.CF1 / self.w
        return self.cur_mean

    def var(self):
        if math.isnan(self.cur_var):  # 只在需要时算一次，算完缓存起来
            self.cur_var = abs(self.CF2 / self.w - math.pow(self.mean(), 2))
        return self.cur_var

    def std(self):
        if math.isnan(self.cur_std):  # 只在需要时算一次，算完缓存起来
            self.cur_std = math.sqrt(self.var())
        return self.cur_std

    def cov(self,ID2):
        for cov in self.covs:
            if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
                return cov.cov()
        return [np.nan]

    def pcc(self,ID2):
        for cov in self.covs:
            if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
                return cov.pcc()
        return [np.nan]

    def cov_pcc(self,ID2):
        for cov in self.covs:
            if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
                return cov.get_stats1()
        return [np.nan]*2

    def radius(self, other_incStats):  # 一组统计对象的"半径"（论文里的 R）
        A = self.var()**2
        for incS in other_incStats:
            A += incS.var()**2
        return math.sqrt(A)

    def magnitude(self, other_incStats):  # 一组统计对象的"幅度"（论文里的 ||S||）
        A = math.pow(self.mean(), 2)
        for incS in other_incStats:
            A += math.pow(incS.mean(), 2)
        return math.sqrt(A)

    # 计算并返回这条流的全部统计量：[权重, 均值, 方差]
    def allstats_1D(self):
        self.cur_mean = self.CF1 / self.w
        self.cur_var = abs(self.CF2 / self.w - math.pow(self.cur_mean, 2))
        return [self.w, self.cur_mean, self.cur_var]

    # 计算并返回这条流的统计量，以及它与指定流之间的相关统计量
    def allstats_2D(self, ID2):
        stats1D = self.allstats_1D()
        # 找出与指定流之间的协方差对象
        stats2D = [np.nan] * 4
        for cov in self.covs:
            if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
                stats2D = cov.get_stats2()
                break
        return stats1D + stats2D

    def getHeaders_1D(self, suffix=True):
        if self.ID is None:
            s0=""
        else:
            s0 = "_0"
        if suffix:
            s0 = "_"+self.ID
        headers = ["weight"+s0, "mean"+s0, "std"+s0]
        return headers

    def getHeaders_2D(self, ID2, suffix=True):
        hdrs1D = self.getHeaders_1D(suffix)
        if self.ID is None:
            s0=""
            s1=""
        else:
            s0 = "_0"
            s1 = "_1"
        if suffix:
            s0 = "_"+self.ID
            s1 = "_" + ID2
        hdrs2D = ["radius_" + s0 + "_" + s1, "magnitude_" + s0 + "_" + s1, "covariance_" + s0 + "_" + s1,
                   "pcc_" + s0 + "_" + s1]
        return hdrs1D+hdrs2D


# 和 incStat 类似，但它记录的是"两条流之间"的统计量
class incStat_cov:
    def __init__(self, incS1, incS2, init_time = 0):
        # 保存两条流各自的 incStat 引用
        self.incStats = [incS1,incS2]
        self.lastRes = [0,0]
        # 初始化外推器（本版本未启用，作者曾用它做过实验性的相关性外推）
        #self.EXs = [extrapolator(),extrapolator()]

        # 初始化"残差乘积和"
        self.CF3 = 0 # 残差乘积的累加和 Σ(A-均值A)(B-均值B)，论文里叫 SR
        self.w3 = 1e-20
        self.lastTimestamp_cf3 = init_time

    # ID 是产生 (v,t) 的那条流的标识
    def update_cov(self, ID, v, t):  # 前提：ID 对应的 incStat 已经用 (t,v) 更新过了（由 incStat.insert() 自动完成）
        # 判断这个值来自哪条流
        if ID == self.incStats[0].ID:
            inc = 0
        elif ID == self.incStats[1].ID:
            inc = 1
        else:
            print("update_cov ID error")
            return ## 出错

        # 先把另一条流按时间衰减一下
        self.incStats[not(inc)].processDecay(t)

        # 再衰减自己的残差累计
        self.processDecay(t,inc)

        # 更新当前流的外推器（本版本未启用）
        #self.EXs[inc].insert(t,v)

        # 外推另一条流的值（本版本未启用）
        #v_other = self.EXs[not(inc)].predict(t)

        # 计算并更新残差
        res = (v - self.incStats[inc].mean())
        resid = (v - self.incStats[inc].mean()) * self.lastRes[not(inc)]
        self.CF3 += resid
        self.w3 += 1
        self.lastRes[inc] = res

    def processDecay(self,t,micro_inc_indx):
        factor = 1
        # 计算 CF3（残差乘积和）在这段时间里应该衰减多少
        timeDiffs_cf3 = t - self.lastTimestamp_cf3
        if timeDiffs_cf3 > 0:
            factor = math.pow(2, (-(self.incStats[micro_inc_indx].Lambda) * timeDiffs_cf3))
            self.CF3 *= factor
            self.w3 *= factor
            self.lastTimestamp_cf3 = t
            self.lastRes[micro_inc_indx] *= factor
        return factor

    # 待办：为 CF3 补上独立的权重 w3

    # 协方差的近似值
    def cov(self):
        return self.CF3 / self.w3

    # 皮尔逊相关系数
    def pcc(self):
        ss = self.incStats[0].std() * self.incStats[1].std()
        if ss != 0:
            return self.cov() / ss
        else:
            return 0

    # 计算并返回相关统计量：[协方差, 相关系数]
    def get_stats1(self):
        return [self.cov(), self.pcc()]

    # 返回相关统计量，以及两条流的 2D 统计量（半径、幅度）
    def get_stats2(self):
        return [self.incStats[0].radius([self.incStats[1]]),self.incStats[0].magnitude([self.incStats[1]]),self.cov(), self.pcc()]

    # 返回相关统计量、2D 统计量，以及两条流各自的常规统计量
    def get_stats3(self):
        return [self.incStats[0].w,self.incStats[0].mean(),self.incStats[0].std(),self.incStats[1].w,self.incStats[1].mean(),self.incStats[1].std(),self.cov(), self.pcc()]

    # 返回两条流的常规统计量 + 2D 统计量 + 相关统计量
    def get_stats4(self):
        return [self.incStats[0].w,self.incStats[0].mean(),self.incStats[0].std(),self.incStats[1].w,self.incStats[1].mean(),self.incStats[1].std(), self.incStats[0].radius([self.incStats[1]]),self.incStats[0].magnitude([self.incStats[1]]),self.cov(), self.pcc()]

    def getHeaders(self,ver,suffix=True): # ver 取 1~5，决定返回哪一组表头
        headers = []
        s0 = "0"
        s1 = "1"
        if suffix:
            s0 = self.incStats[0].ID
            s1 = self.incStats[1].ID

        if ver == 1:
            headers = ["covariance_"+s0+"_"+s1, "pcc_"+s0+"_"+s1]
        if ver == 2:
            headers = ["radius_"+s0+"_"+s1, "magnitude_"+s0+"_"+s1, "covariance_"+s0+"_"+s1, "pcc_"+s0+"_"+s1]
        if ver == 3:
            headers = ["weight_"+s0, "mean_"+s0, "std_"+s0,"weight_"+s1, "mean_"+s1, "std_"+s1, "covariance_"+s0+"_"+s1, "pcc_"+s0+"_"+s1]
        if ver == 4:
            headers = ["weight_" + s0, "mean_" + s0, "std_" + s0, "covariance_" + s0 + "_" + s1, "pcc_" + s0 + "_" + s1]
        if ver == 5:
            headers = ["weight_"+s0, "mean_"+s0, "std_"+s0,"weight_"+s1, "mean_"+s1, "std_"+s1, "radius_"+s0+"_"+s1, "magnitude_"+s0+"_"+s1, "covariance_"+s0+"_"+s1, "pcc_"+s0+"_"+s1]
        return headers


class incStatDB:
    # default_lambda: 所有流共用这个 λ。如果不指定，就必须在每次查询时自己传入 λ
    def __init__(self,limit=np.inf,default_lambda=np.nan):
        self.HT = dict()
        self.limit = limit
        self.df_lambda = default_lambda

    def get_lambda(self,Lambda):
        if not np.isnan(self.df_lambda):
            Lambda = self.df_lambda
        return Lambda

    # 注册一条新流。init_time 用来初始化该 incStat 的 lastTimestamp
    def register(self,ID,Lambda=1,init_time=0,isTypeDiff=False):
        # 是否使用默认 λ？
        Lambda = self.get_lambda(Lambda)

        # 取出对应的 incStat（不存在就新建）
        key = ID+"_"+str(Lambda)
        incS = self.HT.get(key)
        if incS is None: # 这条流还没有登记过
            if len(self.HT) + 1 > self.limit:
                raise LookupError(
                    'Adding Entry:\n' + key + '\nwould exceed incStatHT 1D limit of ' + str(
                        self.limit) + '.\nObservation Rejected.')
            incS = incStat(Lambda, ID, init_time, isTypeDiff)
            self.HT[key] = incS # 新增一条记录
        return incS

    # 为两条流登记协方差跟踪（缺失的流会自动补登记）
    def register_cov(self,ID1,ID2,Lambda=1,init_time=0,isTypeDiff=False):
        # 是否使用默认 λ？
        Lambda = self.get_lambda(Lambda)

        # 取出两条流的 incStat
        incS1 = self.register(ID1,Lambda,init_time,isTypeDiff)
        incS2 = self.register(ID2,Lambda,init_time,isTypeDiff)

        # 检查两者之间是否已经建立过关联
        for cov in incS1.covs:
            if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
                return cov # 已经有关联了，直接返回

        # 建立关联
        inc_cov = incStat_cov(incS1,incS2,init_time)
        incS1.covs.append(inc_cov)
        incS2.covs.append(inc_cov)
        return inc_cov

    # 更新一条流（必要时先注册）
    def update(self,ID,t,v,Lambda=1,isTypeDiff=False):
        incS = self.register(ID,Lambda,t,isTypeDiff)
        incS.insert(v,t)
        return incS

    # 取出指定流的当前统计量
    def get_1D_Stats(self,ID,Lambda=1): # 权重, 均值, 方差
        # 是否使用默认 λ？
        Lambda = self.get_lambda(Lambda)

        # 取出 incStat
        incS = self.HT.get(ID+"_"+str(Lambda))
        if incS is None:  # 这条流还不存在
            return [np.na]*3
        else:
            return incS.allstats_1D()

    # 取出两条流之间的相关统计量
    def get_2D_Stats(self, ID1, ID2, Lambda=1): # 协方差, 相关系数
        # 是否使用默认 λ？
        Lambda = self.get_lambda(Lambda)

        # 取出 incStat
        incS1 = self.HT.get(ID1 + "_" + str(Lambda))
        if incS1 is None:  # 不存在
            return [np.na]*2

        # 找到对应的协方差对象
        return incS1.cov_pcc(ID2)

    # 取出与该流登记过的全部相关统计量
    # 返回 (统计量列表, 对应的流 ID 列表)
    def get_all_2D_Stats(self, ID, Lambda=1):  # 协方差, 相关系数
        # 是否使用默认 λ？
        Lambda = self.get_lambda(Lambda)

        # 取出 incStat
        incS1 = self.HT.get(ID + "_" + str(Lambda))
        if incS1 is None:  # 不存在
            return ([],[])

        # 遍历该流的所有关联，收集统计量和对应的流 ID
        stats = []
        IDs = []
        for cov in incS1.covs:
            stats.append(cov.get_stats1())
            IDs.append([cov.incStats[0].ID,cov.incStats[1].ID])
        return stats,IDs

    # 取出多个流组合起来的多维统计量
    def get_nD_Stats(self,IDs,Lambda=1): # 半径, 幅度（IDs 是一个列表）
        # 是否使用默认 λ？
        Lambda = self.get_lambda(Lambda)

        # 取出各条流的 incStat
        incStats = []
        for ID in IDs:
            incS = self.HT.get(ID + "_" + str(Lambda))
            if incS is not None:  # 存在
                incStats.append(incS)

        # 计算统计量
        rad = 0 # 半径
        mag = 0 # 幅度
        for incS in incStats:
            rad += incS.var()
            mag += incS.mean()**2

        return [np.sqrt(rad),np.sqrt(mag)]

    # 先更新、再返回这条流的 1D 统计量。遇到没见过的流会自动登记
    def update_get_1D_Stats(self, ID,t,v,Lambda=1,isTypeDiff=False):  # 权重, 均值, 方差
        incS = self.update(ID,t,v,Lambda,isTypeDiff)
        return incS.allstats_1D()


    # 先更新、再返回两条流之间的相关统计量。会自动补登记缺失的流和协方差跟踪
    # 注意：AfterImage 目前不支持对"差值型"流做相关性统计
    def update_get_2D_Stats(self, ID1,ID2,t1,v1,Lambda=1,level=1):  # level=1 返回协方差和相关系数；level=2 再加半径和幅度
        # 取出（或新建）协方差跟踪对象
        inc_cov = self.register_cov(ID1, ID2, Lambda,  t1)
        # 更新协方差跟踪对象
        inc_cov.update_cov(ID1,v1,t1)
        if level == 1:
            return inc_cov.get_stats1()
        else:
            return inc_cov.get_stats2()

    # 先更新、再返回 1D + 2D 统计量（这是 Kitsune 实际使用的接口）
    def update_get_1D2D_Stats(self, ID1,ID2,t1,v1,Lambda=1):  # 权重, 均值, 方差 + 半径, 幅度, 协方差, 相关系数
        return self.update_get_1D_Stats(ID1,t1,v1,Lambda) + self.update_get_2D_Stats(ID1,ID2,t1,v1,Lambda,level=2)

    def getHeaders_1D(self,Lambda=1,ID=None):
        # 是否使用默认 λ？
        Lambda = self.get_lambda(Lambda)
        hdrs = incStat(Lambda,ID).getHeaders_1D(suffix=False)
        return [str(Lambda)+"_"+s for s in hdrs]

    def getHeaders_2D(self,Lambda=1,IDs=None, ver=1): # IDs 是两个元素的列表或元组
        # 是否使用默认 λ？
        Lambda = self.get_lambda(Lambda)
        if IDs is None:
            IDs = [0,1]
        hdrs = incStat_cov(incStat(Lambda,IDs[0]),incStat(Lambda,IDs[0]),Lambda).getHeaders(ver,suffix=False)
        return [str(Lambda)+"_"+s for s in hdrs]

    def getHeaders_1D2D(self,Lambda=1,IDs=None, ver=1):
        # 是否使用默认 λ？
        Lambda = self.get_lambda(Lambda)
        if IDs is None:
            IDs = [0,1]
        hdrs1D = self.getHeaders_1D(Lambda,IDs[0])
        hdrs2D = self.getHeaders_2D(Lambda,IDs, ver)
        return hdrs1D + hdrs2D

    def getHeaders_nD(self,Lambda=1,IDs=[]): # IDs 是 n 个元素的列表或元组
        # 是否使用默认 λ？
        ID = ":"
        for s in IDs:
            ID += "_"+s
        Lambda = self.get_lambda(Lambda)
        hdrs = ["radius"+ID, "magnitude"+ID]
        return [str(Lambda)+"_"+s for s in hdrs]


    # 清理掉权重低于阈值的记录（也就是"已经凉了"的流）
    # 返回被清理掉的记录条数
    def cleanOutOldRecords(self,cutoffWeight,curTime):
        n = 0
        dump = sorted(self.HT.items(), key=lambda tup: tup[1][0].getMaxW(curTime))
        for entry in dump:
            entry[1][0].processDecay(curTime)
            W = entry[1][0].w
            if W <= cutoffWeight:
                key = entry[0]
                del entry[1][0]
                del self.HT[key]
                n=n+1
            elif W > cutoffWeight:
                break
        return n

