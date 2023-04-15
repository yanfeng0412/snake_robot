# @Time : 2021/10/19 11:33 上午
# @Author : AaronWang
# @FileName : matsuoka.py
# @Introduction :

"""
设计松冈模型
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time
import csv


class CPG(object):
    """
    参数说明：
    T_r:上升时间常数
    T_a:适应时间常数
    Ue:伸肌内部状态变量Ue
    Ve:伸肌内部状态变量Ve
    Uf:屈肌内部状态变量Uf
    Vf:屈肌内部状态变量Vf
    omega_fe:相互抑制系数
    beta:神经元的自抑制系数
    s0:正常数，表示上层激励信号
    Feed_e:伸肌反馈输入
    Feed_f:屈肌反馈输入
    y_e:伸肌输出
    y_f:屈肌输出
    y:Matsuoka模型的输出
    qs:[u0_e,v0_e,u0_f,v0_f]伸肌神经元和屈肌神经元的初始状态
    omega:神经元之间的连接权重
    """

    def __init__(self,
                 T_r=0.04,
                 T_a=0.4,
                 omega_fe=2.0,
                 beta=2.5,
                 s0=1.0,
                 Feed_e=0,
                 Feed_f=0,
                 qs=[0.011, 0.0081, 0.0022, 0.0057],
                 timesteps=0.01,
                 ):
        self._T_r = T_r
        self._T_a = T_a
        self._omega_fe = omega_fe
        self._beta = beta
        self._s0 = s0
        self._Feed1 = Feed_e
        self._Feed2 = Feed_f
        self._qs = qs
        self._timesteps = timesteps

    def get_parms(self):
        return self._T_r, self._T_a, self._omega_fe, self._beta, self._s0, self._Feed1, self._Feed2

    def Extensor(self, qs, timesteps):
        Ue, Ve, Uf, Vf = qs

        y_e = max(Ue, 0)
        y_f = max(Uf, 0)

        Tr, Ta, omega_fe, beta, s0, Feed_e, Feed_f = self.get_parms()

        dUe = (1 / Tr) * (-Ue - (omega_fe * y_f) - (beta * Ve) + s0 + Feed_e)
        dVe = (1 / Ta) * (-Ve + y_e)
        dUf = (1 / Tr) * (-Uf - (omega_fe * y_e) - (beta * Vf) + s0 + Feed_f)
        dVf = (1 / Ta) * (-Vf + y_f)

        return dUe, dVe, dUf, dVf

    def calculate(self, t):
        data = integrate.odeint(self.Extensor, self._qs, t)
        self.SaveToCSV(data, 500)
        return data

    def show(self):
        t = np.arange(0, 5, self._timesteps)
        data = self.calculate(t)
        fig1 = plt.figure()
        plt.plot(t, data[:, 0], label='Ue')
        plt.plot(t, data[:, 1], label='Uf')

        # fig2 = plt.figure()
        plt.plot(t, data[:, 2], label='Ve')
        plt.plot(t, data[:, 3], label='Vf')
        plt.legend()
        plt.show()

    def SaveToCSV(self, data, columns):
        with open('matsuoka.csv', 'w', encoding='utf-8', newline='' "") as f:
            headers = ['Ue', 'Ve', 'Uf', 'Vf']
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            rows = []
            for i in range(columns):
                for j in range(4):
                    rows.append(data[:, j][i])
                f_csv.writerow(rows)
                rows = []
        f.close()


if __name__ == "__main__":
    h = CPG()
    h.show()
    # t = np.arange(0, 5, 0.01)
    # print(len(CPG().calculate(t)))