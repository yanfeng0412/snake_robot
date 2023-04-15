from CPG_controllers.neutrons.Sinusoid import sin_oscillator


from CPG_controllers.neutrons.Sinusoid import CPG_SinNeutron
from CPG_controllers.neutrons.Matsuoka import CPG_MatsuokaNeutron
import numpy as np

class CPG_network_Sinusoid(object):
    def __init__(self, CPG_node_num,position_vector, dt):
       # print('CPG_nework type :  Sinusoid / Quadruped ')

        self.num_Cell = CPG_node_num
        ExpectedShape = self.num_Cell * 3 + 1
        if len(position_vector) != ExpectedShape:
            raise Exception("The shape of Position vector ({}) does not match the expected shape ({})".format(
                len(position_vector), ExpectedShape))

        kf = position_vector[0]

        self.CPG_node_num = CPG_node_num  # 不包括placemarker

        if len(position_vector) != self.CPG_node_num * 3 + 1:
            assert "Position vector out of range!"

        GAIN, BIAS, PHASE = [], [], []

        for i in range(self.CPG_node_num):
            GAIN.append(position_vector[i + 1])
            BIAS.append(position_vector[self.CPG_node_num + i + 1])
            PHASE.append(position_vector[2 * self.CPG_node_num + i + 1])

            self.parm_list = {
            0: [0.0, 0.0, 0.0, 1.0, 0.0, 0],
        }

        for i in range(self.CPG_node_num):
            parm = {i + 1: [0.0, 0.0, 0.0, GAIN[i], BIAS[i], PHASE[i]]}
            self.parm_list.update(parm)

        # print(parm_list)

        self.kf = position_vector[0]
        self.num_CPG = len(self.parm_list)
        self.CPG_list = []
        self.w_ms_list = [None, 1, 1, 1, 1, 1, 1, 1, ]
        self.master_list = [None, 0, 1, 1, 2, 2, 3, 3, ]

        for i in range(self.num_CPG):
            if i == 0:
                self.CPG_list.append(CPG_SinNeutron(0, master_nuron=None, param=self.parm_list[0], kf=self.kf, w_ms=0))
            else:
                self.CPG_list.append(CPG_SinNeutron(i, master_nuron=self.CPG_list[self.master_list[i]],
                                                    param=self.parm_list[i], kf=self.kf, w_ms=self.w_ms_list[i]))

    def output(self, state):
        output_list = []
        for cpg_n in self.CPG_list:
            cpg_n.next_output(f1=0, f2=0)
            output_list.append(cpg_n.parm['o'])
        
        return output_list

    def update(self, fi_l):
        max_factor = 2
        self.kesi = 1
        #factor_cpg = fi_l
        if len(fi_l) ==39:
            factor_cpg = np.clip((fi_l-0.5) * max_factor/1.5, -max_factor, max_factor)

            for i in range(self.num_Cell):
                self.CPG_list[i+1].parm['R1'] = self.parm_list[i+1][3] * factor_cpg[i]
                self.CPG_list[i + 1].parm['X1'] = self.parm_list[i + 1][4] * factor_cpg[i]
                self.CPG_list[i + 1].parm['f12'] = self.parm_list[i + 1][5] * factor_cpg[i]




        if len(fi_l) == 2:



            gain_left = 1 - (0.5 - fi_l[0]) * self.kesi
            gain_right = 1 - (0.5 - fi_l[1]) * self.kesi

            self.CPG_list[2].parm['R1'] = self.parm_list[2][3] * gain_left
            self.CPG_list[6].parm['R1'] = self.parm_list[6][3] * gain_left
            self.CPG_list[7].parm['R1'] = self.parm_list[7][3] * gain_left

            self.CPG_list[3].parm['R1'] = self.parm_list[3][3] * gain_left
            self.CPG_list[8].parm['R1'] = self.parm_list[8][3] * gain_left
            self.CPG_list[9].parm['R1'] = self.parm_list[9][3] * gain_left

            self.CPG_list[4].parm['R1'] = self.parm_list[4][3] * gain_right
            self.CPG_list[10].parm['R1'] = self.parm_list[10][3] * gain_right
            self.CPG_list[11].parm['R1'] = self.parm_list[11][3] * gain_right

            self.CPG_list[5].parm['R1'] = self.parm_list[5][3] * gain_right
            self.CPG_list[12].parm['R1'] = self.parm_list[12][3] * gain_right
            self.CPG_list[13].parm['R1'] = self.parm_list[13][3] * gain_right

        elif len(fi_l) == 4:

            f_left = 1 - (0.5 - fi_l[0]) * 2
            f_right = 1 - (0.5 - fi_l[1]) * 2
            gain_left = 1 - (0.5 - fi_l[2]) * self.kesi
            gain_right = 1 - (0.5 - fi_l[3]) * self.kesi

            self.CPG_list[2].parm['f12'] = self.parm_list[2][5] * f_left
            self.CPG_list[6].parm['f12'] = self.parm_list[6][5] * f_left
            self.CPG_list[7].parm['f12'] = self.parm_list[7][5] * f_left

            self.CPG_list[3].parm['f12'] = self.parm_list[3][5] * f_left
            self.CPG_list[8].parm['f12'] = self.parm_list[8][5] * f_left
            self.CPG_list[9].parm['f12'] = self.parm_list[9][5] * f_left

            self.CPG_list[4].parm['f12'] = self.parm_list[4][5] * f_right
            self.CPG_list[10].parm['f12'] = self.parm_list[10][5] * f_right
            self.CPG_list[11].parm['f12'] = self.parm_list[11][5] * f_right

            self.CPG_list[5].parm['f12'] = self.parm_list[5][5] * f_right
            self.CPG_list[12].parm['f12'] = self.parm_list[12][5] * f_right
            self.CPG_list[13].parm['f12'] = self.parm_list[13][5] * f_right
            #           #gain
            self.CPG_list[2].parm['R1'] = self.parm_list[2][3] * gain_left
            self.CPG_list[6].parm['R1'] = self.parm_list[6][3] * gain_left
            self.CPG_list[7].parm['R1'] = self.parm_list[7][3] * gain_left

            self.CPG_list[3].parm['R1'] = self.parm_list[3][3] * gain_left
            self.CPG_list[8].parm['R1'] = self.parm_list[8][3] * gain_left
            self.CPG_list[9].parm['R1'] = self.parm_list[9][3] * gain_left

            self.CPG_list[4].parm['R1'] = self.parm_list[4][3] * gain_right
            self.CPG_list[10].parm['R1'] = self.parm_list[10][3] * gain_right
            self.CPG_list[11].parm['R1'] = self.parm_list[11][3] * gain_right

            self.CPG_list[5].parm['R1'] = self.parm_list[5][3] * gain_right
            self.CPG_list[12].parm['R1'] = self.parm_list[12][3] * gain_right
            self.CPG_list[13].parm['R1'] = self.parm_list[13][3] * gain_right




class CPG_network_Sinusoid_Mix(object):
    def __init__(self, CPG_node_num,position_vector , dt):
        #print('CPG_nework type :  Sinusoid_Mix / Quadruped ')

        self.num_Cell = CPG_node_num

        ExpectedShape = self.num_Cell * 4 + 1
        if len(position_vector) != ExpectedShape:
            raise Exception("The shape of Position vector ({}) does not match the expected shape ({})".format(
                len(position_vector), ExpectedShape) )


        GAIN, BIAS, PHASE, WEIGHT = [], [], [], []

        for i in range(self.num_Cell):
            GAIN.append(position_vector[i + 1])
            BIAS.append(position_vector[self.num_Cell + i + 1])
            PHASE.append(position_vector[2 * self.num_Cell + i + 1])
            WEIGHT.append(position_vector[3 * self.num_Cell + i + 1])

       

        parm_list = {
            0: [0.0, 0.0, 0.0, 1.0, 0.0, 0],
        }
        for i in range(self.num_Cell):
            parm = {i + 1: [0.0, 0.0, 0.0, GAIN[i], BIAS[i], PHASE[i]]}
            parm_list.update(parm)

        self.dt = dt
        self.kf = position_vector[0]
        self.num_CPG = len(parm_list)
        self.CPG_list = []
        self.w_ms_list = [None, WEIGHT[0], WEIGHT[1], WEIGHT[2], WEIGHT[3], WEIGHT[4], WEIGHT[5], WEIGHT[6], WEIGHT[7], WEIGHT[8],WEIGHT[9], WEIGHT[10], WEIGHT[11], WEIGHT[12],]
        self.master_list = [None, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        
        for i in range(self.num_CPG):
            if i == 0:
                self.CPG_list.append(CPG_SinNeutron(0, master_nuron=None, param=parm_list[0], dt = self.dt, kf=self.kf, w_ms=0))
            else:
                self.CPG_list.append(CPG_SinNeutron(i, master_nuron=self.CPG_list[self.master_list[i]],
                                                    param=parm_list[i], dt = self.dt, kf=self.kf, w_ms=self.w_ms_list[i]))
    
    def output(self, state):
        output_list = []
        for cpg_n in self.CPG_list:
            cpg_n.next_output(f1=0, f2=0)
            output_list.append(cpg_n.parm['o'])
        
        return output_list


class CPG_network_Matsuoka(object):
    def __init__(self, CPG_node_num,position_vector, dt):

       # print('CPG_nework type :  Matsuoka / Quadruped ')

        self.num_Cell = CPG_node_num
        ExpectedShape = self.num_Cell * 2 + 1
        if len(position_vector) != ExpectedShape:
            raise Exception("The shape of Position vector ({}) does not match the expected shape ({})".format(
                len(position_vector), ExpectedShape))


        kf = position_vector[0]
        GAIN0 = position_vector[1]
        GAIN1 = position_vector[2]
        GAIN2 = position_vector[3]
        GAIN3 = position_vector[4]
        GAIN4 = position_vector[5]
        GAIN5 = position_vector[6]
        GAIN6 = position_vector[7]
        GAIN7 = position_vector[8]
        GAIN8 = position_vector[9]
        GAIN9 = position_vector[10]
        GAIN10 = position_vector[11]
        GAIN11 = position_vector[12]
        GAIN12 = position_vector[13]

        BIAS0 = position_vector[14]
        BIAS1 = position_vector[15]
        BIAS2 = position_vector[16]
        BIAS3 = position_vector[17]
        BIAS4 = position_vector[18]
        BIAS5 = position_vector[19]
        BIAS6 = position_vector[20]
        BIAS7 = position_vector[21]
        BIAS8 = position_vector[22]
        BIAS9 = position_vector[23]
        BIAS10 = position_vector[24]
        BIAS11 = position_vector[25]
        BIAS12 = position_vector[26]
        self.parm_list = {
            0: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            1: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN0, BIAS0],
            2: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN1, BIAS1],
            3: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN2, BIAS2],
            4: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN3, BIAS3],
            5: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN4, BIAS4],
            6: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN5, BIAS5],
            7: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN6, BIAS6],
            8: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN7, BIAS7],
            9: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN8, BIAS8],
            10: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN9, BIAS9],
            11: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN10, BIAS10],
            12: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN11, BIAS11],
            13: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN12, BIAS12],
        }

        self.dt = dt
        self.kf = position_vector[0]
        self.num_CPG = len(self.parm_list)
        self.CPG_list = []
        # self.w_ms_list = [ None,  1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 ]
        self.w_ms_list = [None, 1, 1, 1, - 1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
        # self.w_ms_list =   [None, 1, -1, -1,  1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.master_list = [None, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

        for i in range(self.num_CPG  ):
            if i == 0:
                self.CPG_list.append(CPG_MatsuokaNeutron(0, master_nuron=None, param=self.parm_list[0], dt = self.dt, kf=self.kf, w_ms=0))
            else:
                self.CPG_list.append(CPG_MatsuokaNeutron(1, master_nuron=self.CPG_list[self.master_list[i]],
                                                 param=self.parm_list[i], dt = self.dt, kf=self.kf, w_ms=self.w_ms_list[i]))

        # CPG0 = CPG_neutron(0, master_nuron = None, param=parm_list[0] ,kf= self.kf, w_ms = None)
        # CPG1 = CPG_neutron(1, master_nuron=CPG0,  param=parm_list[1] ,kf=self.kf, w_ms = 1)
        # CPG2 = CPG_neutron(2, master_nuron=CPG1, param=parm_list[2] , kf=self.kf, w_ms = 1)
        # CPG3 = CPG_neutron(3, master_nuron=CPG1,  param=parm_list[3] ,kf=self.kf, w_ms = 1)
        # CPG4 = CPG_neutron(4, master_nuron=CPG1,  param=parm_list[4] ,kf=self.kf, w_ms = -1)
        # CPG5 = CPG_neutron(5, master_nuron=CPG1, param=parm_list[5] , kf=self.kf, w_ms = -1)
        # CPG6 = CPG_neutron(6, master_nuron=CPG2, param=parm_list[6] , kf=self.kf, w_ms = -1)
        # CPG7 = CPG_neutron(7, master_nuron=CPG2, param=parm_list[7] , kf=self.kf, w_ms = -1)
        # CPG8 = CPG_neutron(8, master_nuron=CPG3,  param=parm_list[8] ,kf=self.kf, w_ms = -1)
        # CPG9 = CPG_neutron(9, master_nuron=CPG3,  param=parm_list[9] ,kf=self.kf, w_ms = -1)
        # CPG10 = CPG_neutron(10, master_nuron=CPG4, param=parm_list[10] , kf=self.kf, w_ms = -1)
        # CPG11 = CPG_neutron(12, master_nuron=CPG4,  param=parm_list[11] ,kf=self.kf, w_ms = -1)
        # CPG12 = CPG_neutron(13, master_nuron=CPG5,  param=parm_list[12] ,kf=self.kf, w_ms = -1)
        # CPG13 = CPG_neutron(14, master_nuron=CPG5,  param=parm_list[13] ,kf=self.kf, w_ms = -1)

        # init

    def output(self, state):
        output_list = []
        for cpg_n in self.CPG_list:
            cpg_n.next_output(f1=0, f2=0)
            output_list.append(cpg_n.parm['o'])

        return output_list


class CPG_network_Matsuoka_Mix(object):
    def __init__(self, CPG_node_num, position_vector, dt):
       # print('CPG_nework type :  Matsuoka_Mix / Quadruped ')

        self.num_Cell = CPG_node_num
        ExpectedShape = self.num_Cell * 3 + 1
        if len(position_vector) != ExpectedShape:
            raise Exception("The shape of Position vector ({}) does not match the expected shape ({})".format(
                len(position_vector), ExpectedShape))



        GAIN, BIAS, PHASE, WEIGHT = [], [], [], []

        for i in range(self.num_Cell):
            GAIN.append(position_vector[i + 1])
            BIAS.append(0)
            PHASE.append(position_vector[self.num_Cell + 1])
            WEIGHT.append(position_vector[self.num_Cell * 2 + i + 1])

        self.parm_list = {
            0: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        }
        for i in range(self.num_Cell):
            parm = {i + 1: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN[i], BIAS[i], PHASE[i]]}
            self.parm_list.update(parm)

        self.dt = dt
        self.kf = position_vector[0]
        self.num_CPG = len(self.parm_list)
        self.CPG_list = []
        self.w_ms_list = [None, WEIGHT[0], WEIGHT[1], WEIGHT[2], WEIGHT[3], WEIGHT[4], WEIGHT[5], WEIGHT[6], WEIGHT[7],
                          WEIGHT[8], WEIGHT[9], WEIGHT[10], WEIGHT[11], WEIGHT[12], ]
        self.master_list = [None, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

        for i in range(self.num_CPG):
            if i == 0:
                self.CPG_list.append(CPG_MatsuokaNeutron(0, master_nuron=None, param=self.parm_list[0], dt = self.dt,kf=self.kf, w_ms=0))
            else:
                self.CPG_list.append(CPG_MatsuokaNeutron(1, master_nuron=self.CPG_list[self.master_list[i]],
                                                 param=self.parm_list[i],dt = self.dt, kf=self.kf, w_ms=self.w_ms_list[i]))

    def output(self, state):
        output_list = []
        for cpg_n in self.CPG_list:
            cpg_n.next_output(f1=0, f2=0)
            output_list.append(cpg_n.parm['o'])

        return output_list
        
       