# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
import heapq

class Dstar:
    '''
    Базовый решатель - алгоритм поиска оптимального пути Dstar
    '''

    def __init__(self, world, goal, start, height, max_all_drop = 0.3):
        '''
        :param world: карта мира
        :param goal: точка цели
        :param start: точка старта
        :param height: карта высот (при планировании по бинарной карте равна карте мира)
        :param max_all_drop: максимальный проходимый объектом перепад высот
        '''
        # инвертируем карту мира с (0,0) в левом нижнем углу в матрицу с (0,0) сверху слева
        world_tr = self.map_invert_f_coord_to_mtrx(map_in_coord = world)
        # =================================================================================
        # инвертируем карту высот с (0,0) в левом нижнем углу в матрицу с (0,0) сверху слева
        height_tr = self.map_invert_f_coord_to_mtrx(map_in_coord = height)
        # =================================================================================

        self.occgridnav = world_tr
        self.shape_world = world.shape
        self.height = height_tr.reshape(-1, order = 'F').tolist()
        self.max_all_drop = max_all_drop
        self.goal = goal
        self.start = start
        self.reset()

        self.NEW = 0
        self.OPEN = 1
        self.CLOSED = 2

    def map_invert_f_coord_to_mtrx(self, map_in_coord):
        '''
        Функция инвертирует карту или слой карты с (0,0) в левом нижнем углу в матрицу с (0,0) сверху слева

        :param map_in_coord: карта или слой карты в координатной сетке, где координата (0,0) слева внизу
        :return: матрица, копия карты или слоя с индексами (0,0) слева вверху
        '''
        map_mtrx = np.zeros(map_in_coord.shape)
        for i1 in range(map_in_coord.shape[0]):
             map_mtrx[i1] = map_in_coord[map_in_coord.shape[0]-1 - i1]
        return map_mtrx

    def reset(self):
        '''
        Сброс служебных параметров

        :return: None
        '''
        self.occgrid_to_costmap(self.occgridnav)
        self.b = np.zeros(self.occgridnav.shape, dtype = np.uint32).reshape(-1).tolist()
        self.t = np.zeros(self.occgridnav.shape).reshape(-1).tolist()
        h = np.ones(self.occgridnav.shape).reshape(-1)
        h.fill(5000)
        self.h = h.tolist()

        self.openlist = []

    def occgrid_to_costmap(self, oc_grid, cost = 1):
        '''
        Служебный метод
        '''
        self.costmap = oc_grid

    def coord_in_indx(self, coord):
        '''
        Перевод координат точки в адрес ячейки в матрице

        :param coord: координаты точки
        :return: координаты ячейки матрицы
        '''
        assert coord[1] < self.shape_world[0] and coord[0] < self.shape_world[1], 'Operation invers coordinates in matrix index: Index out'
        assert coord[0] >= 0 and coord[1] >= 0, 'Operation invers coordinates in matrix index: Index is negative'
        row, column = self.shape_world[0] - coord[1] - 1, coord[0]
        return row, column

    def get_line_indx(self, coord):
        '''
        Получить линейный индекс из подстрочных

        :param coord: подстрочные координаты ячейки
        :return: линейные координаты ячейки матрицы
        '''
        subindx_0, subindx_1 = self.coord_in_indx(coord = coord)
        row_num, column_num = self.shape_world[0], self.shape_world[1]
        indx_r = subindx_1 * row_num
        line_indx = indx_r + subindx_0 # linear indx by starts from scratch
        return line_indx

    def get_line_indx_in_mtrx(self, row_col):
        '''
        Получить линейные координаты из координат матрицы

        :param row_col: координаты ячейки в матрице
        :return: линейный индекс ячейки
        '''
        row_num, column_num = self.shape_world[0], self.shape_world[1]
        indx_r = row_col[1] * row_num
        line_indx = indx_r + row_col[0]
        return line_indx

    def get_subindx_f_lindx(self, lindx):
        '''
        Получить подстрочные индексы из линейных

        :param lindx: линейный индекс ячейки
        :return: подстрочные индексы ячейки в матрице
        '''
        column, row = divmod(lindx, self.shape_world[0])
        return row, column

    def Move_plan(self):
        '''
        Метод итерационного вызова расчета состояний PROCESS_STATE

        :return: None
        '''
        self.G = self.get_line_indx(coord = self.goal)
        self.S = self.get_line_indx(coord = self.start)
        self.INSERT(X = self.G, h_new = 0)
        self.val = 0
        while 1:
            if self.t[self.S] != self.CLOSED and self.val != -1.0:
                self.val = self.PROCESS_STATE()
            else:
                break
        if self.t[self.S] == self.NEW:
            print('Dstar: Move_plan: no path')

    def PROCESS_STATE(self):
        '''
        Метод расчета и расширения соседних состояний

        :return: минимальный ключ состояния
        '''
        X, k_old = self.MIN_STATE_KMIN_DELET()
        if X != X:
            print('Dstar: PROCESS_STATE: X is empty and it is not good')
            r = -1.0
            return r
        X = int(X)
        neighbours = self.neighbours(X)
        for Y in neighbours:
            if (self.t[Y] == self.NEW) or ((self.b[Y] == X) and (self.h[Y] != (self.h[X] + self.c(X = X, Y = Y)))):
                self.b[Y] = X
                self.INSERT(X = Y, h_new = self.h[X] + self.c(X = X, Y = Y))
            else:
                if (self.b[Y] != X) and (self.h[Y] > (self.h[X] + self.c(X = X, Y = Y))):
                    if self.h[X] > k_old:
                        if self.t[X] == self.CLOSED:
                            self.INSERT(X = X, h_new = self.h[X])
                    else:
                        self.b[Y] = X
                        self.INSERT(X = Y, h_new = self.h[X] + self.c(X= X, Y = Y))
                else:
                    if (self.b[Y] != X) and (self.h[X] > (self.h[Y] + self.c(X = X, Y = Y))):
                        if self.h[Y] > k_old:
                            if self.t[Y] == self.CLOSED:
                                self.INSERT(X = Y, h_new = self.h[Y])
                        else:
                            self.b[X] = Y
                            self.INSERT(X = X, h_new = self.h[Y] + self.c(X = Y, Y = X))
        r = k_old
        return r

    def MIN_STATE_KMIN_DELET(self):
        '''
        Служебный метод.

        Удаление минимального состояния

        :return: None or nan
        '''
        if not len(self.openlist):
            return float('nan')

        k_min, X = heapq.heappop(self.openlist)
        self.t[X] = self.CLOSED
        return X, k_min

    def INSERT(self, X, h_new):
        '''
        Вставка в OpenList состояния X

        :param X: состояние
        :param h_new: стоимость прохода до состояния
        :return: None
        '''
        if self.t[X] == self.NEW:
            k_new = h_new
            heapq.heappush(self.openlist, [k_new, X])
        if self.t[X] == self.CLOSED:
            k_new = min(self.h[X], h_new)
            heapq.heappush(self.openlist, [k_new, X])
        self.h[X] = h_new
        self.t[X] = self.OPEN

    def neighbours(self, X):
        '''
        Определение соседних состояний относительно X

        :param X: текущее состояние
        :return: список соседних состояний
        '''
        X_sub_indx_0, X_sub_indx_1 = self.get_subindx_f_lindx(lindx = X)
        r, c = X_sub_indx_0, X_sub_indx_1
        y = [[r-1, c-1], [r-1, c], [r-1, c+1], [r, c-1], [r, c+1], [r+1, c-1], [r+1, c], [r+1, c+1]]
        y_1 = []
        for i in range(len(y)):
            if not (y[i][1] < 0 or y[i][0] < 0 or y[i][0] >= self.occgridnav.shape[0] or y[i][1] >= self.occgridnav.shape[1]):
                y_1.append(y[i])
        Y = []
        for count in range(len(y_1)):
            y_to_list = self.get_line_indx_in_mtrx(y_1[count])
            Y.append(y_to_list)
        return Y

    def c(self, X, Y):
        '''
        Расчет стоимости перемещения в состояние Y из состояния X

        :param X: текущее сстояние
        :param Y: следующее состояние
        :return: стоимость перемещения из текущего состояния в соседнее
        '''
        Y_0, Y_1 = self.get_subindx_f_lindx(lindx = Y)
        X_0, X_1 = self.get_subindx_f_lindx(lindx = X)

        dist = sqrt((X_0-Y_0)**2 +(X_1-Y_1)**2)
        dcost = 1
        drop = abs(self.height[Y] - self.height[X])
        if drop > self.max_all_drop:
            dcost = 10000
        else:
            dist = sqrt(dist*dist + drop*drop)
        return dist * dcost

    def query(self, start):
        '''
        Расчет пути от точки цели до точки старта (расчет в обратном порядке)

        :param start: точка старта
        :return: последовательный список состояний в рассчитанной траектории
        '''
        self.robot = start
        path = [start]

        while True:
            self.robot = self.next(current_pos = self.robot)
            if self.robot == self.goal:
                path.append(self.goal)
                break
            else:
                path.append(self.robot)
        pp = np.array(path)
        return pp

    def next(self, current_pos):
        '''
        Служебный метод

        Расчет следующего состояния

        :param current_pos: текущее состояние
        :return: следующее состояние
        '''
        X_line_indx = self.get_line_indx(coord = current_pos)
        X_next_line = self.b[X_line_indx]
        X_next_subindx_0, X_next_subindx_1 = self.get_subindx_f_lindx(lindx = X_next_line)
        X_coord = [X_next_subindx_1, self.occgridnav.shape[0] - X_next_subindx_0 - 1]
        return X_coord

class Dstar_inflation(Dstar):
    '''
    Расширение алгоритма поиска пути для расчета пути с зонами безопасности
    '''
    def __init__(self, world, goal, start, height, OBS_layer, max_all_drop = 0.3):
        '''
        :param world: карта мира
        :param goal: точка цели
        :param start: точка старта
        :param height: карта высот
        :param OBS_layer: слой карты с зонами безопасности
        :param max_all_drop: масимально допустимый перепад высот
        '''
        super().__init__(world, goal, start, height, max_all_drop)
        # инвертируем слой с препятствиями с (0,0) в левом нижнем углу в матрицу с (0,0) сверху слева
        OBS_layer_tr = self.map_invert_f_coord_to_mtrx(map_in_coord = OBS_layer)
        #==================================
        self.OBS_layer = OBS_layer_tr.reshape(-1, order = 'F').tolist()

    def c(self, X, Y):
        '''
         Расчет стоимости перемещения в состояние Y из состояния X
         
         :param X: текущее сстояние
         :param Y: следующее состояние
         :return: стоимость перемещения из текущего состояния в соседнее
         '''
        Y_0, Y_1 = self.get_subindx_f_lindx(lindx=Y)
        X_0, X_1 = self.get_subindx_f_lindx(lindx=X)
        # расчет стоимости перемещения из X в Y по карте высот
        dist = sqrt((X_0 - Y_0) ** 2 + (X_1 - Y_1) ** 2)
        dcost = 1
        drop = abs(self.height[Y] - self.height[X])
        if drop > self.max_all_drop:
            dcost = 10000
        else:
            dist = sqrt(dist * dist + drop * drop)
        cost_f_height = dist * dcost
        # расчет стоимости перемещения из X в Y по слою с препятствиями
        # если следующее состояние (Y) приходится на препятствие, то стоимость перемещения в это состояние 10000
        if self.OBS_layer[Y] >= 1:
            cost_f_obs_layer = 10000
        else:
            cost_f_obs_layer = abs(self.OBS_layer[X] + self.OBS_layer[Y]) / 2
        # расчет суммарной стоимости перемещения
        sum_cost = cost_f_height + cost_f_obs_layer*10
        return sum_cost