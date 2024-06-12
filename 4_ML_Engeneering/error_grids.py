import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Функции для определения зон ошибок Кларка
def clarke_error_zone_detailed(act, pred):
    if (act < 70 and pred < 70) or abs(act - pred) < 0.2 * act:
        return 0
    if act <= 70 and pred >= 180:
        return 8
    if act >= 180 and pred <= 70:
        return 7
    if act >= 240 and 70 <= pred <= 180:
        return 6
    if act <= 70 <= pred <= 180:
        return 5
    if 70 <= act <= 290 and pred >= act + 110:
        return 4
    if 130 <= act <= 180 and pred <= (7 / 5) * act - 182:
        return 3
    if act < pred:
        return 2
    return 1

clarke_error_zone_detailed = np.vectorize(clarke_error_zone_detailed)

# Функции для определения зон ошибок Паркса
def parkes_error_zone_detailed(act, pred, diabetes_type):
    def above_line(x1, y1, x2, y2, strict=False):
        if x1 == x2:
            return False
        y_line = ((y1 - y2) * act + y2 * x1 - y1 * x2) / (x1 - x2)
        return pred > y_line if strict else pred >= y_line

    def below_line(x1, y1, x2, y2, strict=False):
        return not above_line(x1, y1, x2, y2, not strict)

    def parkes_type_1(act, pred):
        if above_line(0, 150, 35, 155) and above_line(35, 155, 50, 550):
            return 7
        if (pred > 100 and above_line(25, 100, 50, 125) and
                above_line(50, 125, 80, 215) and above_line(80, 215, 125, 550)):
            return 6
        if (act > 250 and below_line(250, 40, 550, 150)):
            return 5
        if (pred > 60 and above_line(30, 60, 50, 80) and
                above_line(50, 80, 70, 110) and above_line(70, 110, 260, 550)):
            return 4
        if (act > 120 and below_line(120, 30, 260, 130) and below_line(260, 130, 550, 250)):
            return 3
        if (pred > 50 and above_line(30, 50, 140, 170) and
                above_line(140, 170, 280, 380) and (act < 280 or above_line(280, 380, 430, 550))):
            return 2
        if (act > 50 and below_line(50, 30, 170, 145) and
                below_line(170, 145, 385, 300) and (act < 385 or below_line(385, 300, 550, 450))):
            return 1
        return 0

    def parkes_type_2(act, pred):
        if (pred > 200 and above_line(35, 200, 50, 550)):
            return 7
        if (pred > 80 and above_line(25, 80, 35, 90) and above_line(35, 90, 125, 550)):
            return 6
        if (act > 250 and below_line(250, 40, 410, 110) and below_line(410, 110, 550, 160)):
            return 5
        if (pred > 60 and above_line(30, 60, 280, 550)):
            return 4
        if (below_line(90, 0, 260, 130) and below_line(260, 130, 550, 250)):
            return 3
        if (pred > 50 and above_line(30, 50, 230, 330) and
                (act < 230 or above_line(230, 330, 440, 550))):
            return 2
        if (act > 50 and below_line(50, 30, 90, 80) and below_line(90, 80, 330, 230) and
                (act < 330 or below_line(330, 230, 550, 450))):
            return 1
        return 0

    if diabetes_type == 1:
        return parkes_type_1(act, pred)

    if diabetes_type == 2:
        return parkes_type_2(act, pred)

    raise Exception('Unsupported diabetes type')

parkes_error_zone_detailed = np.vectorize(parkes_error_zone_detailed)

# Функция для расчета точности зон
def zone_accuracy(act_arr, pred_arr, mode='clarke', detailed=False, diabetes_type=1):
    acc = np.zeros(9)
    if mode == 'clarke':
        res = clarke_error_zone_detailed(act_arr, pred_arr)
    elif mode == 'parkes':
        res = parkes_error_zone_detailed(act_arr, pred_arr, diabetes_type)
    else:
        raise Exception('Unsupported error grid mode')

    acc_bin = np.bincount(res)
    acc[:len(acc_bin)] = acc_bin

    if not detailed:
        acc[1] = acc[1] + acc[2]
        acc[2] = acc[3] + acc[4]
        acc[3] = acc[5] + acc[6]
        acc[4] = acc[7] + acc[8]
        acc = acc[:5]

    return acc / sum(acc)

# Функция для построения графика точности зон
def plot_zone_accuracy(act_arr, pred_arr, mode='clarke', detailed=False, diabetes_type=1):
    accuracy = zone_accuracy(act_arr, pred_arr, mode, detailed, diabetes_type)

    if detailed:
        labels = ['A', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2', 'E1', 'E2']
    else:
        labels = ['A', 'B', 'C', 'D', 'E']

    plt.figure(figsize=(10, 6))
    plt.bar(labels, accuracy, color='skyblue')
    plt.xlabel('Zones')
    plt.ylabel('Accuracy')
    plt.title(f'Zone Accuracy ({mode.capitalize()} Error Grid)')
    plt.show()