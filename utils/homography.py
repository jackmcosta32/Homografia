import random

import numpy as np
from scipy.optimize import root

def normalizar(pts):
  x = pts[0, :]
  y = pts[1, :]
  mean_x = np.mean(x)
  mean_y = np.mean(y) #Acha centróide
  dx = (x - mean_x)**2
  dy = (y - mean_y)**2 # Transladando centroide para origem
  s = x.shape[0]*np.sqrt(2)/np.sum(np.sqrt(dx + dy)) # Formula para escalar de raiz de 2
  tx = -s*mean_x # transladar o centroide para origem
  ty = -s*mean_y
  return np.array([[s, 0, tx], [0, s, ty], [0, 0, 1]])

def matA(points, points2): # Gera matriz A
  A = np.zeros((2, 9))
  A[0][3:6] = -points2[2]*points
  A[0][6:] = points2[1]*points
  A[1][0:3] = points2[2]*points
  A[1][6:] = -points2[0]*points
  return A

def Homografia(src, dst, threshold):
    # Inicialização de variaveis
    num_pts = src.shape[-1]
    mask = np.zeros(shape=num_pts, dtype=np.uint8)
    inliers_num = 0

    # Normalizar
    T_src = normalizar(src)  # Matriz de Normalização
    src_n = T_src @ src
    T_dst = normalizar(dst)  # Matriz de Normalização
    T_dst_inv = np.linalg.inv(T_dst)
    dst_n = T_dst @ dst

    # Iteração Variável
    K = 0

    while True:
        # Para pegar Os indices aleatoriamente
        size = random.randint(10, 20)  # usei 15 nos minions, 10 na photo, 12 no outdoor e na comics
        Indice_Aleat = np.random.choice(num_pts, size=size, replace=False)
        src_rnd = src_n[:, Indice_Aleat]
        dst_rnd = dst_n[:, Indice_Aleat]

        # Cálculo da matriz A
        A_l = []
        for s, d in zip(src_rnd.T, dst_rnd.T):
            A_l.append(matA(s, d))  # Gera um vetor 3xN

        A = np.concatenate(A_l, axis=0)  # Modifica com concatenate por não consegui colocar axis=0 na fução append para ficar Nx3

        # Achando a Matriz de homografia
        _, _, Vt = np.linalg.svd(A)  # Matriz singular
        H = Vt[-1, :].reshape(3, 3)  # Botando em 3x3, primeiro valor para H
        H_inv = np.linalg.inv(H)

        # Distância entre os pontos
        loss_dis1 = np.linalg.norm((H @ src_n) - dst_n, axis=0)  # distancia
        loss_dis2 = np.linalg.norm((H_inv @ dst_n) - src_n, axis=0)  # distancia

        # Selecionando Inliers
        mask_index = np.argwhere((loss_dis1 <= threshold) & (
                    loss_dis2 <= threshold))  # Acha os indices em que o erro são menores que o threshold
        it_inliers_num = mask_index.shape[0]  # Variavel que guarda o numero de inliers
        if it_inliers_num > inliers_num:  # Guarda se for a maior quantidade de inliers
            inliers_num = it_inliers_num
            mask[:] = 0
            mask[mask_index] = 1  # Coloca 1 nos indices que representam os inliers

        # Lógica para parar o while
        e = 1 - (inliers_num / num_pts)  # Quantidade de inliers em relação ao numero de pontos

        den_max_K = 1 - (1 - e) ** size  # Denominador da função N que define o numero maximo de iteração

        if (np.round(den_max_K, 3) == 0):
            break

        max_K = np.ceil(np.log(1 - 0.99) / np.log(den_max_K))  # Define o numero de iterações que escolhi p = 0.99

        if K < max_K:  # Se K ainda for menor que o numero de iteração
            K += 1
        else:  # Se não
            break

    # Calculo da nova homografia com todos inliers
    src_in = src_n[:, mask.astype(bool)]  # Indices que receberam 1
    dst_in = dst_n[:, mask.astype(bool)]

    # Nova Matriz A
    A_l = []
    for s, d in zip(src_in.T, dst_in.T):
        A_l.append(matA(s, d))  # Gera um vetor 3xN

    A = np.concatenate(A_l, axis=0)  # Modifica com concatenate por não consegui colocar axis=0 na fução append para ficar Nx3
    _, _, Vt = np.linalg.svd(A)

    # Nova Matriz de Homografia
    H = Vt[-1, :].reshape(3, 3)

    H = H / H[-1, -1]  # Não recomendado aqui h9=1

    return H, T_dst_inv, T_src, src_in, dst_in


def symmetric_transfer_error(H, src, dst):
    H = H.reshape((3, 3))
    H_inv = np.linalg.inv(H)

    dst_pred = np.dot(H, src)
    src_pred = np.dot(H_inv, dst)

    src_loss = np.linalg.norm((src_pred / src_pred[-1, :]) - src, axis=0)
    dst_loss = np.linalg.norm((dst_pred / dst_pred[-1, :]) - dst, axis=0)
    cost = src_loss + dst_loss
    return cost


def optimize(H_ini, src, dst):
    res = root(symmetric_transfer_error, H_ini, args=(src, dst), method='lm', options={'xtol': 1e-08, 'maxiter': 5000})
    H = res.x.reshape((3, 3))
    H = H / H[-1, -1]  # Não recomendado aqui h9=1

    return H

