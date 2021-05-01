import numpy as np
import cv2 as cv
from utils.homography import Homografia, optimize
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def main():
    # Substituindo o exemplo dado pela função de Homografia feita

    # Read images
    img1 = cv.imread('./public/images/book1.jpeg', 0)
    img2 = cv.imread('./public/images/book2.jpeg', 0)

    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN matcher
    flann_matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    matches = flann_matcher.knnMatch(des1, des2, 2)

    # Apply ratio test and store all the good matches
    good = []
    for m, n in matches:
        if m.distance < 0.55 * n.distance:
            good.append(m)

    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1, kp1, img2, kp2, good, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show detected matches
    cv.cvtColor(img_matches, cv.COLOR_BGR2RGB)
    plt.imshow(img_matches)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Transformando em coordenadas Homogeneas [[ 6.94751230e-01 -5.90283385e-01  2.53874465e+02]
        src = src_pts.reshape(-1, 2).T
        src = np.concatenate((src, np.ones((1, src.shape[-1]))))
        dst = dst_pts.reshape(-1, 2).T
        dst = np.concatenate((dst, np.ones((1, dst.shape[-1]))))

        # Calculando a homografia
        M0, T_dst_inv, T_src, src_in, dst_in = Homografia(src, dst, 7.0)  # O threshold de 6 me deu um resultado melhor nos minions
        print('Homografia com Ransac: \n', M0)

        # Otimizando
        M = optimize(M0, src_in, dst_in)
        print('Homography optimized with Symmetric Transfer Error: \n', M)

        M = T_dst_inv @ M @ T_src  # Denormalizando
        print('Homografia Denormalizada: \n', M)

        img3 = cv.warpPerspective(img2, np.linalg.inv(M), (img2.shape[1], img2.shape[0]))
        img4 = cv.warpPerspective(img1, M, (img1.shape[1], img1.shape[0]))
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(img1, 'gray')
    axs[0, 0].set_title('Primeira Imagem')

    axs[1, 0].imshow(img2, 'gray')
    axs[1, 0].set_title('Segunda Imagem')

    axs[0, 1].imshow(img4, 'gray')
    axs[0, 1].set_title('Primeira Imagem depois da transformação')

    axs[1, 1].imshow(img3, 'gray')
    axs[1, 1].set_title('Segunda imagem depois da transformação')

    plt.show()


if __name__ == '__main__':
    main()
