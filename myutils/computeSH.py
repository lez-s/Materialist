# https://github.com/lzqsd/SingleImageShapeAndSVBRDF/blob/master/computeSH.py
import cv2
import numpy as np
from PIL import Image
import scipy.io as io
import torch
import global_config
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import math
import os

def P_0_0(theta):
    return np.ones(theta.shape, dtype=np.float32)

def P_1_0(theta):
    return np.cos(theta)

def P_1_1(theta):
    return -np.sin(theta)

def P_2_0(theta):
    return 0.5 * (3 * np.power(np.cos(theta), 2) - 1)

def P_2_1(theta):
    return -3 * np.cos(theta) * np.sin(theta)

def P_2_2(theta):
    return 3 * np.power(np.sin(theta), 2.0)

def P_3_0(theta):
    return 0.5 * (5 * np.power(np.cos(theta), 3) - 3 * np.cos(theta) )

def P_3_1(theta):
    return -1.5 * (5 * np.power(np.cos(theta), 2) - 1) * np.sin(theta)

def P_3_2(theta):
    return 15 * np.cos(theta) * np.power(np.sin(theta), 2)

def P_3_3(theta):
    return -15 * np.power(np.sin(theta), 3)

def P_4_0(theta):
    return 0.125 * (35 * np.power(np.cos(theta), 4) - 30 * np.power(np.cos(theta), 2) + 3)

def P_4_1(theta):
    return -2.5 * (7 * np.power(np.cos(theta), 3) - 3 * np.cos(theta) ) * np.sin(theta)

def P_4_2(theta):
    return 7.5 * (7 * np.power(np.cos(theta), 2) - 1) * np.power(np.sin(theta), 2)

def P_4_3(theta):
    return -105 * np.cos(theta) * np.power(np.sin(theta), 3)

def P_4_4(theta):
    return 105 * np.power(np.sin(theta), 4)

def computeK(l, m):
    m = np.absolute(m)
    l_s_m = l - m
    l_a_m = l + m
    for n in range(0, len(l) ):
        l_s_m[n] = np.math.factorial(l_s_m[n])
        l_a_m[n] = np.math.factorial(l_a_m[n])
    l_s_m = l_s_m.astype(np.float32)
    l_a_m = l_a_m.astype(np.float32)
    K2 = (2*l + 1) * l_s_m /l_a_m / 4 / np.pi
    return np.sqrt(K2)

def angleToUV(theta, phi):
    u = (phi + np.pi) / 2 / np.pi
    v = 1 - theta / np.pi
    return u, v

def uvToEnvmap(envmap, u, v):
    height, width = envmap.shape[0], envmap.shape[1]
    c, r = u * (width-1), (1-v) * (height-1)
    cs, rs = int(c), int(r)
    ce = min(width-1, cs + 1)
    re = min(height-1, rs + 1)
    wc, wr = c - cs, r - rs
    color1 = (1-wc) * envmap[rs, cs, :] + wc * envmap[rs, ce, :]
    color2 = (1-wc) * envmap[re, cs, :] + wc * envmap[re, ce, :]
    color = (1 - wr) * color1 + wr * color2
    return color

def Integration(phi, colors, K, LP):
    sampleNum = colors.shape[0]
    W = 4 * np.pi / sampleNum

    if len(phi.shape) == 1:
        phi = phi[:, np.newaxis]

    # level 0
    y_0 = K[0] * W * LP['P_0_0'] * colors
    y_0 = np.sum(y_0, axis=0)[np.newaxis, :]

    # level 1
    y_1_n1 = np.sqrt(2) * K[1] * np.sin(1 * phi) * W * LP['P_1_1'] * colors
    y_1_0 =  K[2] * W * LP['P_1_0'] * colors
    y_1_p1 = np.sqrt(2) * K[3] * np.cos(1 * phi) * W * LP['P_1_1'] * colors

    y_1_n1 = np.sum(y_1_n1, axis=0)[np.newaxis, :]
    y_1_0 = np.sum(y_1_0, axis=0)[np.newaxis, :]
    y_1_p1 = np.sum(y_1_p1, axis=0)[np.newaxis, :]

    # level 2
    y_2_n2 = np.sqrt(2) * K[4] * np.sin(2 * phi) * W * LP['P_2_2'] * colors
    y_2_n1 = np.sqrt(2) * K[5] * np.sin(1 * phi) * W * LP['P_2_1'] * colors
    y_2_0 = K[6] * W * LP['P_2_0'] * colors
    y_2_p1 = np.sqrt(2) * K[7] * np.cos(1 * phi) * W * LP['P_2_1'] * colors
    y_2_p2 = np.sqrt(2) * K[8] * np.cos(2 * phi) * W * LP['P_2_2'] * colors

    y_2_n2 = np.sum(y_2_n2, axis=0)[np.newaxis, :]
    y_2_n1 = np.sum(y_2_n1, axis=0)[np.newaxis, :]
    y_2_0 = np.sum(y_2_0, axis=0)[np.newaxis, :]
    y_2_p1 = np.sum(y_2_p1, axis=0)[np.newaxis, :]
    y_2_p2 = np.sum(y_2_p2, axis=0)[np.newaxis, :]

    # level 3
    y_3_n3 = np.sqrt(2) * K[9 ] * np.sin(3 * phi) * W * LP['P_3_3'] * colors
    y_3_n2 = np.sqrt(2) * K[10] * np.sin(2 * phi) * W * LP['P_3_2'] * colors
    y_3_n1 = np.sqrt(2) * K[11] * np.sin(1 * phi) * W * LP['P_3_1'] * colors
    y_3_0 = K[12] * W * LP['P_3_0'] * colors
    y_3_p1 = np.sqrt(2) * K[13] * np.cos(1 * phi) * W * LP['P_3_1'] * colors
    y_3_p2 = np.sqrt(2) * K[14] * np.cos(2 * phi) * W * LP['P_3_2'] * colors
    y_3_p3 = np.sqrt(2) * K[15] * np.cos(3 * phi) * W * LP['P_3_3'] * colors

    y_3_n3 = np.sum(y_3_n3, axis=0)[np.newaxis, :]
    y_3_n2 = np.sum(y_3_n2, axis=0)[np.newaxis, :]
    y_3_n1 = np.sum(y_3_n1, axis=0)[np.newaxis, :]
    y_3_0 = np.sum(y_3_0, axis=0)[np.newaxis, :]
    y_3_p1 = np.sum(y_3_p1, axis=0)[np.newaxis, :]
    y_3_p2 = np.sum(y_3_p2, axis=0)[np.newaxis, :]
    y_3_p3 = np.sum(y_3_p3, axis=0)[np.newaxis, :]

    # level 4
    y_4_n4 = np.sqrt(2) * K[16] * np.sin(4 * phi) * W * LP['P_4_4'] * colors
    y_4_n3 = np.sqrt(2) * K[17] * np.sin(3 * phi) * W * LP['P_4_3'] * colors
    y_4_n2 = np.sqrt(2) * K[18] * np.sin(2 * phi) * W * LP['P_4_2'] * colors
    y_4_n1 = np.sqrt(2) * K[19] * np.sin(1 * phi) * W * LP['P_4_1'] * colors
    y_4_0 = K[20] * W * LP['P_4_0'] * colors
    y_4_p1 = np.sqrt(2) * K[21] * np.cos(1 * phi) * W * LP['P_4_1'] * colors
    y_4_p2 = np.sqrt(2) * K[22] * np.cos(2 * phi) * W * LP['P_4_2'] * colors
    y_4_p3 = np.sqrt(2) * K[23] * np.cos(3 * phi) * W * LP['P_4_3'] * colors
    y_4_p4 = np.sqrt(2) * K[24] * np.cos(4 * phi) * W * LP['P_4_4'] * colors

    y_4_n4 = np.sum(y_4_n4, axis=0)[np.newaxis, :]
    y_4_n3 = np.sum(y_4_n3, axis=0)[np.newaxis, :]
    y_4_n2 = np.sum(y_4_n2, axis=0)[np.newaxis, :]
    y_4_n1 = np.sum(y_4_n1, axis=0)[np.newaxis, :]
    y_4_0 = np.sum(y_4_0, axis=0)[np.newaxis, :]
    y_4_p1 = np.sum(y_4_p1, axis=0)[np.newaxis, :]
    y_4_p2 = np.sum(y_4_p2, axis=0)[np.newaxis, :]
    y_4_p3 = np.sum(y_4_p3, axis=0)[np.newaxis, :]
    y_4_p4 = np.sum(y_4_p4, axis=0)[np.newaxis, :]

    return np.concatenate([y_0,
                           y_1_n1, y_1_0, y_1_p1,
                           y_2_n2, y_2_n1, y_2_0, y_2_p1, y_2_p2,
                           y_3_n3, y_3_n2, y_3_n1, y_3_0, y_3_p1, y_3_p2, y_3_p3,
                           y_4_n4, y_4_n3, y_4_n2, y_4_n1, y_4_0, y_4_p1, y_4_p2, y_4_p3, y_4_p4], axis=0)


def projection(phi, theta, K, coef):
    phi = phi[:, np.newaxis]
    p_0_0 = P_0_0(theta)[:, np.newaxis]
    p_1_0 = P_1_0(theta)[:, np.newaxis]
    p_1_1 = P_1_1(theta)[:, np.newaxis]
    p_2_0 = P_2_0(theta)[:, np.newaxis]
    p_2_1 = P_2_1(theta)[:, np.newaxis]
    p_2_2 = P_2_2(theta)[:, np.newaxis]
    p_3_0 = P_3_0(theta)[:, np.newaxis]
    p_3_1 = P_3_1(theta)[:, np.newaxis]
    p_3_2 = P_3_2(theta)[:, np.newaxis]
    p_3_3 = P_3_3(theta)[:, np.newaxis]
    p_4_0 = P_4_0(theta)[:, np.newaxis]
    p_4_1 = P_4_1(theta)[:, np.newaxis]
    p_4_2 = P_4_2(theta)[:, np.newaxis]
    p_4_3 = P_4_3(theta)[:, np.newaxis]
    p_4_4 = P_4_4(theta)[:, np.newaxis]

    # level 0
    y_0 = np.matmul(K[0] *  p_0_0, coef[0:1, :])

    # level 1
    y_1_n1 = np.matmul(np.sqrt(2) * K[1] * np.sin(1 * phi) * p_1_1, coef[1:2, :])
    y_1_0 =  np.matmul(K[2] * p_1_0, coef[2:3, :])
    y_1_p1 = np.matmul(np.sqrt(2) * K[3] * np.cos(1 * phi) * p_1_1, coef[3:4, :])

    # level 2
    y_2_n2 = np.matmul(np.sqrt(2) * K[4] * np.sin(2 * phi) * p_2_2, coef[4:5, :])
    y_2_n1 = np.matmul(np.sqrt(2) * K[5] * np.sin(1 * phi) * p_2_1, coef[5:6, :])
    y_2_0 = np.matmul(K[6] * p_2_0, coef[6:7, :])
    y_2_p1 = np.matmul(np.sqrt(2) * K[7] * np.cos(1 * phi) * p_2_1, coef[7:8, :])
    y_2_p2 = np.matmul(np.sqrt(2) * K[8] * np.cos(2 * phi) * p_2_2, coef[8:9, :])

    # level 3
    y_3_n3 = np.matmul(np.sqrt(2) * K[9 ] * np.sin(3 * phi) * p_3_3, coef[ 9:10, :])
    y_3_n2 = np.matmul(np.sqrt(2) * K[10] * np.sin(2 * phi) * p_3_2, coef[10:11, :])
    y_3_n1 = np.matmul(np.sqrt(2) * K[11] * np.sin(1 * phi) * p_3_1, coef[11:12, :])
    y_3_0 = np.matmul(K[12] * p_3_0, coef[12:13, :])
    y_3_p1 = np.matmul(np.sqrt(2) * K[13] * np.cos(1 * phi) * p_3_1, coef[13:14, :])
    y_3_p2 = np.matmul(np.sqrt(2) * K[14] * np.cos(2 * phi) * p_3_2, coef[14:15, :])
    y_3_p3 = np.matmul(np.sqrt(2) * K[15] * np.cos(3 * phi) * p_3_3, coef[15:16, :])

    # level 4
    y_4_n4 = np.matmul(np.sqrt(2) * K[16] * np.sin(4 * phi) * p_4_4, coef[16:17, :])
    y_4_n3 = np.matmul(np.sqrt(2) * K[17] * np.sin(3 * phi) * p_4_3, coef[17:18, :])
    y_4_n2 = np.matmul(np.sqrt(2) * K[18] * np.sin(2 * phi) * p_4_2, coef[18:19, :])
    y_4_n1 = np.matmul(np.sqrt(2) * K[19] * np.sin(1 * phi) * p_4_1, coef[19:20, :])
    y_4_0 = np.matmul(K[20] * p_4_0, coef[20:21, :])
    y_4_p1 = np.matmul(np.sqrt(2) * K[21] * np.cos(1 * phi) * p_4_1, coef[21:22, :])
    y_4_p2 = np.matmul(np.sqrt(2) * K[22] * np.cos(2 * phi) * p_4_2, coef[22:23, :])
    y_4_p3 = np.matmul(np.sqrt(2) * K[23] * np.cos(3 * phi) * p_4_3, coef[23:24, :])
    y_4_p4 = np.matmul(np.sqrt(2) * K[24] * np.cos(4 * phi) * p_4_4, coef[24:25, :])

    img = y_0 \
          + y_1_n1 + y_1_0 + y_1_p1 \
          + y_2_n2 + y_2_n1 + y_2_0 + y_2_p1 + y_2_p2 \
          + y_3_n3 + y_3_n2 + y_3_n1 + y_3_0 + y_3_p1 + y_3_p2 + y_3_p3 \
          + y_4_n4 + y_4_n3 + y_4_n2 + y_4_n1 + y_4_0 + y_4_p1 + y_4_p2 + y_4_p3 + y_4_p4

    return img

def reconstImageFromSH(coef, nrows, ncols, K = None, isClip = True):
    if K is None:
        larr = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, \
                4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=np.int32)
        marr = np.array([0, -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3, \
                -4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.int32)
        K = computeK(larr, marr)
    x, y = np.meshgrid(np.linspace(-1, 1, ncols+1), np.linspace(0, 1, nrows+1) )
    phi, theta = np.pi * x, np.pi * y
    phi, theta = phi[0:nrows, 0:ncols], theta[0:nrows, 0:ncols]
    img = projection(phi.reshape(-1), theta.reshape(-1), K, coef)
    img = img.reshape([nrows, ncols, 3] )
    if isClip:
        img = np.clip(img, 0, 1)
    return img

def reconstImageFromSHAfterRotate(coef, cameraLoc, cameraUp, nrows=512, ncols=1024, K = None, isClip = True, isInv = False):
    if K is None:
        larr = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, \
                4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=np.int32)
        marr = np.array([0, -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3, \
                -4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.int32)
        K = computeK(larr, marr)
    x, y = np.meshgrid(np.linspace(-1, 1, ncols+1), np.linspace(0, 1, nrows+1) )
    phi, theta = np.pi * x, np.pi * y
    phi, theta = phi[0:nrows, 0:ncols], theta[0:nrows, 0:ncols]
    envmap = projection(phi.reshape(-1), theta.reshape(-1), K, coef)
    envmap = envmap.reshape([nrows, ncols, 3] )
    if isClip:
        envmap = np.clip(envmap, 0, 1)

    cameraLoc = np.asarray(cameraLoc, dtype=np.float32)
    cameraUp = np.asarray(cameraUp, dtype=np.float32)
    cameraLoc = cameraLoc / np.sqrt(np.sum(cameraLoc * cameraLoc), dtype=np.float32)
    cameraUp = cameraUp / np.sqrt(np.sum(cameraUp * cameraUp), dtype=np.float32)
    rz, ry = cameraLoc, cameraUp
    rx = np.cross(ry, rz)
    rx = rx / np.sqrt(np.sum(rx * rx) )
    ry = np.cross(rz, rx)
    ry = ry / np.sqrt(np.sum(ry * ry) )

    if isInv == True:
        rx, ry, rz = rx[:, np.newaxis], ry[:, np.newaxis], rz[:, np.newaxis]
        rotMatrix = np.concatenate([rx, ry, rz], axis=1)
        rotMatrix = rotMatrix.transpose([1, 0] )
        rx, ry, rz = rotMatrix[:, 0], rotMatrix[:, 1], rotMatrix[:, 2]

    envmapRot = np.zeros(envmap.shape, dtype=np.float32)
    height, width = envmapRot.shape[0], envmapRot.shape[1]
    for r in range(0, height):
        for c in range(0, width):
            theta = r / float(height-1) * np.pi
            phi = c / float(width) * np.pi * 2 - np.pi
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            coord = x * rx + y * ry + z * rz
            nx, ny, nz = coord[0], coord[1], coord[2]
            thetaNew = np.arccos(nz)
            nx = nx / (np.sqrt(1-nz*nz) + 1e-12)
            ny = ny / (np.sqrt(1-nz*nz) + 1e-12)
            nx = np.clip(nx, -1, 1)
            ny = np.clip(ny, -1, 1)
            nz = np.clip(nz, -1, 1)
            phiNew = np.arccos(nx)
            if ny < 0:
                phiNew = - phiNew
            u, v = angleToUV(thetaNew, phiNew)
            color = uvToEnvmap(envmap, u, v)
            envmapRot[r, c, :] = color

    return envmapRot

def computeSHFromImage(im):
    nSampleX, nSampleY = im.shape[1], im.shape[0]
    angles = np.zeros([nSampleX * nSampleY, 2] )
    for r in range(nSampleY):
        for c in range(nSampleX):
            ind = r * nSampleX + c
            y = (r + np.random.random() ) / float(nSampleY)
            x = (c + np.random.random() ) / float(nSampleX)
            phi = 2 * np.pi * x - np.pi
            theta = 2 * np.arccos( np.sqrt(1 - y) )
            angles[ind, 0] = theta
            angles[ind, 1] = phi

    # Compute the coef for spherical Harmonics
    larr = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, \
            4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=np.int32)
    marr = np.array([0, -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3, \
            -4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.int32)
    karr = computeK(larr, marr)

    # Compute Associated Legendre Polynomial
    LP = {}
    LP['P_0_0'] = P_0_0(angles[:, 0])[:, np.newaxis]
    LP['P_1_0'] = P_1_0(angles[:, 0])[:, np.newaxis]
    LP['P_1_1'] = P_1_1(angles[:, 0])[:, np.newaxis]
    LP['P_2_0'] = P_2_0(angles[:, 0])[:, np.newaxis]
    LP['P_2_1'] = P_2_1(angles[:, 0])[:, np.newaxis]
    LP['P_2_2'] = P_2_2(angles[:, 0])[:, np.newaxis]
    LP['P_3_0'] = P_3_0(angles[:, 0])[:, np.newaxis]
    LP['P_3_1'] = P_3_1(angles[:, 0])[:, np.newaxis]
    LP['P_3_2'] = P_3_2(angles[:, 0])[:, np.newaxis]
    LP['P_3_3'] = P_3_3(angles[:, 0])[:, np.newaxis]
    LP['P_4_0'] = P_4_0(angles[:, 0])[:, np.newaxis]
    LP['P_4_1'] = P_4_1(angles[:, 0])[:, np.newaxis]
    LP['P_4_2'] = P_4_2(angles[:, 0])[:, np.newaxis]
    LP['P_4_3'] = P_4_3(angles[:, 0])[:, np.newaxis]
    LP['P_4_4'] = P_4_4(angles[:, 0])[:, np.newaxis]


    colors = []
    # Bilinear interpolate the color in Enivronment map
    for n in range(angles.shape[0] ):
        theta, phi = angles[n, 0], angles[n, 1]
        u, v = angleToUV(theta, phi)
        color = uvToEnvmap(im, u, v).squeeze()[np.newaxis, :]
        colors.append(color)
    colors = np.concatenate(colors, axis=0)
    coef = Integration(angles[:, 1], colors, karr, LP)
    return coef

def computeSHFromImageAfterRotate(envmap, cameraLoc, cameraUp, isInv=False):
    cameraLoc = np.asarray(cameraLoc, dtype=np.float32)
    cameraUp = np.asarray(cameraUp, dtype=np.float32)
    cameraLoc = cameraLoc / np.sqrt(np.sum(cameraLoc * cameraLoc), dtype=np.float32)
    cameraUp = cameraUp / np.sqrt(np.sum(cameraUp * cameraUp), dtype=np.float32)
    rz, ry = cameraLoc, cameraUp
    rx = np.cross(ry, rz)
    rx = rx / np.sqrt(np.sum(rx * rx) )
    ry = np.cross(rz, rx)
    ry = ry / np.sqrt(np.sum(ry * ry) )

    if isInv == True:
        rx, ry, rz = rx[:, np.newaxis], ry[:, np.newaxis], rz[:, np.newaxis]
        rotMatrix = np.concatenate([rx, ry, rz], axis=1)
        rotMatrix = rotMatrix.transpose([1, 0] )
        rx, ry, rz = rotMatrix[:, 0], rotMatrix[:, 1], rotMatrix[:, 2]

    envmapRot = np.zeros(envmap.shape, dtype=np.float32)
    height, width = envmapRot.shape[0], envmapRot.shape[1]
    for r in range(0, height):
        for c in range(0, width):
            theta = r / float(height-1) * np.pi
            phi = c / float(width) * np.pi * 2 - np.pi
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            coord = x * rx + y * ry + z * rz
            nx, ny, nz = coord[0], coord[1], coord[2]
            thetaNew = np.arccos(nz)
            nx = nx / (np.sqrt(1-nz*nz) + 1e-12)
            ny = ny / (np.sqrt(1-nz*nz) + 1e-12)
            nx = np.clip(nx, -1, 1)
            ny = np.clip(ny, -1, 1)
            nz = np.clip(nz, -1, 1)
            phiNew = np.arccos(nx)
            if ny < 0:
                phiNew = - phiNew
            u, v = angleToUV(thetaNew, phiNew)
            color = uvToEnvmap(envmap, u, v)
            envmapRot[r, c, :] = color

    coef = computeSHFromImage(envmapRot)
    return coef



def compute_sh_coefficients(hdr_map, num_samples):
    theta, phi = np.mgrid[0:np.pi:num_samples*1j, 0:2*np.pi:num_samples*2j]
    sin_theta = np.sin(theta)
    
    coeff = np.zeros((9, 3)) 
    
    for i in range(hdr_map.shape[0]):
        for j in range(hdr_map.shape[1]):
            color = hdr_map[i, j]
            for l in range(3): 
                for m in range(-l, l+1):
                    Y_lm = sph_harm(m, l, phi[i, j], theta[i, j])
                    coeff[l*(l+1)+m] += color * Y_lm.real * sin_theta[i, j]
    
    return coeff
def compute_sh_coeff_torch(hdr_image, l_max=2):
    device = torch.device('cuda')
    height, width, _ = hdr_image.shape
    coeffs = torch.zeros((l_max+1, 2*l_max+1, 3), dtype=torch.float32,device=device)  
    phis = torch.linspace(0, 2 * math.pi, width) 
    thetas = torch.linspace(0, math.pi, height)
    phis, thetas = torch.meshgrid(phis, thetas, indexing='xy')
    sin_thetas = torch.sin(thetas).to(device)

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Ylm = torch_sph_harm(m, l, phis, thetas).to(device)

            Ylm_sin_theta = Ylm * sin_thetas

            for c in range(3): 
                x = torch.sum(hdr_image[:, :, c] * Ylm_sin_theta)
                coeffs[l, m + l, c] += x 
    normalization_factor = 1
    coeffs *= 4 * math.pi / (width * height) * normalization_factor
    return coeffs

def compute_sh_coeff_minh(hdr_image, l_max=2):
    height, width, _ = hdr_image.shape
    coeffs = np.zeros((l_max+1, l_max+1, 3), dtype=np.complex64)

    phis = np.array([np.pi * (x + 0.5) / width for x in range(width)])
    thetas = np.array([np.pi * (x + 0.5) /height for x in range(height)])
    phis, thetas = np.array(np.meshgrid(phis, thetas))
    sin_thetas = np.sin(thetas)

    sqrt_2 = np.sqrt(2)
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            if m >= 0:
                Ylm = sqrt_2 * (-1)**m * lpmv(m, l, np.cos(thetas)) * np.sin(m * phis) if m != 0 else lpmv(m, l, np.cos(thetas))
            else:
                Ylm = sqrt_2 * (-1)**m * lpmv(-m, l, np.cos(thetas)) * np.cos(-m * phis)

            Ylm_sin_theta = np.multiply(Ylm, sin_thetas)
            for c in range(3): 
                x = np.sum(np.multiply(hdr_image[:, :, c], Ylm_sin_theta))
                coeffs[l, m, c] += x


    coeffs *= 4 * np.pi / (width * height)
    return coeffs.real 

def legendre_polynomial(m, l, x):
    if l == m:
        return (-1)**m * (2**m) * torch.prod(torch.sqrt(1 - x**2))**m
    elif l == m + 1:
        return x * (2*m + 1) * legendre_polynomial(m, m, x)
    else:
        P_lm_minus_1 = legendre_polynomial(m, l-1, x)
        P_lm_minus_2 = legendre_polynomial(m, l-2, x)
        return ((2*l - 1) * x * P_lm_minus_1 - (l + m - 1) * P_lm_minus_2) / (l - m)

def torch_sph_harm(m, l, phi, theta):
    cos_theta = torch.cos(theta)
    P_lm = legendre_polynomial(abs(m), l, cos_theta)
    # K = math.sqrt((2*l + 1) / (4*math.pi) * math.factorial(l - abs(m)) / math.factorial(l + abs(m)))
    K = computeK(l, m)
    if m > 0:
        return math.sqrt(2) * torch.cos(m * phi) * P_lm * K
    elif m < 0:
        return math.sqrt(2) * torch.sin(-m * phi) * P_lm * K
    else:
        return P_lm * K

def reconstruct_envmap_from_sh(sh_coeffs, width, height, l_max=2):
    device = torch.device('cuda')
    envmap = torch.zeros((height, width, 3), dtype=torch.float32,device=device)
    
    phis = torch.linspace(0, 2 * math.pi, width)  # Note that the phi range here is usually [0, 2*pi]
    thetas = torch.linspace(0, math.pi, height)
    phis, thetas = torch.meshgrid(phis, thetas, indexing='xy')

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Ylm = torch_sph_harm(m, l, phis, thetas).to(device)
            for c in range(3):  
                envmap[:, :, c] += sh_coeffs[l, m + l, c] * Ylm

    return envmap



