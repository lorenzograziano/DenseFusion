import math

import numpy as np


def normalize_quaternion(q):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return q / n


def q_to_matrix(q):
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    dcm = np.zeros((3, 3))

    dcm[0, 0] = x2 - y2 - z2 + w2
    # print(str(1-2*y2-2*z2))
    dcm[1, 0] = 2 * (xy + zw)
    dcm[2, 0] = 2 * (xz - yw)

    dcm[0, 1] = 2 * (xy - zw)
    dcm[1, 1] = - x2 + y2 - z2 + w2
    # print(str(1-2*x2-2*z2))
    dcm[2, 1] = 2 * (yz + xw)

    dcm[0, 2] = 2 * (xz + yw)
    dcm[1, 2] = 2 * (yz - xw)
    dcm[2, 2] = - x2 - y2 + z2 + w2
    # print(str(1-2*x2-2*y2))

    return dcm


def convert_coordinates(six_dof, camera_coord):
    quaternion = six_dof[0:4]
    transl = six_dof[4:7]

    # print("quaternion", quaternion)
    # print("transl", transl)

    n_quaternion = normalize_quaternion(quaternion)
    # print("normalized quaternion", n_quaternion)

    rotation_matrix = q_to_matrix(n_quaternion)
    # print(rotation_matrix)

    n, m = (np.shape(camera_coord))
    objcoord = np.zeros(shape=(n, m))

    t_matrix = np.zeros(shape=(4, 4))

    t_matrix[0:3, 0:3] = rotation_matrix
    t_matrix[3, 3] = 1
    t_matrix[0:3, 3] = transl
    print(t_matrix)
    # - cam_R_m2c: [0.08260540, 0.98752999, -0.13401900, 0.74214798, -0.15070900, -0.65307200, -0.66512603, -0.04551440,
    #               -0.74534303]
    # cam_t_m2c: [142.58925727, -133.41315293, 1000.32204526]

    # t_matrix[0, 0] = 0.08260540
    # t_matrix[0, 1] = 0.98752999
    # t_matrix[0, 2] = -0.13401900
    # t_matrix[1, 0] = 0.74214798
    # t_matrix[1, 1] = -0.15070900
    # t_matrix[1, 2] = -0.65307200
    # t_matrix[2, 0] = -0.6651260
    # t_matrix[2, 1] = -0.04551440
    # t_matrix[2, 2] = -0.74534303
    # t_matrix[0, 3] =0.14258925727
    # t_matrix[1, 3] =-0.13341315293
    # t_matrix[2, 3] =1.00032204526
    # print("eheheh\n" ,t_matrix)
    # # print(camera_coord[:,0])
    # print(camera_coord[:,10])
    # print(camera_coord[:,20])
    # print(camera_coord[:,30])




    for i in range(m):
        d = camera_coord[:, i]
        c_d = np.transpose(np.concatenate((d, np.ones(1))))
        objcoord[:, i] = (t_matrix.dot(c_d))[0:3]
    print(objcoord)
    return objcoord


def convert3dpointto2d(coord3d):
    n, m = (np.shape(coord3d))
    coord2d = np.zeros(shape=(2, m))

    for i in range(m):
        # distanza telecamera = norma vettore t ????
        magn = np.linalg.norm(coord3d[:, i])

        coord2d[0, i] = coord3d[0, i] * 640 / coord3d[2, i] * magn + 320
        coord2d[1, i] = coord3d[1, i] * 480 / coord3d[2, i] * magn + 240
        # coord2d[0, i] = (coord3d[0, i] * 640) / (2.0 * coord3d[2, i]) + 320
        # coord2d[1, i] = (coord3d[1, i] * 480) / (2.0 * coord3d[2, i]) + 240
    return coord2d
