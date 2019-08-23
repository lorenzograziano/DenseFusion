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

    n_quaternion = normalize_quaternion(quaternion)
    rotation_matrix = q_to_matrix(n_quaternion)

    n, m = (np.shape(camera_coord))
    objcoord = np.zeros(shape=(n, m))

    t_matrix = np.zeros(shape=(4, 4))

    t_matrix[0:3, 0:3] = rotation_matrix
    t_matrix[3, 3] = 1
    t_matrix[0:3, 3] = transl
    print(t_matrix)

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

    return coord2d
