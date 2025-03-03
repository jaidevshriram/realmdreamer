import numpy as np
import torch
import torch.nn.functional as F
from kornia.geometry.conversions import (quaternion_to_rotation_matrix,
                                         rotation_matrix_to_quaternion)
from torchtyping import TensorType

# From - https://github.com/gsgen3d/gsgen/blob/e60994e557b685d5df57ec0bb48460782091ec4d/utils/transforms.py#L53


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def qsvec2rotmat_batched(qvec: TensorType["N", 4], svec: TensorType["N", 3]) -> TensorType["N", 3, 3]:
    unscaled_rotmat = quaternion_to_rotation_matrix(qvec)

    # TODO: check which I current think that scale should be copied row-wise since in eq (6) the S matrix is right-hand multplied to R
    rotmat = svec.unsqueeze(-2) * unscaled_rotmat
    # rotmat = svec.unsqueeze(-1) * unscaled_rotmat
    # rotmat = torch.bmm(unscaled_rotmat, torch.diag(svec))

    # print("rotmat", rotmat.shape)

    return rotmat


def rotmat2wxyz(rotmat):
    return rotation_matrix_to_quaternion(rotmat)


def qvec2rotmat_batched(qvec: TensorType["N", 4]):
    return quaternion_to_rotation_matrix(qvec)


def qsvec2covmat_batched(qvec: TensorType["N", 4], svec: TensorType["N", 3]):
    rotmat = qsvec2rotmat_batched(qvec, svec)
    return torch.bmm(rotmat, rotmat.transpose(-1, -2))
