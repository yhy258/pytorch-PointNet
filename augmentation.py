import random
import math

import numpy as np

import torch

class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):  # 각 포인트에 대한 area 구하기
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5  # 삼각형이 안 이뤄질수도 있다. -> 음수가 나옴.

    def sample_point(self, pt1, pt2, pt3):
        # 왜? 내가 생각하기에는 포인트 클라우드를 구성 할 때 꼭 모든 포인트에 대해 가져올 필요가 없음. 좀 더 적은 수로 가져와도 괜찮다.
        # 그래서 어느정도 노이즈를 통해서 내가 원하는 수의 포인트만을 가져온다. (형태를 유지시킬 수 있으면서.)

        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html

        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))

        sampled_faces = (random.choices(faces,
                                        weights=areas,
                                        cum_weights=None,
                                        k=self.output_size))

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))

        return sampled_points


# PointCloud 데이터를 unit sphere로 ㄱㄱ (중앙이 0, 0, 0에 있는.)
class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud


class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        # Rotation
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                               [math.sin(theta), math.cos(theta), 0],
                               [0, 0, 1]])

        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud


class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))

        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud


class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        return torch.from_numpy(pointcloud)
