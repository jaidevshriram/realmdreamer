# distutils: language = c++

cimport cython
from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.vector cimport vector


cdef extern from "find_occluded_voxels.h":
    vector[vector[vector[bool]]] findOccludedVoxels(const vector[vector[vector[bool]]]& grid, const vector[int]& viewpoint, const vector[int]& min_corner, float voxel_size, int gridSizeX, int gridSizeY, int gridSizeZ)

@cython.boundscheck(False)
@cython.wraparound(False)
def find_occluded_voxels(grid, viewpoint, min_corner, voxel_size, sizex, sizey, sizez):
    cdef vector[vector[vector[bool]]] c_grid = grid  # Convert Python lists to C++ vectors
    cdef vector[int] c_viewpoint = viewpoint
    cdef vector[int] c_min_corner = min_corner
    cdef float c_voxel_size = voxel_size
    cdef int c_sizex = sizex
    cdef int c_sizey = sizey
    cdef int c_sizez = sizez
    return findOccludedVoxels(c_grid, c_viewpoint, c_min_corner, c_voxel_size, sizex, sizey, sizez)
