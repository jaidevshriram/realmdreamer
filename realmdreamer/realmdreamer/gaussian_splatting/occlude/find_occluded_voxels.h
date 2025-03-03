#include <vector>
#include <cmath>
#include <bits/stdc++.h>

int roundup(float x) {
    return std::ceil(x);
}

std::vector<std::tuple<int, int, int>> Bresenham3D(int x1, int y1, int z1, int x2, int y2, int z2) {
    std::vector<std::tuple<int, int, int>> listOfPoints;
    int dx = std::abs(x2 - x1);
    int dy = std::abs(y2 - y1);
    int dz = std::abs(z2 - z1);
    int xs = (x2 > x1) ? 1 : -1;
    int ys = (y2 > y1) ? 1 : -1;
    int zs = (z2 > z1) ? 1 : -1;

    if (dx >= dy && dx >= dz) {
        int p1 = 2 * dy - dx;
        int p2 = 2 * dz - dx;
        while (x1 != x2) {
            x1 += xs;
            if (p1 >= 0) {
                y1 += ys;
                p1 -= 2 * dx;
            }
            if (p2 >= 0) {
                z1 += zs;
                p2 -= 2 * dx;
            }
            p1 += 2 * dy;
            p2 += 2 * dz;
            listOfPoints.push_back(std::make_tuple(x1, y1, z1));
        }
    } else if (dy >= dx && dy >= dz) {
        int p1 = 2 * dx - dy;
        int p2 = 2 * dz - dy;
        while (y1 != y2) {
            y1 += ys;
            if (p1 >= 0) {
                x1 += xs;
                p1 -= 2 * dy;
            }
            if (p2 >= 0) {
                z1 += zs;
                p2 -= 2 * dy;
            }
            p1 += 2 * dx;
            p2 += 2 * dz;
            listOfPoints.push_back(std::make_tuple(x1, y1, z1));
        }
    } else {
        int p1 = 2 * dy - dz;
        int p2 = 2 * dx - dz;
        while (z1 != z2) {
            z1 += zs;
            if (p1 >= 0) {
                y1 += ys;
                p1 -= 2 * dz;
            }
            if (p2 >= 0) {
                x1 += xs;
                p2 -= 2 * dz;
            }
            p1 += 2 * dy;
            p2 += 2 * dx;
            listOfPoints.push_back(std::make_tuple(x1, y1, z1));
        }
    }

    return listOfPoints;
}

std::vector<std::vector<std::vector<bool>>> findOccludedVoxels(
    const std::vector<std::vector<std::vector<bool>>>& grid,
    const std::vector<int> viewpoint,
    const std::vector<int> min_corner,
    float voxel_size,
    int gridSizeX, 
    int gridSizeY,
    int gridSizeZ) {

    std::vector<int> origin_grid = {
        std::floor((viewpoint[0] - min_corner[0]) / voxel_size),
        std::floor((viewpoint[1] - min_corner[1]) / voxel_size),
        std::floor((viewpoint[2] - min_corner[2]) / voxel_size)
    };

    std::vector<std::vector<std::vector<bool>>> seen3d(gridSizeX, std::vector<std::vector<bool>>(gridSizeY, std::vector<bool>(gridSizeZ, 0)));

    // Go around the edges of the voxel grid - all faces of the grid and draw lines to them
    for (int x = 0; x < gridSizeX; x++) {
        for (int y = 0; y < gridSizeY; y++) {
            for (int z = 0; z < gridSizeZ; z++) {

                if (x != 0 && x != gridSizeX - 1 && y != 0 && y != gridSizeY - 1 && z != 0 && z != gridSizeZ - 1) {
                    continue;
                }

                // std::cout << "Computing occluded voxels [faces]... " << x << " " << y << " " << z << std::endl;

                int targetx = roundup(x + 10 * voxel_size), targety = roundup(y + 10 * voxel_size), targetz = roundup(z + 10 * voxel_size);
                std::vector<std::tuple<int, int, int>> line = Bresenham3D(origin_grid[0], origin_grid[1], origin_grid[2], targetx, targety, targetz);

                for (auto point : line) {
                    int pointx, pointy, pointz;
                    std::tie(pointx, pointy, pointz) = point;
                    pointx = std::floor(pointx);
                    pointy = std::floor(pointy);
                    pointz = std::floor(pointz);

                    // std::cout << "Point: " << pointx << " " << pointy << " " << pointz << " ";

                    if (pointx < 0 || pointx >= gridSizeX || pointy < 0 || pointy >= gridSizeY || pointz < 0 || pointz >= gridSizeZ) {
                        continue;
                    }

                    if (grid[pointx][pointy][pointz]) {
                        seen3d[pointx][pointy][pointz] = true;
                        break;
                    } else {
                        seen3d[pointx][pointy][pointz] = true;
                    }

                    // std::cout << "[Done]" << std::endl;
                }
            }


        }
    }

    // std::cout << "Computing occluded voxels [interior]...";

    // Go around all the voxels inside as long as they haven't already been computed or seen
    for (int x = 0; x < gridSizeX; x++) {

        for (int y = 0; y < gridSizeY; y++) {
            for (int z = 0; z < gridSizeZ; z++) {

                if (grid[x][y][z]) {
                    seen3d[x][y][z] = true;
                }

                if (seen3d[x][y][z] || grid[x][y][z]) {
                    continue;
                }

                // std::cout << "Computing occluded voxels [interior]... " << x << " " << y << " " << z << std::endl;

                std::vector<std::tuple<int, int, int>> line = Bresenham3D(origin_grid[0], origin_grid[1], origin_grid[2], x, y, z);

                // std::cout << "Traversing line..." << std::endl;

                for (auto point : line) {
                    int pointx, pointy, pointz;
                    std::tie(pointx, pointy, pointz) = point;
                    pointx = std::round(pointx);
                    pointy = std::round(pointy);
                    pointz = std::round(pointz);

                    // std::cout << "Point: " << pointx << " " << pointy << " " << pointz << std::endl;

                    if (pointx < 0 || pointx >= gridSizeX || pointy < 0 || pointy >= gridSizeY || pointz < 0 || pointz >= gridSizeZ) {
                        continue;
                    }
                    if (grid[pointx][pointy][pointz]) {
                        seen3d[pointx][pointy][pointz] = true;
                        break;
                    } else {
                        seen3d[pointx][pointy][pointz] = true;
                    }
                }
            }


        }
    }

    return seen3d;
}


std::vector<std::vector<std::vector<int>>> bfsVoxels(
    const std::vector<std::vector<std::vector<int>>>& grid,
    const std::vector<int> viewpoint,
    const std::vector<int> min_corner,
    float voxel_size,
    int gridSizeX, 
    int gridSizeY,
    int gridSizeZ) {

    std::vector<int> origin_grid = {
        std::floor((viewpoint[0] - min_corner[0]) / voxel_size),
        std::floor((viewpoint[1] - min_corner[1]) / voxel_size),
        std::floor((viewpoint[2] - min_corner[2]) / voxel_size)
    };

    std::vector<std::vector<std::vector<int>>> seen3d(gridSizeX, std::vector<std::vector<int>>(gridSizeY, std::vector<int>(gridSizeZ, 0)));
    std::vector<std::vector<std::vector<int>>> notOccluded(gridSizeX, std::vector<std::vector<int>>(gridSizeY, std::vector<int>(gridSizeZ, 0)));

    std::queue<std::tuple<int, int, int>> q;
    q.push(std::make_tuple(origin_grid[0], origin_grid[1], origin_grid[2]));

    std::cout<< "Starting BFS ";

    while (!q.empty()) {
        int x, y, z;
        std::tie(x, y, z) = q.front();
        q.pop();

        if (x < 0 || x >= gridSizeX || y < 0 || y >= gridSizeY || z < 0 || z >= gridSizeZ) {
            continue;
        }

        if (seen3d[x][y][z] > 0) {
            continue;
        }

        seen3d[x][y][z] = 1;

        if (grid[x][y][z] > 0) {
            continue;
        }

        notOccluded[x][y][z] = 1;

        // Move in different directions - up down left right front back (not necessarily in this order)
        q.push(std::make_tuple(x + 1, y, z));
        q.push(std::make_tuple(x - 1, y, z));
        q.push(std::make_tuple(x, y + 1, z));
        q.push(std::make_tuple(x, y - 1, z));
        q.push(std::make_tuple(x, y, z + 1));
        q.push(std::make_tuple(x, y, z - 1));

        // Move in diagonals
        q.push(std::make_tuple(x + 1, y + 1, z));
        q.push(std::make_tuple(x - 1, y - 1, z));
        q.push(std::make_tuple(x + 1, y - 1, z));
        q.push(std::make_tuple(x - 1, y + 1, z));

        q.push(std::make_tuple(x + 1, y, z + 1));
        q.push(std::make_tuple(x - 1, y, z - 1));
        q.push(std::make_tuple(x + 1, y, z - 1));
        q.push(std::make_tuple(x - 1, y, z + 1));

        q.push(std::make_tuple(x, y + 1, z + 1));
        q.push(std::make_tuple(x, y - 1, z - 1));
        q.push(std::make_tuple(x, y + 1, z - 1));
        q.push(std::make_tuple(x, y - 1, z + 1));
    }

    std::cout<< "\rBFS Done" << std::endl;

    return notOccluded;
}