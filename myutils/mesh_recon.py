#!/usr/bin/env python
import mitsuba as mi
import sys
# mi.set_variant('cuda_ad_rgb')
import pathlib
import itertools
import open3d as o3d
import numpy as np
import math
import logging
from skimage.io import imread
from tqdm import tqdm
import torch
# from transformers import pipeline,AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
from skimage.transform import resize
fov = np.deg2rad(35)
focal = (512/2)/math.tan(fov/2)
wh = 512
center_ = (wh-1)/2
DEFAULT_CAMERA = o3d.camera.PinholeCameraIntrinsic(
    width=wh, height=wh,
    fx=focal, fy=focal,
    cx=center_, cy=center_
)

logger = logging.getLogger(__name__)

def _pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(np.int32)
    y = np.linspace(0, height - 1, height).astype(np.int32)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


def depth_file_to_mesh(image, cameraMatrix=DEFAULT_CAMERA, minAngle=3.0, sun3d=False, depthScale=1000.0):
    """
    Converts a depth image file into a open3d TriangleMesh object

    :param image: path to the depth image file
    :param cameraMatrix: numpy array of the intrinsic camera matrix
    :param minAngle: Minimum angle between viewing rays and triangles in degrees
    :param sun3d: Specify if the depth file is in the special SUN3D format
    :returns: an open3d.geometry.TriangleMesh containing the converted mesh
    """
    # depth_raw = imread(image).astype('uint16')
    depth_raw = image
    width = depth_raw.shape[1]
    height = depth_raw.shape[0]

    if sun3d:
        depth_raw = np.bitwise_or(depth_raw>>3, depth_raw<<13)

    depth_raw = depth_raw.astype('float32')
    depth_raw /= depthScale

    logger.debug('Image dimensions:%s x %s', width, height)
    logger.debug('Camera Matrix:%s', cameraMatrix)

    if cameraMatrix is None:
        camera = DEFAULT_CAMERA
    else:
        camera = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height,
            fx=cameraMatrix[0,0], fy=cameraMatrix[1,1],
            cx=cameraMatrix[0,2], cy=cameraMatrix[1,2]
        )
    # return depth_to_mesh_close_gap(depth_raw.astype('float32'), camera, minAngle)
    return detect_boundary_points(depth_raw.astype('float32'), camera, minAngle)

def is_valid_triangle(verts, minAngle):
    v1 = verts[0] - verts[1]
    v2 = verts[0] - verts[2]
    n = np.cross(v1, v2)
    n /= np.linalg.norm(n)
    center = (verts[0] + verts[1] + verts[2]) / 3.0
    u = center / np.linalg.norm(center)
    # return abs(np.dot(n, u)) > np.cos(30./180.*math.pi)
    angle = math.degrees(math.asin(abs(np.dot(n, u))))
    return angle > minAngle
def detect_boundary_points(depth, camera=DEFAULT_CAMERA, minAngle=3.0):
    """
    Converts an open3d.geometry.Image depth image into a open3d.geometry.TriangleMesh object

    :param depth: np.array of type float32 containing the depth image
    :param camera: open3d.camera.PinholeCameraIntrinsic
    :param minAngle: Minimum angle between viewing rays and triangles in degrees
    :returns: an open3d.geometry.TriangleMesh containing the converted mesh
    """
    logger.info('Reprojecting points...')
    K = camera.intrinsic_matrix
    K_inv = np.linalg.inv(K)
    pixel_coords = _pixel_coord_np(depth.shape[1], depth.shape[0])

    w = camera.width
    h = camera.height

    refer_map_x = np.ones((w, h)).astype("int")
    refer_map_y = np.ones((w, h)).astype("int")
    refer_map_x *= -1
    refer_map_y *= -1
    cam_coords = K_inv @ pixel_coords * depth.flatten()
    boundary_points = []
    copy_coords = []
    copy_map = np.ones((w, h)).astype("int")
    copy_map *= -1
    
    with tqdm(total=(h - 1) * (w - 1),file=sys.stdout) as pbar:
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                neighbors = np.array([w * i + j, w * (i + 1) + j, w * i + (j + 1), w * (i - 1) + j, w * i + (j - 1)])

                verts = cam_coords[:, neighbors].T
                if [0, 0, 0] in map(list, verts):
                    continue

                combination = [(0,1,2),(0,2,3),(0,3,4),(0,4,1)]
                direction = np.array([[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]])
                refer_idx_x = -1
                refer_idx_y = -1
                for idx, order in enumerate(combination):
                    v1 = verts[order[0]] - verts[order[1]]
                    v2 = verts[order[0]] - verts[order[2]]

                    n = np.cross(v1, v2)
                    n /= np.linalg.norm(n)
                    center = (verts[order[0]] + verts[order[1]] + verts[order[2]]) / 3.0
                    u = center / np.linalg.norm(center)
                    angle = math.degrees(math.asin(abs(np.dot(n, u))))
                    if angle < minAngle:
                        if depth[i,j] < depth[i,j+direction[idx][1]] or depth[i,j] < depth[i+direction[idx][0],j]:
                            largest_depth = 0
                            largest_idx_x = 0
                            largest_idx_y = 0
                            boundary_points.append(verts[0])
                            if depth[i,j+direction[idx][1]] > depth[i+direction[idx][0],j]:
                                largest_depth = depth[i,j+direction[idx][1]]
                                largest_idx_x = i
                                largest_idx_y = j+direction[idx][1]
                            else:
                                largest_depth = depth[i+direction[idx][0],j]
                                largest_idx_x = i+direction[idx][0]
                                largest_idx_y = j
                            if refer_idx_x >= 0 and refer_idx_y >= 0:
                                if depth[refer_idx_x, refer_idx_y] < depth[largest_idx_x, largest_idx_y]:
                                    refer_idx_x, refer_idx_y = largest_idx_x, largest_idx_y
                            else:
                                refer_idx_x, refer_idx_y = largest_idx_x, largest_idx_y
                refer_map_x[i, j] = refer_idx_x
                refer_map_y[i, j] = refer_idx_y
                pbar.update(1)

    new_depth = depth
    count = 0
    # Modify depth
    with tqdm(total=(h - 1) * (w - 1),desc='Modify depth',file=sys.stdout) as pbar:
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                current_pix_x = i
                current_pix_y = j
                while refer_map_x[current_pix_x, current_pix_y] != -1 and refer_map_y[current_pix_x, current_pix_y] != -1:
                    temp_x = current_pix_x
                    current_pix_x = refer_map_x[current_pix_x, current_pix_y]
                    current_pix_y = refer_map_y[temp_x, current_pix_y]
                if current_pix_x != -1 and current_pix_y != -1:
                    new_depth[i, j] = depth[current_pix_x, current_pix_y]
                    count += 1
                else:
                    print("weird")
                pbar.update(1)

    new_cam_coords = K_inv @ pixel_coords * new_depth.flatten()

    indices = o3d.utility.Vector3iVector()

    # Connect
    with tqdm(total=(h - 1) * (w - 1),desc='Connect',file=sys.stdout) as pbar:
        for i in range(0, h - 1):
            for j in range(0, w - 1):
                verts = [
                    new_cam_coords[:, w * i + j],  # current pixel
                    new_cam_coords[:, w * (i + 1) + j],  # pixel below
                    new_cam_coords[:, w * i + (j + 1)],  # pixel to the right
                ]
                y_idx = [i, i+1, i]
                x_idx = [j, j ,j+1]
                verts_depth = np.array([new_depth[i, j], new_depth[i+1,j], new_depth[i, j+1]])
                if [0, 0, 0] in map(list, verts):
                    continue

                combine = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
                order = combine[0]  # 1,2,5
                if is_valid_triangle(verts, minAngle):
                    lst = [w * i + j, w * (i + 1) + j, w * i + (j + 1)]
                    # indices.append([w*i+j, w*(i+1)+j, w*i+(j+1)]) # original
                    indices.append([lst[order[0]], lst[order[1]], lst[order[2]]])
                else:
                    # find the closest point
                    lst = [w * i + j, w * (i + 1) + j, w * i + (j + 1)]
                    largest_depth = np.max(verts_depth)
                    closest_idx = np.argmin(verts_depth)
                    closest_x, closest_y = x_idx[closest_idx], y_idx[closest_idx]
                    copy_idx = copy_map[closest_x, closest_y]
                    if copy_idx != -1:
                        verts[closest_idx] = copy_coords[copy_idx]
                    else:
                        coord = np.array([[x_idx[closest_idx], y_idx[closest_idx], 1]]).T
                        coord = K_inv @ coord * largest_depth
                        coord = coord.T[0]
                        copy_map[closest_x, closest_y] = len(copy_coords)
                        copy_coords.append(coord)
                        copy_idx = copy_map[closest_x, closest_y]
                        verts[closest_idx] = coord
                    verts_depth[closest_idx] = largest_depth
                    lst[closest_idx] = new_cam_coords.shape[1] + copy_idx
                    if is_valid_triangle(verts, minAngle):
                        indices.append(lst)
                    # find the second closest point
                    else:
                        closest_idx = np.argmin(verts_depth)
                        closest_x, closest_y = x_idx[closest_idx], y_idx[closest_idx]
                        copy_idx = copy_map[closest_x, closest_y]
                        if copy_idx != -1:
                            verts[closest_idx] = copy_coords[copy_idx]
                        else:
                            coord = np.array([[closest_x, closest_y, 1]]).T
                            coord = K_inv @ coord * largest_depth
                            coord = coord.T[0]
                            copy_map[closest_x, closest_y] = len(copy_coords)
                            copy_coords.append(coord)
                            copy_idx = copy_map[closest_x, closest_y]
                            verts[closest_idx] = coord
                        lst[closest_idx] = new_cam_coords.shape[1] + copy_idx
                        if is_valid_triangle(verts, minAngle):
                            indices.append(lst)
                        # else:
                        #     print("weird")

                verts = [
                    new_cam_coords[:, w * i + (j + 1)],
                    new_cam_coords[:, w * (i + 1) + j],
                    new_cam_coords[:, w * (i + 1) + (j + 1)],
                ]
                y_idx = [i, i + 1, i + 1]
                x_idx = [j + 1, j, j + 1]
                verts_depth = np.array([new_depth[i, j+1], new_depth[i + 1, j], new_depth[i+1, j + 1]])
                if [0, 0, 0] in map(list, verts):
                    continue

                if is_valid_triangle(verts, minAngle):
                    lst = [w * i + (j + 1), w * (i + 1) + j, w * (i + 1) + (j + 1)]
                    indices.append([lst[order[0]], lst[order[1]], lst[order[2]]])
                else:
                    # find the closest point
                    lst = [w * i + (j + 1), w * (i + 1) + j, w * (i + 1) + (j + 1)]
                    largest_depth = np.max(verts_depth)
                    closest_idx = np.argmin(verts_depth)
                    closest_x, closest_y = x_idx[closest_idx], y_idx[closest_idx]
                    copy_idx = copy_map[closest_x, closest_y]
                    if copy_idx != -1:
                        verts[closest_idx] = copy_coords[copy_idx]
                    else:
                        coord = np.array([[x_idx[closest_idx], y_idx[closest_idx], 1]]).T
                        coord = K_inv @ coord * largest_depth
                        coord = coord.T[0]
                        copy_map[closest_x, closest_y] = len(copy_coords)
                        copy_coords.append(coord)
                        copy_idx = copy_map[closest_x, closest_y]
                        verts[closest_idx] = coord
                    verts_depth[closest_idx] = largest_depth
                    lst[closest_idx] = new_cam_coords.shape[1] + copy_idx
                    if is_valid_triangle(verts, minAngle):
                        indices.append(lst)
                    # find the second closest point
                    else:
                        closest_idx = np.argmin(verts_depth)
                        closest_x, closest_y = x_idx[closest_idx], y_idx[closest_idx]
                        copy_idx = copy_map[closest_x, closest_y]
                        if copy_idx != -1:
                            verts[closest_idx] = copy_coords[copy_idx]
                        else:
                            coord = np.array([[x_idx[closest_idx], y_idx[closest_idx], 1]]).T
                            coord = K_inv @ coord * largest_depth
                            coord = coord.T[0]
                            copy_map[closest_x, closest_y] = len(copy_coords)
                            copy_coords.append(coord)
                            verts[closest_idx] = coord
                            copy_idx = copy_map[closest_x, closest_y]
                        lst[closest_idx] = new_cam_coords.shape[1] + copy_idx
                        if is_valid_triangle(verts, minAngle):
                            indices.append(lst)
                        # else:
                        #     print("weird")

                pbar.update(1)

        # indices.extend(new_indices)

    # # Assign depth of copied vertices
    # copy_depth = []
    # for vertex in copy_verts:
    #     i = vertex[0]
    #     j = vertex[1]
    #     direction = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    #     largest_depth = np.max(new_depth[np.array([i, j]) + direction])
    #     copy_depth.append(largest_depth)
    #
    # copy_depth = np.array(copy_depth)
    # copy_coords = K_inv @ np.array(copy_verts).T * copy_depth.flatten()

    new_cam_coords = np.hstack((new_cam_coords, np.array(copy_coords).T))

    points = o3d.utility.Vector3dVector(new_cam_coords.transpose())
    try:
        boundary_points = o3d.utility.Vector3dVector(np.array(boundary_points))
    except:
        breakpoint()

    boundary_points = o3d.geometry.PointCloud(boundary_points)

    copy_points = o3d.utility.Vector3dVector(np.array(copy_coords))

    copy_points = o3d.geometry.PointCloud(copy_points)

    mesh = o3d.geometry.TriangleMesh(points, indices)
    return mesh, copy_points

def depth_to_mesh_close_gap(depth, camera=DEFAULT_CAMERA, minAngle=3.0):
    """
    Converts an open3d.geometry.Image depth image into a open3d.geometry.TriangleMesh object

    :param depth: np.array of type float32 containing the depth image
    :param camera: open3d.camera.PinholeCameraIntrinsic
    :param minAngle: Minimum angle between viewing rays and triangles in degrees
    :returns: an open3d.geometry.TriangleMesh containing the converted mesh
    """
    logger.info('Reprojecting points...')
    K = camera.intrinsic_matrix
    K_inv = np.linalg.inv(K)
    pixel_coords = _pixel_coord_np(depth.shape[1], depth.shape[0])
    modified_pixel_coords = pixel_coords

    w = camera.width
    h = camera.height
    cam_coords = K_inv @ pixel_coords * depth.flatten()
    boundary_points = []

    # Detect boundary pixels
    with tqdm(total=(h - 1) * (w - 1)) as pbar:
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                neighbors = np.array([w*i+j, w * (i + 1) + j,w * i + (j + 1),w * (i - 1) + j,w * i + (j - 1)])

                verts = cam_coords[:, neighbors].T
                if [0, 0, 0] in map(list, verts):
                    continue

                combination = [(0,1,2),(0,2,3),(0,3,4),(0,4,1)]
                direction = np.array([[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]])
                for idx, order in enumerate(combination):
                    v1 = verts[order[0]] - verts[order[1]]
                    v2 = verts[order[0]] - verts[order[2]]
                    n = np.cross(v1, v2)
                    n /= np.linalg.norm(n)
                    center = (verts[order[0]] + verts[order[1]] + verts[order[2]]) / 3.0
                    u = center / np.linalg.norm(center)
                    angle = math.degrees(math.asin(abs(np.dot(n, u))))
                    if angle < minAngle:
                        if depth[i,j] > depth[i,j+direction[idx][1]] or depth[i,j]>depth[i+direction[idx][0],j]:
                            modified_pixel_coords[:, w*i+j]+=direction[idx]
                            boundary_points.append(cam_coords[:, w*i+j])
                pbar.update(1)


    modified_cam_coords = K_inv @ modified_pixel_coords * depth.flatten()

    print("Hack done..")

    indices = o3d.utility.Vector3iVector()

    with tqdm(total=(h - 1) * (w - 1)) as pbar:
        for i in range(0, h - 1):
            for j in range(0, w - 1):
                verts = [
                    cam_coords[:, w * i + j],  # current pixel
                    cam_coords[:, w * (i + 1) + j],  # pixel below
                    cam_coords[:, w * i + (j + 1)],  # pixel to the right
                    # cam_coords[:, w * (i - 1) + j],  # pixel above
                    # cam_coords[:, w * i + (j - 1)]  # pixel to the left
                ]
                if [0, 0, 0] in map(list, verts):
                    continue

                # if abs(depth[i, j] - depth[i+1, j]) > depth_threshold or \
                # abs(depth[i, j] - depth[i, j+1]) > depth_threshold:
                #     verts = [v+np.array([0.5,0.5,0]) for v in verts]
                #     # breakpoint()

                v1 = verts[0] - verts[1]
                v2 = verts[0] - verts[2]
                n = np.cross(v1, v2)
                n /= np.linalg.norm(n)
                center = (verts[0] + verts[1] + verts[2]) / 3.0
                u = center / np.linalg.norm(center)
                angle = math.degrees(math.asin(abs(np.dot(n, u))))
                combine = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
                order = combine[0]  # 1,2,5
                if angle > minAngle:
                    lst = [w * i + j, w * (i + 1) + j, w * i + (j + 1)]
                    # indices.append([w*i+j, w*(i+1)+j, w*i+(j+1)]) # original
                    indices.append([lst[order[0]], lst[order[1]], lst[order[2]]])
                # else:
                # verts = [
                #     cam_coords[:, w*i+j],
                #     cam_coords[:, w*(i+1)+j],
                #     cam_coords[:, w*i+(j+1)],
                # ]

                #
                # new_vert = verts[2].copy()
                # new_vert[2] = np.max([verts[0][2], verts[1][2]])

                #
                # new_points.append(new_vert)

                #
                # new_index = cam_coords.shape[1] + len(new_points) - 1  # index of new vertex
                # lst = [w*i+j, w*(i+1)+j, new_index]
                # new_indices.append([lst[order[0]], lst[order[1]], lst[order[2]]])

                verts = [
                    cam_coords[:, w * i + (j + 1)],
                    cam_coords[:, w * (i + 1) + j],
                    cam_coords[:, w * (i + 1) + (j + 1)],
                ]
                if [0, 0, 0] in map(list, verts):
                    continue
                v1 = verts[0] - verts[1]
                v2 = verts[0] - verts[2]
                n = np.cross(v1, v2)
                n /= np.linalg.norm(n)
                center = (verts[0] + verts[1] + verts[2]) / 3.0
                u = center / np.linalg.norm(center)
                angle = math.degrees(math.asin(abs(np.dot(n, u))))
                if angle > minAngle:
                    lst = [w * i + (j + 1), w * (i + 1) + j, w * (i + 1) + (j + 1)]
                    # indices.append([w*i+(j+1),w*(i+1)+j, w*(i+1)+(j+1)]) # original
                    indices.append([lst[order[0]], lst[order[1]], lst[order[2]]])

                pbar.update(1)
        # cam_coords = np.hstack((cam_coords, np.array(new_points).T))
        # indices.extend(new_indices)

    points = o3d.utility.Vector3dVector(modified_cam_coords.transpose())

    boundary_points = o3d.utility.Vector3dVector(np.array(boundary_points))

    boundary_points = o3d.geometry.PointCloud(boundary_points)

    mesh = o3d.geometry.TriangleMesh(points, indices)

    # mesh.compute_vertex_normals()
    # mesh.compute_triangle_normals()

    return mesh, boundary_points

def depth_to_mesh(depth, camera=DEFAULT_CAMERA, minAngle=3.0):
    """
    Converts an open3d.geometry.Image depth image into a open3d.geometry.TriangleMesh object

    :param depth: np.array of type float32 containing the depth image
    :param camera: open3d.camera.PinholeCameraIntrinsic
    :param minAngle: Minimum angle between viewing rays and triangles in degrees
    :returns: an open3d.geometry.TriangleMesh containing the converted mesh
    """

    logger.info('Reprojecting points...')
    K = camera.intrinsic_matrix
    K_inv = np.linalg.inv(K)
    pixel_coords = _pixel_coord_np(depth.shape[1], depth.shape[0])

    cam_coords = K_inv @ pixel_coords * depth.flatten()
    cam_coords_out = cam_coords

    indices = o3d.utility.Vector3iVector()
    w = camera.width
    h = camera.height
    new_points = []  
    new_indices = []  
    with tqdm(total=(h-1)*(w-1)) as pbar:
        for i in range(1, h-1):
            for j in range(1, w-1):
                verts = [
                    cam_coords[:, w*i+j], # current pixel
                    cam_coords[:, w*(i+1)+j], # pixel below
                    cam_coords[:, w*i+(j+1)], # pixel to the right
                ]
                if [0,0,0] in map(list, verts):
                    continue

                
                # if abs(depth[i, j] - depth[i+1, j]) > depth_threshold or \
                # abs(depth[i, j] - depth[i, j+1]) > depth_threshold:
                #     verts = [v+np.array([0.5,0.5,0]) for v in verts]
                #     # breakpoint()
                
                v1 = verts[0] - verts[1]
                v2 = verts[0] - verts[2]
                n = np.cross(v1, v2)
                n /= np.linalg.norm(n)
                center = (verts[0] + verts[1] + verts[2]) / 3.0
                u = center / np.linalg.norm(center)
                angle = math.degrees(math.asin(abs(np.dot(n, u))))
                combine = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
                order = combine[0] #1,2,5
                if angle > minAngle:
                    lst = [w*i+j, w*(i+1)+j, w*i+(j+1)]
                    # indices.append([w*i+j, w*(i+1)+j, w*i+(j+1)]) # original
                    indices.append([lst[order[0]], lst[order[1]], lst[order[2]]])
                else:
                    verts = [
                        cam_coords[:, w*i+j],
                        cam_coords[:, w*(i+1)+j],
                        cam_coords[:, w*i+(j+1)],
                    ]
                    

                    new_vert = verts[2].copy()
                    new_vert[2] = np.max([verts[0][2], verts[1][2]])
                    

                    new_points.append(new_vert)
                    

                    new_index = cam_coords.shape[1] + len(new_points) - 1  # index of new vertex
                    lst = [w*i+j, w*(i+1)+j, new_index]
                    new_indices.append([lst[order[0]], lst[order[1]], lst[order[2]]])
                    

                verts = [
                    cam_coords[:, w*i+(j+1)],
                    cam_coords[:, w*(i+1)+j],
                    cam_coords[:, w*(i+1)+(j+1)],
                ]
                if [0,0,0] in map(list, verts):
                    continue
                v1 = verts[0] - verts[1]
                v2 = verts[0] - verts[2]
                n = np.cross(v1, v2)
                n /= np.linalg.norm(n)
                center = (verts[0] + verts[1] + verts[2]) / 3.0
                u = center / np.linalg.norm(center)
                angle = math.degrees(math.asin(abs(np.dot(n, u))))
                if angle > minAngle:
                    lst = [w*i+(j+1),w*(i+1)+j, w*(i+1)+(j+1)]
                    # indices.append([w*i+(j+1),w*(i+1)+j, w*(i+1)+(j+1)]) # original
                    indices.append([lst[order[0]], lst[order[1]], lst[order[2]]])


                pbar.update(1)
    # cam_coords = np.hstack((cam_coords, np.array(new_points).T))
    # indices.extend(new_indices)
    
    points = o3d.utility.Vector3dVector(cam_coords.transpose())

    mesh = o3d.geometry.TriangleMesh(points, indices)
    
    # mesh.compute_vertex_normals()
    # mesh.compute_triangle_normals()
    
    return mesh

def depth_to_mesh_with_normals(depth, normals, camera=DEFAULT_CAMERA, minAngle=3.0):
    """
    Converts a depth map and normal map into a open3d.geometry.TriangleMesh object.

    :param depth: np.array of type float32 containing the depth image of shape (H, W)
    :param normals: np.array of type float32 containing the normal map of shape (H, W, 3)
    :param camera: open3d.camera.PinholeCameraIntrinsic
    :param minAngle: Minimum angle between viewing rays and triangles in degrees
    :returns: an open3d.geometry.TriangleMesh containing the converted mesh
    """
    
    logger.info('Reprojecting points...')
    K = camera.intrinsic_matrix
    K_inv = np.linalg.inv(K)
    pixel_coords = _pixel_coord_np(depth.shape[1], depth.shape[0])
    cam_coords = K_inv @ pixel_coords * depth.flatten()

    indices = o3d.utility.Vector3iVector()
    w = camera.width
    h = camera.height

    with tqdm(total=(h-1)*(w-1)) as pbar:
        for i in range(0, h-1):
            for j in range(0, w-1):
                verts = [
                    cam_coords[:, w*i+j],
                    cam_coords[:, w*(i+1)+j],
                    cam_coords[:, w*i+(j+1)],
                ]
                if [0,0,0] in map(list, verts):
                    continue

                # Using normals from a normal map
                normal_verts = [
                    normals[i, j],
                    normals[i+1, j],
                    normals[i, j+1],
                ]
                
                n = np.mean(normal_verts, axis=0)  # average normal
                n /= np.linalg.norm(n)
                center = (verts[0] + verts[1] + verts[2]) / 3.0
                u = center / np.linalg.norm(center)
                angle = math.degrees(math.asin(abs(np.dot(n, u))))
                if angle > minAngle:
                    # indices.append([w*i+j, w*(i+1)+j, w*i+(j+1)])
                    indices.append([w*i+j, w*i+(j+1), w*(i+1)+j]) # change the vertex order

                verts = [
                    cam_coords[:, w*i+(j+1)],
                    cam_coords[:, w*(i+1)+j],
                    cam_coords[:, w*(i+1)+(j+1)],
                ]
                if [0,0,0] in map(list, verts):
                    continue

                normal_verts = [
                    normals[i, j+1],
                    normals[i+1, j],
                    normals[i+1, j+1],
                ]
                
                n = np.mean(normal_verts, axis=0)  # average normal
                n /= np.linalg.norm(n)
                center = (verts[0] + verts[1] + verts[2]) / 3.0
                u = center / np.linalg.norm(center)
                angle = math.degrees(math.asin(abs(np.dot(n, u))))
                if angle > minAngle:
                    # indices.append([w*i+(j+1),w*(i+1)+j, w*(i+1)+(j+1)])
                    indices.append([w*i+(j+1),w*(i+1)+(j+1),w*(i+1)+j,]) # change the vertex order
                
                pbar.update(1)
    
    points = o3d.utility.Vector3dVector(cam_coords.transpose())

    mesh = o3d.geometry.TriangleMesh(points, indices)
    
    # Set vertex normal
    vertex_normals = np.mean(normals, axis=2)
    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
    
    mesh.compute_triangle_normals()
    
    return mesh

import copy

def rotate_mesh_around_x(mesh, angle_in_degrees):
    """
    Rotates the mesh around the x-axis by the specified angle.
    
    :param mesh: open3d.geometry.TriangleMesh
    :param angle_in_degrees: The rotation angle in degrees.
    :return: Rotated mesh.
    """
    # Convert degrees to radians
    angle_in_radians = np.radians(angle_in_degrees)
    
    # Define the rotation matrix for x-axis rotation
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                                [0, np.sin(angle_in_radians), np.cos(angle_in_radians)]])
    
    # Apply the rotation matrix to the mesh
    mesh.rotate(rotation_matrix, center=(0, 0, 0))
    
    return mesh


def rotate_pc_around_x(pc, angle_in_degrees):
    """
    Rotates the pc around the x-axis by the specified angle.

    :param mesh: open3d.geometry.PointCloud
    :param angle_in_degrees: The rotation angle in degrees.
    :return: Rotated mesh.
    """
    # Convert degrees to radians
    angle_in_radians = np.radians(angle_in_degrees)

    # Define the rotation matrix for x-axis rotation
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                                [0, np.sin(angle_in_radians), np.cos(angle_in_radians)]])

    # Apply the rotation matrix to the mesh
    pc.rotate(rotation_matrix, center=(0, 0, 0))

    return pc


from scipy.spatial import Delaunay,ConvexHull

def reverse_triangle_order(mesh):
    """
    Reverse the vertex order of all triangles in the mesh.

     :param mesh: open3d.geometry.TriangleMesh object
     :return: TriangleMesh object with reversed vertex order
    """
    triangles = np.asarray(mesh.triangles)

    triangles[:, [1, 2]] = triangles[:, [2, 1]]

    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Recalculate normals
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    return mesh

# def estimate_depth(image_path):
#     image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
#     model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
#     img = Image.open(image_path)
#     img = img.resize((518,518))
#     inputs = image_processor(images=img, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
#         predicted_depth = outputs.predicted_depth
#
#     depth_pred = predicted_depth.squeeze().cpu().numpy()
#     depth_pred = resize(depth_pred,(512,512))
#     return depth_pred
#

def main():
    # depth = mi.Bitmap('./teapot.exr')
    depth = mi.Bitmap('./room_depth.exr')
    # depth_gt = mi.Bitmap('/home/lewa/inverse_rendering/my_inverse_sss/env_render/depth.exr')
    # depth_gt = np.array(depth_gt)
    depth_np = np.array(depth)
    # normal = mi.Bitmap('/home/lewa/inverse_rendering/my_inverse_sss/env_render/geonormal.exr')
    # normal = np.array(normal)
    # breakpoint()
    # flip horizontally and vertically to depth_np
    # depth_np = np.flip(depth_np, axis=0)
    # depth_np= np.flip(depth_np, axis=1) # flip vertically
    print(depth_np.max())
    depth_np = 2 * depth_np.max() - depth_np

    mesh, b_points = depth_file_to_mesh(depth_np, cameraMatrix=None, minAngle=1.5, sun3d=False, depthScale=1.0)
    # mesh = depth_to_mesh_with_normals(depth_np, normal, camera=DEFAULT_CAMERA, minAngle=3.0)
    mesh = rotate_mesh_around_x(mesh, 180)
    b_points = rotate_pc_around_x(b_points, 180)
    
    # o3d.io.write_triangle_mesh("./output_imgs/teapot_expand.ply", mesh)
    o3d.io.write_triangle_mesh("./output_imgs/room.ply", mesh)
    # o3d.io.write_point_cloud("./output_imgs/teapot_b_points.ply", b_points)
    breakpoint()
    # filled_mesh = fill_holes(mesh)
    # o3d.io.write_triangle_mesh("output_imgs/filled_teapot.ply", filled_mesh)
    # breakpoint()

    return
    
if __name__ == '__main__':
    mi.set_variant('cuda_ad_rgb')
    main()