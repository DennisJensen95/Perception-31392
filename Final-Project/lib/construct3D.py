import cv2
import numpy as np

def downsample_image(img, ratio=0.2):
    small_img = cv2.resize(img,  # original image
                           (0, 0),  # set fx and fy, not the final size
                           fx=ratio,
                           fy=ratio,
                           interpolation=cv2.INTER_NEAREST)

    return small_img

def return_pointcloud(disparity_map, Q):
    h, w = disparity_map.shape[:2]
    f = .8 * w  # guess for focal length. If you 3D reconstruction looks skewed in the viewing direction, try adjusting this parameter.
    # Q = np.float32([[1, 0, 0, -0.5 * w],
    #                 [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
    #                 [0, 0, 0, -f],  # so that y-axis looks up
    #                 [0, 0, 1, 0]])
    points = cv2.reprojectImageTo3D(disparity_map, Q)
    return points

def export_pointcloud(disparity_map, colors, Q):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''

    def write_ply(fn, verts, colors):
        verts = verts.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        verts = np.hstack([verts, colors])
        with open(fn, 'wb') as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

    h, w = disparity_map.shape[:2]
    f = .8 * w  # guess for focal length. If you 3D reconstruction looks skewed in the viewing direction, try adjusting this parameter.
    Q_test = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                    [0, 0, 0, -f],  # so that y-axis looks up
                    [0, 0, 1, 0]])

    points = cv2.reprojectImageTo3D(disparity_map, Q)

    mask = disparity_map > disparity_map.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print(f'{out_fn} saved')

def construct_z_coordinate(disparity_map, baseline, focal_length, last_depth):
    """
    Will calculate the depth
    :param disparity_map:
    :param baseline:
    :param focal_length:
    :return: depth in mm
    """
    try:
        depth_map = baseline * focal_length / disparity_map
    except ZeroDivisionError as e:
        print(e)
        return last_depth

    return depth_map