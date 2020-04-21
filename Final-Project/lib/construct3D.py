import cv2

def downsample_image(img, ratio=0.2):
    small_img = cv2.resize(img,  # original image
                           (0, 0),  # set fx and fy, not the final size
                           fx=ratio,
                           fy=ratio,
                           interpolation=cv2.INTER_NEAREST)

    return small_img