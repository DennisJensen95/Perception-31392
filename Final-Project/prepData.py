from lib.prepareData import *
from sklearn.preprocessing import LabelEncoder
from tempfile import TemporaryFile
from lib.fastRCNNPretrained import *

le = LabelEncoder()
le.fit(['box', 'book', 'cup'])

data = le.transform(['box', 'box', 'box', 'cup', 'book', 'book', 'book'])

debug = False

images = getImages('data/Stereo_conveyor_without_occlusions/left/')
if debug:
    """"""
    # watchFrames(images, 1)

box_1 = [100, 250, 'box']
box_2 = [330, 450, 'box']
book_1 = [500, 630, 'book']
book_2 = [660, 830, 'book']
book_3 = [870, 1030, 'book']
cup_1 = [1070, 1220, 'cup']
cup_2 = [1260, 1460, 'cup']

intervals = [box_1, box_2, book_1, book_2, book_3, cup_1, cup_2]
y = np.empty(0)
x = np.empty(0)
for i in range(len(intervals)):
    object_images = np.asarray(images[intervals[i][0]:intervals[i][1]])
    frames = intervals[i][1] - intervals[i][0]
    y_objects = np.full(frames, intervals[i][2])
    y = np.append(y, y_objects)
    x = np.append(x, object_images)

data = [x, y]
img = cv2.imread(data[0][0])
shape = img.shape
scale = 10
width, height = int(shape[1]/scale), int(shape[0]/scale)
print(f'width: {width}, height: {height}')
resize = True
img_resized = cv2.resize(img, (width, height))
# print(img_resized.shape)
if debug:
    if resize:
        watchFrames(images, 10, (width, height), object=False, rezise=True)
    else:
        watchFrames(data, 10, object=True, rezise=False)

X, Y = preproccessData(data, (width, height), ['box', 'book', 'cup'])

print(X)
print(Y)

out_X = './data/imageX.npy'
out_Y = './data/imageY.npy'

np.save(out_X, X)
np.save(out_Y, Y)

