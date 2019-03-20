import numpy as np
from PIL import Image
from PCNN.components.row import Row, Base_row


def visualise_random(iter=10):
    base = Base_row(shape=(100, 100))
    row = Row(100, 100, plot=False, prev_row=base)
    imgs = []
    arrs = []
    gif = Image.new("1", (row.y, row.x))
    for i in range(iter):
        img, arr = gen_frame(row)
        imgs.append(img)
        arrs.append(arr)
    gif.save(
        "output/test.gif",
        save_all=True,
        append_images=imgs,
        duration=100,
    )


def visualise_input(filename, iter=10, fpms=1000):
    inp = Image.open(f"input/{filename}").convert('1')
    data = np.array([np.float64(x)/255 for x in inp.getdata()], dtype=np.float16).reshape(inp.size[0], inp.size[1])
    base = Base_row(shape=inp.size, arr=data)
    row = Row(inp.size[0], inp.size[1], plot=False, prev_row=base)
    arrs = []
    imgs = []
    gif = Image.new("1", (row.y, row.x))
    for i in range(iter):
        arr, img = gen_frame(row)
        imgs.append(img)
        arrs.append(arr)
    heatmap = np.zeros((row.y, row.x), dtype=np.int32)
    heatmap = heatmap.flatten()
    for arr in arrs:
        for idx, val in enumerate(arr.flatten()):
            heatmap[idx] += val
    heatmap = heatmap.reshape((row.y, row.x))
    print(heatmap)
    heatmap = np.interp(heatmap, [0, np.max(heatmap)], [0, 255])
    print(heatmap)
    heatmap = Image.fromarray(heatmap, "L")
    heatmap.save("output/heatmap.png")
    gif.save(
        "output/pulses.gif",
        save_all=True,
        append_images=imgs,
        duration=fpms, 
    )
    

def gen_frame(row):
    row.iterate()
    vals = np.nan_to_num(row.values)
    arr = []
    for i in range(row.y):
        arrx = []
        for j in range(row.x):
            arrx.append(vals[i, j]*255)
        arr.append(arrx)
    arr = np.array(arr, dtype=np.uint8)
    return arr, Image.fromarray(arr, 'L')
