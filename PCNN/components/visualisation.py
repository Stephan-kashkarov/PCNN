import numpy as np
from PIL import Image
from PCNN.components.row import Row, Base_row


def visualise_random(iter=10):
    base = Base_row(shape=(100, 100))
    row = Row(100, 100, plot=False, prev_row=base)
    imgs = []
    gif = Image.new("1", (row.y, row.x))
    for i in range(iter):
        imgs.append(gen_frame(row))
    gif.save(
        "output/test.gif",
        save_all=True,
        append_images=imgs,
        duration=100,
    )


def visualise_input(filename, iter=10):
    inp = Image.open(f"input/{filename}").convert('1')
    size = list(inp.size)
    size[0] -= 1
    size[1] -= 1
    base = Base_row(shape=size, arr=inp.getdata())
    row = Row(inp.size[0], inp.size[1], plot=False, prev_row=base)
    imgs = []
    gif = Image.new("1", (row.y, row.x))
    for i in range(iter):
        imgs.append(gen_frame(row))
    gif.save(
        "output/test.gif",
        save_all=True,
        append_images=imgs,
        duration=1000,
    )
    

def gen_frame(row):
    row.iterate()
    vals = row.values
    arr = []
    for i in range(row.y):
        arrx = []
        for j in range(row.x):
            arrx.append(vals[i, j]*255)
        arr.append(arrx)
    return Image.fromarray(np.array(arr, dtype=np.uint8), 'L')
