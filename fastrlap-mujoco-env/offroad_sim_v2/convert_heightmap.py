from PIL import Image
import struct
import numpy as np
import scipy

def convert(in_path, out_path, img_size):
    img = Image.open(f'{in_path}').resize(img_size)
    img = np.asarray(img) / 255.
    img = np.flipud(img) + np.random.uniform(size=img_size) * 0.05
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    with open(out_path, 'wb') as f:
        f.write(struct.pack('2I', *img_size))
        f.write(struct.pack(f'{img_size[0]*img_size[1]}f', *img.flatten()))

if __name__ == '__main__':
    convert('envs/textures/heightmap.tif', 'envs/textures/heightmap.bin', (256, 256))
