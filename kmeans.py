import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation, PillowWriter


class KMeans:
    def __init__(self, image, k):
        self.image = image
        self.pixel = image.reshape((-1, 3))
        self.n_examples = self.pixel.shape[0]
        self.centroid = self.pixel[np.random.choice(self.n_examples, size=k)].astype(np.float)
        self.k = k

    def assign(self):
        dist = np.linalg.norm(self.pixel[:, np.newaxis, :] - self.centroid[np.newaxis, :, :], axis=2)
        self.assignment = np.argmin(dist, axis=1)

    def update(self):
        for i in range(self.k):
            pixel = self.pixel[self.assignment == i]
            if pixel.shape[0] == 0:
                self.centroid[i] = np.zeros(3)
            else:
                self.centroid[i] = np.mean(pixel, axis=0)

    def visualize(self, im):
        img = self.centroid[self.assignment].reshape(self.image.shape).astype(np.int16)
        im.set_data(img)

    def error(self):
        img = self.centroid[self.assignment].reshape(self.image.shape).astype(np.int16)
        return np.absolute(self.image - img).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=50,
                        help="The number of centroid (default to 50)")
    parser.add_argument("--input", type=str, default=os.path.join("res", "input.jpg"),
                        help=("The filepath for input image (default"
                              " to os.path.join(\"res\", \"input.jpg\"))"))
    args = parser.parse_args()

    img = mpimg.imread(args.input)
    kmeans = KMeans(img, args.k)

    fig, ax = plt.subplots()
    im = plt.imshow(img)

    def animate(i):
        kmeans.assign()
        kmeans.update()
        ax.set_title("K: %d, Iteration: %d, Reconstruction error: %.4f" % (args.k, i, kmeans.error()))
        kmeans.visualize(im)
        plt.savefig(os.path.join('res', 'frame_%d_%d.png' % (args.k, i)))
        return im

    os.makedirs("res", exist_ok=True)
    ani = FuncAnimation(fig, animate)
    writer = PillowWriter(fps=5)
    ani.save(os.path.join("res", "result_%d.gif" % args.k), writer=writer)