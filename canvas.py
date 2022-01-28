import numpy as np
import matplotlib.pyplot as plt
import random, torch

class Canvas:
    showing = False
    def __init__(self, dimension):
        self.dimension = dimension
        # self.canvas = np.random.random(self.dimension)
        self.canvas = np.zeros(self.dimension)
        self.figure = plt.figure(1)
        self.showing = False

    def init_figure(self, number=1):
        self.figure = plt.figure(number)

    def set_canvas(self, image):
        assert(image.shape == self.canvas.shape)
        self.canvas = np.copy(image)

    def show_canvas(self):
        plt.imshow(self.canvas, cmap='seismic', interpolation='nearest', vmin=-0.3, vmax=0.3, origin='upper', extent=[0,1,1,0])

        if not self.showing:
            # plt.colorbar()
            self.figure.show()
            self.showing = True
            plt.pause(0.001)
        else:
            plt.draw()
            plt.pause(0.001)

    def clear(self):
        plt.cla()
        plt.draw()

    @staticmethod
    def show_image(canvas):
        plt.clf()
        plt.imshow(canvas, cmap='seismic', interpolation='nearest', vmin=-0.3, vmax=0.3, origin='upper',
                   extent=[0, 1, 1, 0])
        if not Canvas.showing:
            plt.colorbar()
            plt.figure(1).show()
            Canvas.showing = True
            plt.draw()
            plt.pause(0.001)
        else:
            plt.draw()
            plt.pause(0.001)

    @staticmethod
    def save_image(canvas, fname, cmap_range=0.3):
        plt.imsave(fname, canvas, cmap='seismic', vmin=-cmap_range, vmax=cmap_range, origin='upper')
        # Canvas.show_image(canvas)
        # plt.savefig(fname)

    @staticmethod
    def save_discrete_cmap(data, fname):
        plt.close()
        cmap = plt.get_cmap('RdBu', np.max(data) - np.min(data) + 1)
        # set limits .5 outside true range
        mat = plt.matshow(data,cmap=cmap,vmin = np.min(data)-.5, vmax = np.max(data)+.5)
        plt.axis('off')
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close()

    @staticmethod
    def show_points(x, y, color='r'):
        plt.scatter(x, y, color=color, s=1)
        plt.draw()
        plt.pause(0.001)

    @staticmethod
    def show_points_color(x, y, c='r'):
        plt.scatter(x, y, c=c, s=5)
        plt.draw()
        plt.pause(0.001)

    @staticmethod
    def show_points_cmap(x, y, c):
        plt.scatter(x, y, c=c, cmap='seismic', s=1, vmin=-0.3, vmax=0.3)
        plt.draw()

    @staticmethod
    def assign_color_hardmax(sm):
        num_color = sm.max()+1
        colors = torch.tensor(Canvas.rand_colors(int(num_color)))
        return colors[sm]

    @staticmethod
    def rand_colors(n):
        ret = []
        r = 0
        g = 256
        b = 128
        step = 256 / n
        for i in range(n):
            r += step
            g -= step
            b += step
            r = int(r) % 256
            g = int(g) % 256
            b = int(b) % 256
            ret.append((r / 255., g / 255., b / 255.))
        return ret

