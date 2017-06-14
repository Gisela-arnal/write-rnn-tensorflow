import os
import pickle
import xml.etree.ElementTree as ET
from itertools import chain, repeat

import h5py
import numpy as np
from tfglib.seq2seq_normalize import maxmin_scaling


# import svgwrite
# from IPython.display import SVG, display

def get_bounds(data, factor):
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)


# old version, where each path is entire stroke (smaller svg size, but have to keep same color)
def draw_strokes(data, factor=10, svg_filename='sample.svg'):
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)

    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

    lift_pen = 1

    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)

    command = "m"

    for i in range(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        lift_pen = data[i, 2]
        p += command + str(x) + "," + str(y) + " "

    the_color = "black"
    stroke_width = 1

    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))

    dwg.save()
    display(SVG(dwg.tostring()))


def draw_strokes_eos_weighted(stroke, param, factor=10, svg_filename='sample_eos.svg'):
    c_data_eos = np.zeros((len(stroke), 3))
    for i in range(len(param)):
        c_data_eos[i, :] = (1 - param[i][6][0]) * 225  # make color gray scale, darker = more likely to eos
    draw_strokes_custom_color(stroke, factor=factor, svg_filename=svg_filename, color_data=c_data_eos, stroke_width=3)


def draw_strokes_random_color(stroke, factor=10, svg_filename='sample_random_color.svg', per_stroke_mode=True):
    c_data = np.array(np.random.rand(len(stroke), 3) * 240, dtype=np.uint8)
    if per_stroke_mode:
        switch_color = False
        for i in range(len(stroke)):
            if switch_color == False and i > 0:
                c_data[i] = c_data[i - 1]
            if stroke[i, 2] < 1:  # same strike
                switch_color = False
            else:
                switch_color = True
    draw_strokes_custom_color(stroke, factor=factor, svg_filename=svg_filename, color_data=c_data, stroke_width=2)


def draw_strokes_custom_color(data, factor=10, svg_filename='test.svg', color_data=None, stroke_width=1):
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)

    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y

    for i in range(len(data)):

        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor

        prev_x = abs_x
        prev_y = abs_y

        abs_x += x
        abs_y += y

        if (lift_pen == 1):
            p = "M " + str(abs_x) + "," + str(abs_y) + " "
        else:
            p = "M +" + str(prev_x) + "," + str(prev_y) + " L " + str(abs_x) + "," + str(abs_y) + " "

        lift_pen = data[i, 2]

        the_color = "black"

        if (color_data is not None):
            the_color = "rgb(" + str(int(color_data[i, 0])) + "," + str(int(color_data[i, 1])) + "," + str(
                int(color_data[i, 2])) + ")"

        dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill(the_color))
    dwg.save()
    display(SVG(dwg.tostring()))


def draw_strokes_pdf(data, param, factor=10, svg_filename='sample_pdf.svg'):
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)

    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

    abs_x = 25 - min_x
    abs_y = 25 - min_y

    num_mixture = len(param[0][0])

    for i in range(len(data)):

        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor

        for k in range(num_mixture):
            pi = param[i][0][k]
            if pi > 0.01:  # optimisation, ignore pi's less than 1% chance
                mu1 = param[i][1][k]
                mu2 = param[i][2][k]
                s1 = param[i][3][k]
                s2 = param[i][4][k]
                sigma = np.sqrt(s1 * s2)
                dwg.add(
                    dwg.circle(center=(abs_x + mu1 * factor, abs_y + mu2 * factor), r=int(sigma * factor)).fill('red',
                                                                                                                opacity=pi / (
                                                                                                                    sigma * sigma * factor)))

        prev_x = abs_x
        prev_y = abs_y

        abs_x += x
        abs_y += y

    dwg.save()
    display(SVG(dwg.tostring()))


class DataLoader():
    def __init__(self, dataset_path, files_list, dataset_file, batch_size=50, seq_length=3 * 30, scale_factor=10,
                 limit=500, save_h5=False):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.scale_factor = scale_factor  # divide data by this factor
        self.limit = limit  # removes large noisy gaps in the data
        self.num_batches = None

        assert os.path.exists(os.path.join(self.dataset_path, files_list))
        # print("creating training data pkl file from raw source")
        file = open(os.path.join(self.dataset_path, files_list), 'r')
        file = file.readlines()

        self.files_list = []
        for name in file:
            self.files_list.append(name.split('\n')[0])

        self.dataset_file = dataset_file

        # raw_data_dir = self.data_dir+"/lineStrokes"

        # self.preprocess(raw_data_dir, data_file)

        self.valid_data = None

        self.id_num = 25
        self.col_filter = (2, 3, 4)
        self.parameters_length = self.id_num * len(self.col_filter)

        self.max_mat = -1e+20 * np.ones(self.parameters_length)
        self.min_mat = 1e+20 * np.ones(self.parameters_length)

        self.min_mat, self.max_mat = self.load_preprocessed(save_h5)
        # self.max_mat = maxi
        # self.min_mat = mini

        # self.reset_batch_pointer()

    def preprocess(self, data_dir, data_file):
        # create data file from raw xml files from iam handwriting source.

        # build the list of xml files
        filelist = []
        # Set the directory you want to start from
        rootDir = data_dir
        for dirName, subdirList, fileList in os.walk(rootDir):
            # print('Found directory: %s' % dirName)
            for fname in fileList:
                # print('\t%s' % fname)
                filelist.append(dirName + "/" + fname)

        # function to read each individual xml file
        def getStrokes(filename):
            tree = ET.parse(filename)
            root = tree.getroot()

            result = []

            x_offset = 1e20
            y_offset = 1e20
            y_height = 0
            for i in range(1, 4):
                x_offset = min(x_offset, float(root[0][i].attrib['x']))
                y_offset = min(y_offset, float(root[0][i].attrib['y']))
                y_height = max(y_height, float(root[0][i].attrib['y']))
            y_height -= y_offset
            x_offset -= 100
            y_offset -= 100

            for stroke in root[1].findall('Stroke'):
                points = []
                for point in stroke.findall('Point'):
                    points.append([float(point.attrib['x']) - x_offset, float(point.attrib['y']) - y_offset])
                result.append(points)

            return result

        # converts a list of arrays into a 2d numpy int16 array
        def convert_stroke_to_array(stroke):

            n_point = 0
            for i in range(len(stroke)):
                n_point += len(stroke[i])
            stroke_data = np.zeros((n_point, 3), dtype=np.int16)

            prev_x = 0
            prev_y = 0
            counter = 0

            for j in range(len(stroke)):
                for k in range(len(stroke[j])):
                    stroke_data[counter, 0] = int(stroke[j][k][0]) - prev_x
                    stroke_data[counter, 1] = int(stroke[j][k][1]) - prev_y
                    prev_x = int(stroke[j][k][0])
                    prev_y = int(stroke[j][k][1])
                    stroke_data[counter, 2] = 0
                    if (k == (len(stroke[j]) - 1)):  # end of stroke
                        stroke_data[counter, 2] = 1
                    counter += 1
            return stroke_data

        # build stroke database of every xml file inside iam database
        strokes = []
        for i in range(len(filelist)):
            if (filelist[i][-3:] == 'xml'):
                print('processing ' + filelist[i])
                strokes.append(convert_stroke_to_array(getStrokes(filelist[i])))

        f = open(data_file, "wb")
        pickle.dump(strokes, f, protocol=2)
        f.close()

    def load_file(self, data_file):
        print(os.path.join(self.dataset_path, data_file))
        data = np.loadtxt(os.path.join(self.dataset_path, data_file), delimiter=',', dtype=np.float32)

        aux = np.array([slice.reshape(self.parameters_length) for slice in
                        np.split(data[:, self.col_filter], data.shape[0] / self.id_num)])
        a_size = aux.shape[0]
        a_modulus = a_size % self.seq_length

        data_aux = aux[:a_size - a_modulus].reshape(-1, self.seq_length, self.parameters_length)

        # Calculamos los valores maximos y minimos
        self.min_mat = np.minimum(np.min(data_aux, axis=(0, 1)), self.min_mat)
        self.max_mat = np.maximum(np.max(data_aux, axis=(0, 1)), self.max_mat)

        mini = self.min_mat
        maxi = self.max_mat

        return data_aux, mini, maxi

    def load_preprocessed(self, save_h5=False):
        # self.valid_data = []
        if save_h5:

            d = np.empty((0, self.seq_length, self.parameters_length))

            for file_name in self.files_list:
                print(file_name)
                fil, mini, maxi = self.load_file(file_name)
                d = np.concatenate((d, fil))

            self.data = d

            for i, sequence in enumerate(self.data):
                self.data[i] = maxmin_scaling(sequence, self.max_mat, self.min_mat)

            with h5py.File(self.dataset_file, 'w') as file:
                file.create_dataset('dataset', data=self.data, compression='gzip', compression_opts=9)

        else:
            with h5py.File(self.dataset_file, 'r') as file:
                self.data = file['dataset'][:]
            maxi = -1e+20 * np.ones(self.parameters_length)
            mini = 1e+20 * np.ones(self.parameters_length)

        valid_index = int(np.floor(self.data.shape[0] * 0.05))

        if valid_index < self.batch_size:
            valid_index = self.batch_size + \
                          1

        self.valid_data = self.data[-valid_index:]
        self.data = self.data[:-valid_index]
        self.num_batches = int(np.floor(self.data.shape[0] / self.batch_size))

        return mini, maxi

    def validation_data(self):
        # returns validation data
        # x_batch = []
        # y_batch = []
        # for i in range(self.batch_size):
        #     data = self.valid_data[i % len(self.valid_data)]
        #     idx = 0
        #     x_batch.append(np.copy(data[idx:idx + self.seq_length]))
        #     y_batch.append(np.copy(data[idx + 1:idx + self.seq_length + 1]))
        # return x_batch, y_batch
        return self.valid_data[:-1][:self.batch_size], self.valid_data[1:][:self.batch_size]

    def next_batch(self):
        # # returns a randomised, seq_length sized portion of the training data
        # x_batch = []
        # y_batch = []
        # for i in range(self.batch_size):
        #     data = self.data[self.pointer]
        #     n_batch = int(len(data) / ((self.seq_length + 2)))  # number of equiv batches this datapoint is worth
        #     idx = random.randint(0, len(data) - self.seq_length - 2)
        #     x_batch.append(np.copy(data[idx:idx + self.seq_length]))
        #     y_batch.append(np.copy(data[idx + 1:idx + self.seq_length + 1]))
        #     if random.random() < (1.0 / float(n_batch)):  # adjust sampling probability.
        #         # if this is a long datapoint, sample this data more with higher probability
        #         self.tick_batch_pointer()
        # return x_batch, y_batch
        x_data = self.data[:-1]
        y_data = self.data[1:]

        x_split = np.split(x_data[:self.num_batches * self.batch_size], self.num_batches)
        y_split = np.split(y_data[:self.num_batches * self.batch_size], self.num_batches)
        # split_data = np.split(self.data[:self.num_batches * self.batch_size], self.num_batches)

        for x_batch_data, y_batch_data in zip(chain.from_iterable(repeat(x_split)),
                                              chain.from_iterable(repeat(y_split))):
            yield x_batch_data, y_batch_data

    def tick_batch_pointer(self):
        self.pointer += 1
        if (self.pointer >= len(self.data)):
            self.pointer = 0

    def reset_batch_pointer(self):
        self.pointer = 0
