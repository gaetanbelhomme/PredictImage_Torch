import argparse
import os
import numpy as np
from six.moves import cPickle as pickle
import nrrd

# PARSER :

parser = argparse.ArgumentParser()


# Folders names :
parser.add_argument('-test', action='store', dest='test', help='testing folder',
                    default='SquareRegisteredTestingData')
parser.add_argument('-train', action='store', dest='train', help='training folder',
                    default='SquareRegisteredTrainingData')

parser.add_argument('-start', action='store', dest='start', help='data_start folder',
                    default='V06')
parser.add_argument('-end', action='store', dest='end', help='data_end folder',
                    default='V24')

parser.add_argument('-root', action='store', dest='root', help='root of testing and training folders. Default : '
                                                               'courant repository',
                    default='/work/gaetanb/Data/')

# Data storage :
parser.add_argument('-store', action='store', dest='filetype', help='chose the storage type : npz/npy/pickle',
                    default='npy')
parser.add_argument('-o', action='store', dest='outputfilename', help='name of file.npz/.pickle',
                    default='DataNrrd2D')
parser.add_argument('-ho', action='store', dest='headerfilename', help='name of headers file.npz/.pickle',
                    default='DataNrrd2D_headers')
parser.add_argument('-pathData', action='store', dest='pathData', help='save data in this folder',
                    default='../Data/input/')

# Image size :
parser.add_argument('-size', action='store', dest='image_size', help='image size',
                    default=64)
parser.add_argument('-r', action='store_true', dest='resize', help='resize images into weight*height*numChannel. '
                                                                   'Default (resize=True) numChannel*weight*height',
                    default=True)
parser.add_argument('-numC', action='store', dest='num_channel', help='number of channel',
                    default=1)

# Data size :
parser.add_argument('-test_size', action='store', dest='test_size', help='size of testing dataset',
                    default=1000)
parser.add_argument('-train_size', action='store', dest='train_size', help='size of training dataset',
                    default=2000)
parser.add_argument('-valid_size', action='store', dest='valid_size', help='size of valid dataset',
                    default=100)

options = parser.parse_args()


# INITIALIZATION :

test = options.test
train = options.train

start = options.start
end = options.end

fileType = options.filetype
outputName = options.outputfilename
headeroutput = options.headerfilename
pathData = options.pathData

image_size = options.image_size
num_channels = options.num_channel
root = options.root

resize = options.resize

test_size = options.test_size
train_size = options.train_size
valid_size = options.valid_size

pixel_depth = 255.0


# FUNCTIONS :

# Check the presence of folders
# Add the path to the folders
def CheckFolders(folder, root_name=root):
    FolderPath = os.path.join(root_name, folder)

    if os.path.isdir(FolderPath):
        print folder, "is present"
        return FolderPath
    else:
        raise Exception('Failed to find %s' % folder)


# Normalise les images
# Return dataset, images array

def load_images(folder):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
    headerset = []
    print(folder)
    num_images = 0
    for Image in image_files:
        image_file = os.path.join(folder, Image)
        try:
            nrrd_image, header = nrrd.read(image_file)
            headerset.append(header)
            image_data = (nrrd_image.astype(float) - pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images += 1
        except IOError as e:
            print('Could not read: ', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    print ('Full dataset tensor:', dataset.shape)
    print ('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset, headerset


# Construit un fichier pickle dans lequel sont sauvees les images normalisees.
# Renvoie le nom de ces fichiers pickle.

def maybe_pickle(start_folder, end_folder, force=False):
    dataset_names = []
    headerset_names = []
    for folder in start_folder, end_folder:
        set_filename = folder + '.pickle'
        set_headername = folder + '_header.pickle'
        dataset_names.append(set_filename)
        headerset_names.append(set_headername)
        if os.path.exists(set_filename) and os.path.exists(set_headername) and not force:
            # You may override by setting force=True.
            print('%s already present and %s - Skipping pickling.' % (set_filename, set_headername))
        else:
            print('Pickling %s.' % set_filename)
            dataset, headers = load_images(folder)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)
            try:
                with open(set_headername, 'wb') as f:
                    pickle.dump(headers, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save header to', set_headername, ':', e)

    return dataset_names, headerset_names


# Build 2 arrays according a number of images:
# - dataset_start array
# - labels_end array
def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset_s = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        dataset_e = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        dataset_h = np.ndarray(nb_rows)
    else:
        dataset_s, dataset_e, dataset_h = None, None, None
    return dataset_s, dataset_e, dataset_h


# Settings : pickle_files (folder sorted by labels with its images), train size & valid size.
# Permet de bien arranger les donnees, cad de sortir un vecteur contenant le nombre d images voulues,
# avec autant d'images par classe, ainsi que le vecteur de label associe.
# Creation d'un vecteur de la taille du nombres d images voulues, pour le train et le valid.
# Recuperation du nombre d'images par classe pour le valid et le train dataset(floor(valid_size, num_classes)
# Ouverture du pickle, cad d'une classe.
# Recuperation des images de la classe et melange de ces images.
# Recuperation de toutes du nombre d'images defini ci-dessuse que l'on met dans un vecteur.
# Creation du vecteur de labels associes. Pareil pour le train et le valid dataset.
# Retour de tous lse vecteurs :
# valid_dataset, valid_labels, train_dataset, train_label


def merge_datasets(pickle_files, pickle_headers, train_size, valid_size=0):
    valid_dataset_start, valid_dataset_end, valid_dataset_header = make_arrays(valid_size, image_size)
    validation_sets = [valid_dataset_start, valid_dataset_end]
    validation_headers = [valid_dataset_header, valid_dataset_header]

    train_dataset_start, train_dataset_end, train_dataset_header = make_arrays(train_size, image_size)
    train_sets = [train_dataset_start, train_dataset_end]
    train_headers = [train_dataset_header, train_dataset_header]

    # # lets shuffle the images to have random validation and training set
    #permutation = np.random.permutation(train_sets[0].shape[0])

    start_v, start_t = 0, 0
    end_v, end_t = valid_size, train_size
    end_l = valid_size + train_size
    for i in 0, 1:
        try:
            with open(pickle_headers[i], 'rb') as h:
                try:
                    with open(pickle_files[i], 'rb') as f:
                        data_set = pickle.load(f)
                        header_set = pickle.load(h)
                        # # lets shuffle the images to have random validation and training set
                        #data_set = data_set[permutation, :, :]

                        if validation_sets[i] is not None:
                            valid_data = data_set[:valid_size, :, :]
                            validation_sets[i][start_v:end_v, :, :] = valid_data

                            valid_head = header_set[:valid_size]
                            validation_headers[i] = valid_head
                            print 'valid', np.shape(validation_headers)

                        train_data = data_set[valid_size:end_l, :, :]
                        train_sets[i][start_t:end_t, :, :] = train_data
                        print 'train', np.shape(train_sets)

                        train_head = header_set[valid_size:end_l]
                        train_headers[i] = train_head
                        print np.size(train_headers[i])

                except Exception as e:
                    print('Unable to process data from', pickle_files[i], ':', e)
                    raise
        except Exception as e:
            print('Unable to process data from', pickle_headers[i], ':', e)
            raise
    return validation_sets[0], validation_sets[1], validation_headers, train_sets[0], train_sets[0], train_headers


def randomize(dataset_s, dataset_e, dataset_h):
  permutation = np.random.permutation(dataset_s.shape[0])

  shuffled_dataset_h_0 = []
  shuffled_dataset_h_1 = []
  for key in permutation:
      shuffled_dataset_h_0.append(dataset_h[0][key])
      shuffled_dataset_h_1.append(dataset_h[1][key])

  shuffled_dataset_h = []
  shuffled_dataset_s = dataset_s[permutation, :, :]
  shuffled_dataset_e = dataset_e[permutation, :, :]
  shuffled_dataset_h.append(shuffled_dataset_h_0)
  shuffled_dataset_h.append(shuffled_dataset_h_1)
  return shuffled_dataset_s, shuffled_dataset_e, shuffled_dataset_h


def reformat(dataset_start, dataset_end):
    if resize:
        dataset_start = dataset_start.reshape((-1, num_channels, image_size, image_size)).astype(np.float32)
        dataset_end = dataset_end.reshape((-1, num_channels, image_size, image_size)).astype(np.float32)
    else:
        dataset_start = dataset_start.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
        dataset_end = dataset_end.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset_start, dataset_end


def storeNpz(save):
    nfile = pathData + outputName + '.npz'
    try:
        np.savez(nfile, train_dataset_start=save["train_dataset_start"], train_dataset_end=save["train_dataset_end"],
                 valid_dataset_start=save["valid_dataset_start"], valid_dataset_end=save["valid_dataset_end"],
                 test_dataset_start=save["test_dataset_start"], test_dataset_end=save["test_dataset_end"])

    except Exception as e:
        print('Unable to save data to', nfile, ':', e)
        raise
    statinfo = os.stat(nfile)
    print('Numpy file size:', statinfo.st_size)


def storeNpy(save):
    for key, value in save.iteritems():
        nfile = pathData + key + '.npy'
        try:
            np.save(nfile, value)
        except Exception as e:
            print('Unable to save data to', nfile, ':', e)
            raise
        statinfo = os.stat(nfile)
        print(nfile, ' size:', statinfo.st_size)


def storePick(save, name):
    pickle_file = pathData + name + '.pickle'
    try:
        f = open(pickle_file, 'wb')
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)


def store(save, type=fileType, name=outputName):
    if type == 'pickle':
        storePick(save, name)
    elif type == 'npz':
        storeNpz(save)
    else:
        storeNpy(save)
