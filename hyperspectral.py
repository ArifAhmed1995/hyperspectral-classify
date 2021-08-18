'''This code is an implementation of algorithm in the paper

Spectral-Spatial Classification of Hyperspectral Images Using ICA and
Edge-Preserving Filter via an Ensemble Strategy by Xia et. al.

IEEE Transactions on Geoscience and Remote Sensing,
Institute of Electrical and Electronics Engineers,
2016, 54 (8), pp.4971 - 4982. 10.1109/TGRS.2016.2553842. hal-01379723

This particular implementation achieves a average classification accuracy
of around 80%.

It is inferior to more recent techniques such as Deep Neural Nets(eg. MRA-NET)
which achieves above 99%.

In any case, this mini-project was done out of curiosity and to implement
an image processing algorithm from scratch.

-------------------------------------------------------------------------------

Data
====

The dataset(Pavia Centre and University) can be got from here:
http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

'''

import random

import numpy as np
from numpy import dot

import pandas as pd

from itertools import chain

from scipy.io import loadmat

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA

class HyperSpec():
    def __init__(self):
        # Load MAT files
        self.im = loadmat('Pavia.mat')['pavia']
        self.im_gt = loadmat('Pavia_gt.mat')['pavia_gt']

        # Set high_limit to how much of the image
        # to operate on. Comment the next three lines
        # to run on entire image
        high_limit = 400
        self.im = self.im[:high_limit, :high_limit, :]
        self.im_gt = self.im_gt[:high_limit, :high_limit]

        self.im_shape = self.im.shape
        self.im_gt_shape = self.im_gt.shape

        # Images after convolution
        self.edge_recovered_image = None
        self.edge_recovery_padded_image = None

        # Dataframe to hold data
        self.df = None
        self.combined_df = None

        # Subset params
        self.m = 10
        self.k = 5

        # Contains band values  
        self.feature_subsets = []
        self.feature_subset_indices = []
        
        # Contains band values after ICA
        self.ica_subsets = []

        # Processed images
        self.rgf_images = []

        # Grad descent params
        self.lr = 0.03
        self.num_epochs = 200

        self.unique_band_indices = []
        self.num_unique_bands = 0

    def pixel_band_value_matrix(self):
        '''Returns matrix of pixels and corresponding band values
        '''
        return self.im.reshape(-1, self.im.shape[2])

    def extract_pixels(self):
        '''Load image into dataframe to make further data wrangling easier
        '''
        q = self.pixel_band_value_matrix()

        # Load into dataframe
        self.df = pd.DataFrame(data=q)

        # Scale input data for SGD
        self.df /= 1000

        # Concatenate the dataframe with output values(ground truth), columnwise
        self.band_values = self.df
        self.band_values.columns = [f'band{i}' for i in range(1, 1 + self.im_shape[2])]

        self.df = pd.concat([self.df, pd.DataFrame(data=self.im_gt.ravel())], axis=1)
        self.df.rename(columns={ self.df.columns[-1]: "class" }, inplace = True)

    def generate_feature_subsets(self):
        '''Takes a random subset of band values(M bands) for K subsets
        Unique band indices are extracted and stored.
        '''
        # K subsets, M bands each
        self.feature_subset_indices = [random.sample(list(self.band_values.columns), self.m) for i in range(self.k)]
        self.feature_subsets = [self.band_values[index_list] for index_list in self.feature_subset_indices]
        self.unique_band_indices = list(set(list(chain.from_iterable(self.feature_subset_indices))))

    def sigmoid(self, x):
        '''Sigmoid function
        '''
        return 1/(1 + np.exp(-x))

    def fast_ICA(self):
        '''Independent Components Analysis: This is done across the spectral bands
        to separate the values and ensure spectrally independent components.
        '''
        for subset in self.feature_subsets:
            num_spectral_bands = subset.shape[1]

            # Perform ICA
            ica = FastICA()
            source_pixels = ica.fit_transform(subset)

            # Standardize output and reshape to subset shape
            source_pixels /= source_pixels.std(axis=0)
            ica_subset = source_pixels.reshape(subset.shape)

            # Add to ica_subsets
            self.ica_subsets.append(ica_subset.reshape(self.im_shape[0], self.im_shape[1], num_spectral_bands))

    def extract_independent_components(self):
        '''Similar result as FastICA but lower performance
        '''
        for subset in self.feature_subsets:
            num_spectral_bands = subset.shape[1]
            w_tanh_adam = np.eye(num_spectral_bands)
            m_tanh_adam = np.zeros_like(w_tanh_adam)
            v_tanh_adam = np.zeros_like(w_tanh_adam)

            for iter in range(self.num_epochs):
                U = np.tanh(dot(subset, w_tanh_adam))
                grad = np.linalg.inv(w_tanh_adam.T) - (2/len(subset)) * dot(subset.T, U)

                m_tanh_adam = 0.9 * m_tanh_adam + 0.1 * grad
                v_tanh_adam = 0.999 * v_tanh_adam + 0.001 * grad ** 2

                m_hat = m_tanh_adam/0.1
                v_hat = v_tanh_adam/0.001
                w_tanh_adam += self.lr/np.sqrt(v_hat + 1e-30) * m_hat

            source_pixels = dot(subset, w_tanh_adam)

            # Normalize and add to ica subsets
            source_pixels /= source_pixels.std(axis=0)
            ica_subset = source_pixels.reshape(subset.shape)
            self.ica_subsets.append(ica_subset.reshape(self.im_shape[0], self.im_shape[1], num_spectral_bands))

    def pad_image(self, image, kernel_h, kernel_w):
        '''Padding to keep original image resolution constant. This is equivalent to padding='same'
        during convolutional filters in ML libraries
        '''
        if len(image.shape) == 3:
            image_pad = np.pad(image, pad_width=((kernel_h // 2, kernel_h // 2), (kernel_w // 2, kernel_w // 2), (0,0)),
                    mode='constant', constant_values=0).astype(np.float32)
        elif len(image.shape) == 2:
            image_pad = np.pad(image, pad_width=((kernel_h // 2, kernel_h // 2), (kernel_w // 2, kernel_w // 2)),
                    mode='constant', constant_values=0).astype(np.float32)

        return image_pad

    def generate_gaussian_kernel(self, filter_size, sigma):
        '''Generates a gaussian kernel. This is a constant kernel, means that the
        kernel values do not depend on the pixel values on which it is operating on.
        '''
        K_p = 0
        m = n = filter_size // 2
        gaussian_kernel = np.zeros((filter_size, filter_size), np.float32)

        for x in range(-m, m + 1):
            for y in range(-n, n + 1):
                gaussian_kernel[x + m][y + n] = np.exp(-(x**2 + y**2)/(2 * sigma ** 2))
                K_p += gaussian_kernel[x + m][y + n]

        return gaussian_kernel/K_p

    def generate_edge_recovery_kernel(self, image, filter_size, sigma_s, sigma_r, x_offset, y_offset):
        '''Generates an edge recovery kernel. This is a variable kernel, means that the
        kernel values are dependent on the pixel values it is currently operating on.
        '''
        K_p = 0
        m = n = filter_size // 2
        edge_recovery_kernel = np.zeros((filter_size, filter_size), np.float32)

        for x in range(-m, m + 1):
            for y in range(-n, n + 1):
                edge_recovery_kernel[x + m][y + n] = np.exp(-(x**2 + y**2)/(2 * sigma_s ** 2) - ((image[x + x_offset][y + y_offset] - image[x_offset][y_offset]) ** 2)/(2 * sigma_r ** 2))
                K_p += edge_recovery_kernel[x + m][y + n]

        return edge_recovery_kernel/K_p

    def small_structure_removal(self, image: np.ndarray, sigma: np.float):
        '''In other words, this is a gaussian filter applied to blur the image.
        '''
        print("Start SSR.......")
        num_spectral_bands = np.uint16(image.shape[2])

        filter_size = 1 + 2 * int(4 * sigma + 0.5)
        m = n = filter_size // 2

        padded_image = self.pad_image(image, filter_size, filter_size)
        image_conv = np.zeros(padded_image.shape, dtype=np.float32)

        # Convolution
        gaussian_kernel = self.generate_gaussian_kernel(filter_size, sigma)

        for band_index in range(num_spectral_bands):
            for i in range(m, padded_image.shape[0] - m):
                for j in range(n, padded_image.shape[1] - n):
                    x = padded_image[i - m : i - m + filter_size, j - n : j - n + filter_size, band_index]
                    x = x.flatten() * gaussian_kernel.flatten()
                    image_conv[i][j] = x.sum()
        
        h_end = -m
        w_end = -n

        if m == 0:
            return image_conv[m:, n:w_end]

        if n == 0:
            return image_conv[m:h_end, n:]
        
        print("End SSR.........")
        return image_conv[m:h_end, n:w_end]
        

    def convolution_edge_recovery(self, i, j, m, n, filter_size, band_index, sigma_s, sigma_r):
        '''Convolution step for edge recovery process. This kernel is
        re-generated per pixel since it depends on the image values.
        '''
        x = self.edge_recovery_padded_image[i - m : i - m + filter_size, j - n : j - n + filter_size, band_index]
        x = x.flatten() * self.generate_edge_recovery_kernel(self.edge_recovery_padded_image[:, :, band_index], filter_size, sigma_s, sigma_r, i, j).flatten()
        self.edge_recovered_image[i][j] = x.sum()
        return True

    def edge_recovery(self, ssr_image, sigma_s, sigma_r):
        '''Edge Recovery Step. This is the second step of the rolling guidance filter and
        helps to identify the edges which separate one class of region from another.
        '''
        print("Starting Edge Recovery....................")
        K_p = 0
        Jt_1 = ssr_image
        num_spectral_bands = ssr_image.shape[2]
        filter_size = 1 + 2 * int(4 * sigma_s + 0.5)
        m = n = filter_size // 2

        self.edge_recovery_padded_image = self.pad_image(ssr_image, filter_size, filter_size)
        self.edge_recovered_image = np.zeros(self.edge_recovery_padded_image.shape, dtype=np.float32)

        # This is the major bottleneck step in the entire algorithm. It is mainly expensive not because of the
        # nested loops but solely due to the fact that the edge recovery kernel has to re-generated for each pixel
        # of the image.
        # Improvements to performance are definitely possible though, but since there are better performing 
        # techniques to analyze hyperspectral images using neural nets, it seemed futile to pursue performance
        # improvement.
        for iter in range(4):
            # Every iteration makes edges clearer and more identifiable by the classifier later
            for band_index in range(num_spectral_bands):
                for i in range(m, self.edge_recovery_padded_image.shape[0] - m):
                    for j in range(n, self.edge_recovery_padded_image.shape[1] - n):
                        self.convolution_edge_recovery(i, j, m, n, filter_size, band_index, sigma_s, sigma_r)

        # This convolution does not maintain original image size, so truncate the convolved output.
        h_end = -m
        w_end = -n

        if m == 0:
            self.edge_recovered_image = self.edge_recovered_image[m:, n:w_end]
        if n == 0:
            self.edge_recovered_image = self.edge_recovered_image[m:h_end, n:]

        print("End Edge Recovery.......")
        return self.edge_recovered_image[m:h_end, n:w_end]

    def rolling_guidance_filter(self):
        '''Rolling Guidance Filter(RGF) consists of gaussian filtering and then edge recovery, both of which
        are convolution steps performing on an image which has undergone ICA to seperate mixing of band information
        '''
        self.rgf_images = [self.edge_recovery(self.small_structure_removal(image, 1), 1, 1).reshape(-1, self.m) for image in self.ica_subsets]

    def combine_bands(self):
        '''Iterate over the images after RGF operation and then club together the unique bands.
        '''
        self.combined_df = pd.DataFrame()
        for subset_index in range(len(self.rgf_images)):
            for index, band_label in enumerate(self.feature_subset_indices[subset_index]):
                if band_label in self.unique_band_indices:
                    self.combined_df = pd.concat([self.combined_df, pd.DataFrame(data=self.rgf_images[subset_index][:, index])], axis=1)
                    self.unique_band_indices.remove(band_label)
                    self.num_unique_bands += 1

    def classifier(self):
        '''Random Forest Classifier to classify images after RGF step + concatenation of unique
        bands across subsets.
        '''
        X = self.combined_df.to_numpy()
        X = X.reshape(-1, self.num_unique_bands)
        y = self.im_gt.reshape(-1, 1).ravel()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print(metrics.accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    hyp = HyperSpec()

    hyp.extract_pixels()
    hyp.generate_feature_subsets()

    #hyp.extract_independent_components()
    hyp.fast_ICA()

    print("Starting RGF........")
    hyp.rolling_guidance_filter()
    print("End RGF.......")

    hyp.combine_bands()
    hyp.classifier()
