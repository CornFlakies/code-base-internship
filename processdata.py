from matplotlib import axis
import numpy as np
import os
import time
from scipy.ndimage import gaussian_filter1d
import skimage as sk
from typing import List
from dataclasses import dataclass
import matplotlib.pyplot as plt 
from scipy.signal.windows import hann

from helper_functions import HelperFunctions

@dataclass
class Chunks:
    chunk: int
    padding: int
    chunk_length_px: int
    xidx: List
    yidx: List

class ProcessData:
    def __init__(self, input_folder, file_header='npy'):    
        '''
        Initializing ProcessData class with input_folder containing .npy files of all raw image data
        '''
        # Buffer used for writing out numbered files
        self.FILL = 5

        # Loading images
        self.image_paths, self.image_names = HelperFunctions.load_images(input_folder, file_header) 
        
        # Image size in pixels
        try:
            self.image_size = np.size(np.load(self.image_paths[0]), axis=0) # Assuming square image
        except:
            self.image_size = np.size(sk.io.imread(self.image_paths[0]), axis=0)

        # Finding the amount of frames
        self.frames = len(self.image_paths)


    def compute_power_spectrum(self, output_folder, chunk_amount):          
        '''
        Function loads in the raw data and computes the power spectra
        '''
        # Create an output folder if it does not exist already
        HelperFunctions.create_output_folder(output_folder)
        chunks = self.__generate_chunk_data(chunk_amount)
        window = hann(self.frames) 
        avg_spectra = np.zeros(self.frames // 2 + 1)

        # Running through the data
        start_time_tot = time.time()
        for nchunk in range(int(chunks.chunk * chunks.chunk)): 
            # Finding chunk location
            self.nx = chunks.xidx[nchunk % chunks.chunk]
            self.ny = chunks.yidx[nchunk // chunks.chunk]
            
            # Timing run and loading in the chunk as a (frames, px, px) array
            print(f'chunk {nchunk + 1}/{chunks.chunk**2}')
            print(f'nx: {self.nx}, ny: {self.ny}')
            start_time_chunk = time.time()  
            img_chunk = self.__load_chunk(chunks)
           
            # Computing fft 
            print('  computing spectra ...')
            res_fft = np.zeros(self.frames // 2 + 1)
            num = 0
            for ii in range(np.size(img_chunk, axis=1)):
                for jj in range(np.size(img_chunk, axis=2)):
                    res_fft += HelperFunctions.process_temporal_fft(img_chunk[:, ii, jj], window)
                    num += 1
            res_fft /= num            
        
            # Compile data
            avg_spectra += res_fft
            power_spectra = res_fft
            filename = 'power_spectra_' + str(nchunk + 1).zfill(self.FILL) + '.npy' 
            file_destination = os.path.join(output_folder, filename)
            np.save(file_destination, power_spectra)
            print(f'  chunk done in {int(time.time() - start_time_chunk)} seconds ...')
        
        avg_spectra /= chunks.chunk * chunks.chunk
        filename = 'avg_spectra.npy'
        file_destination = os.path.join(output_folder, filename)
        np.save(file_destination, avg_spectra)

        print('done in %s seconds' % (int(time.time() - start_time_tot)))

    def compute_quick_spectra(self, output_folder):
        '''
        Algorithm takes very 10th point of the image to make a quick average of the power spectra
        '''
        HelperFunctions.create_output_folder(output_folder)

        step_size = self.image_size // 10
        npoints = np.arange(step_size, self.image_size - step_size, step_size) 
        xidx, yidx = np.meshgrid(npoints, npoints)

        height  = np.zeros((self.frames, len(npoints), len(npoints)))
        spectra = np.zeros((self.frames // 2 + 1, len(npoints), len(npoints)))
        window  = hann(self.frames) 
        
        for i, image in enumerate(self.image_paths):
            img = np.load(image)
            height[i] = img[xidx, yidx]
            
            if (i % int(self.frames / 10) == 0):
                HelperFunctions.print(f'  {int(np.ceil(i / self.frames * 100))}% ...', 'o')
        
        for ii in range(np.size(spectra, axis=1)):
            for jj in range(np.size(spectra, axis=2)):
                height_loc = height[:, ii, jj]
                spectra[:, ii, jj] = HelperFunctions.process_temporal_fft(height_loc, window)

        spectra = np.mean(spectra, axis=(1, 2))
        filename = 'avg_spectra.npy'
        file_path = os.path.join(output_folder, filename)
        np.save(file_path, spectra)

    def compute_quick_spectra_welch(self, output_folder, fps):
        '''
        Algorithm does the same as compute_quick_spectra but uses a Welch transform
        '''
        HelperFunctions.create_output_folder(output_folder)
        
        step_size = self.image_size // 10
        npoints = np.arange(step_size, self.image_size - step_size, step_size) 
        xidx, yidx = np.meshgrid(npoints, npoints)

        height  = np.zeros((self.frames, len(npoints), len(npoints)))

        for i, image in enumerate(self.image_paths): 
            img = np.load(image)
            height[i] = img[xidx, yidx]
    
            if (i % int(self.frames / 10) == 0):
                HelperFunctions.print(f'  {int(np.ceil(i / self.frames * 100))}% ...', 'o')
       
        powers_tot = np.zeros(HelperFunctions.process_temporal_welch(height[:, 0, 0], fps)[0].shape)
        for ii in range(len(npoints)):
            for jj in range(len(npoints)):
                height_loc = height[:, ii, jj]
                freq, powers = HelperFunctions.process_temporal_welch(height_loc, fps)
                powers_tot += powers 

        powers_tot /= len(npoints)**2
        filename = 'avg_spectra.npy'
        file_path = os.path.join(output_folder, filename)
        np.save(file_path, powers_tot)

        filename = 'frequencies.npy'
        file_path = os.path.join(output_folder, filename)
        np.save(file_path, freq)

    def compute_dispersion_rel(self, output_folder, chunk_amount):
        '''
        Compute dispersion relation of the dataset, either the raw image data from the ffts.
        '''
        # Start computing the chunks (write the complete chunks out to memory??)
        # For a test, lets assume we chose 4x4 chunks, and we only get the middle 4 ones.
        # First, get the relevant chunk data again
        foldername = 'chunks'
        output_folder_chunks = os.path.join(output_folder, foldername, '')
        HelperFunctions.create_output_folder(output_folder_chunks)
        
        foldername = 'slices'
        output_folder_slices = os.path.join(output_folder, foldername, '')
        HelperFunctions.create_output_folder(output_folder_slices)
        
        foldername = 'averages'
        output_folder_averages = os.path.join(output_folder, foldername, '')
        HelperFunctions.create_output_folder(output_folder_averages)

        self.image_size = np.size(np.load(self.image_paths[0]), axis=0)
        chunks = self.__generate_chunk_data(chunk_amount, doPad = False)
        middle_chunks = [5, 6, 9, 10] # ASSUMING 4X4 CHUNKS DON'T FORGET 

        start_time_tot = time.time()
        window = hann(self.frames) 
        for nchunk in middle_chunks:
            # Finding chunk location
            self.nx = chunks.xidx[nchunk % chunks.chunk]
            self.ny = chunks.yidx[nchunk // chunks.chunk]
            
            # Timing run and loading in the chunk as a (frames, px, px) array
            print(f'chunk {nchunk + 1}/{chunks.chunk**2}')
            start_time_chunk = time.time()  
            img_chunk = self.__load_chunk(chunks)
            
            # Computing ffts
            proc_img_chunk = np.zeros((self.frames // 2 + 1, chunks.chunk_length_px, chunks.chunk_length_px))
            for ii in range(np.size(img_chunk, axis=1)):
                for jj in range(np.size(img_chunk, axis=2)):
                    proc_img_chunk[:, ii, jj] = HelperFunctions.process_kspace_fft(img_chunk[:, ii, jj], window)

            filename = 'dispersion_' + str(nchunk + 1).zfill(self.FILL) + '.npy' 
            file_destination = os.path.join(output_folder_chunks, filename)
            np.save(file_destination, proc_img_chunk)
            print(f'  chunk done in {int(time.time() - start_time_chunk)} seconds ...') 
        print('done in %s seconds' % (int(time.time() - start_time_tot)))
        
        # Build slices
        print('converting chunks to slices ...')
        start_time_tot = time.time()
        self.convert_chunks_to_slices(output_folder_chunks, output_folder_slices)
        print('done in %s seconds' % (int(time.time() - start_time_tot)))

        # Build build averages 
        print('building the averages ...')
        start_time_tot = time.time()
        self.build_averages(output_folder_slices, output_folder_averages)
        print('done in %s seconds' % (int(time.time() - start_time_tot)))
    
    def compute_spatial_fft(self, output_folder, chunk_amount):
        '''
        Compute spatial fourier transforms of the raw data set
        '''
        HelperFunctions.create_output_folder(output_folder)
        chunks = self.__generate_chunk_data(chunk_amount)
 
        # Create a synthetic image, and multiply with a window
        padd = (self.image_size - 2 * chunks.padding) // 4
        window = hann(int((1.5 * (self.image_size - 2 * chunks.padding))))
        window_2d = np.outer(window, window)

        for i, image in enumerate(self.image_paths):
            filename = 'fft_' + self.image_names[i]
            img = np.load(image)[chunks.padding:-chunks.padding, chunks.padding:-chunks.padding]
            img = HelperFunctions.gen_synthetic_img(img)
            img_fft = HelperFunctions.process_2d_fft(img, window_2d)
            img_fft = img_fft[padd:-padd, padd:-padd]
            np.save(os.path.join(output_folder, filename), img_fft)

            if (i % int(self.frames / 10) == 0):
                HelperFunctions.print(f'  {int(np.ceil(i / self.frames * 100))}% ...', 'o')
    
    def convert_chunks_to_slices(self, input_folder, output_folder):
        '''
        Function used to convert the chunks back to slices again
        '''
        HelperFunctions.create_output_folder(output_folder)
        self.image_paths, _ = HelperFunctions.load_images(input_folder)

        # Get the size of the chunks by loading in a single image
        sample_img = np.load(self.image_paths[0])
        chunk_size = sample_img.shape
        del sample_img

        # Pre-allocated the joined image, and create indices to store the chunks correctly into the slice
        joined_img = np.zeros((2 * chunk_size[1], 2 * chunk_size[2]))
        npoints = (np.arange(0, 2 * chunk_size[1], chunk_size[1]), np.arange(0, 2 * chunk_size[2], chunk_size[2])) 
        xidx, yidx = np.meshgrid(npoints[0], npoints[1])
        xidx = xidx.flatten()
        yidx = yidx.flatten()
        self.frames = np.size(np.load(self.image_paths[0]), axis=0)
         
        # Loop over the amount of frames and generate slices
        for i in range(self.frames):
            HelperFunctions.print(f'  {int(np.ceil(i / self.frames * 100))}% ...', mode='o')     
            for j, image in enumerate(self.image_paths):
                img = np.load(image)
                img = img[i]
                joined_img[yidx[j]:(yidx[j] + chunk_size[2]), xidx[j]:(xidx[j] + chunk_size[1])] = img
                
            filename = 'disp_' + str(i + 1).zfill(self.FILL) + '.npy'
            dest = os.path.join(output_folder, filename)
            np.save(dest, joined_img)
        
            if (i % int(self.frames / 10) == 0):
                HelperFunctions.print(f'  {int(np.ceil(i / self.frames * 100))}% ...', mode='o')    

    def build_averages(self, input_folder, output_folder):
        '''
        Function used to load in the slices and get a horizontal sice
        '''
        import cv2

        HelperFunctions.create_output_folder(output_folder) 
        self.image_paths, _ = HelperFunctions.load_images(input_folder)

        sample_img = np.load(self.image_paths[0])
        axis_size = np.size(sample_img, axis=1) // 2
        hor_img_slice = np.zeros((len(self.image_paths), axis_size))
        ver_img_slice = hor_img_slice.copy()
        pol_img = np.zeros((len(self.image_paths), axis_size))
        
        for i, image in enumerate(self.image_paths):
            # Load image
            img = np.load(image) 
            
            # Get angular average
            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            value = np.sqrt((center[0]**2 + center[1]**2) / 2)
            transformed_image = cv2.linearPolar(img, center, value, cv2.WARP_FILL_OUTLIERS)
            pol_img[i] = np.mean(transformed_image[:, ::2], axis=0)
            
            # Get kx and ky slice
            hor_img_slice[i] = img[axis_size, axis_size:]
            ver_img_slice[i] = img[axis_size:, axis_size]
            if (i % int(self.frames / 10) == 0):
                HelperFunctions.print(f'  {int(np.ceil(i / self.frames * 100))}% ...', mode='o')    
         
        filename = 'angular_avg.npy'
        np.save(os.path.join(output_folder, filename), pol_img)
        filename = 'kx_slice.npy'
        np.save(os.path.join(output_folder, filename), hor_img_slice)
        filename = 'ky_slice.npy'
        np.save(os.path.join(output_folder, filename), ver_img_slice)

    def compute_grad(self, output_folder):
        from scipy.ndimage import gaussian_filter
        HelperFunctions.create_output_folder(output_folder)

        padding = 50
        smoothing = len(self.image_paths) // 20
        std = 0
        ii = 0

        for image in self.image_paths[::smoothing]:
            img = np.load(image)[padding:-padding, padding:-padding]
            img = gaussian_filter(img, sigma=5)
            gradx, grady = np.gradient(img, 2.6E-2)
            grad = np.sqrt(gradx**2 + grady**2)
            std += np.std(grad)
            ii += 1

            filename = 'gradient_' + str(ii * smoothing).zfill(self.FILL) + '.npy'
            np.save(os.path.join(output_folder, filename), grad)
        
        print('wave steepness approx: ' + str(std / ii) + ' cm')

    def compute_conv_factor(self):
        return None

    def plot_power_spectra(self):
        '''
        Plot the power spectra, one needs to input the chunk data
        '''
        from numpy.fft import rfftfreq

        def power_law(A, x, p):
            return A * (x/x[0])**(p)

        height = [290] 
        lower  = [100]
        upper  = [150]

        height1 = [40] 
        lower1  = [150]
        upper1  = [260]
        
        for i, image in enumerate(self.image_paths):
            img = np.load(image)
            freq = rfftfreq(np.size(img, axis=0) * 2 - 1, d=1/125) * 2 * np.pi
            plt.figure()
            plt.title('Power spectrum, $f_{max} = 4\, Hz$, $A=20\, mm$, $h0 = 4.5\, cm$')
            plt.loglog(freq, img**2, '.-')
            plt.loglog(freq[lower[i]:upper[i]], power_law(height[i], freq[lower[i]:upper[i]], -4), '-', label='$\omega^{-4}$', color='red')
            plt.loglog(freq[lower1[i]:upper1[i]], power_law(height1[i], freq[lower1[i]:upper1[i]], -6), '-', label='$\omega^{-6}$', color='orange')
            plt.axvline(x = int(4 * 2 * np.pi), linestyle='--', color='black', label='$\omega_{max} = 4\cdot 2\pi\, (rad\, Hz)$')
            plt.xlabel('$\omega\, (rad\, Hz)$')
            plt.ylabel('$PSD_{\mathcal{F}}\, [h(t)]$')
            plt.legend()
            plt.grid()
        plt.show()

    def plot_welch_spectra(self):
        
        def power_law(A, x, p):
            return A * (x/x[0])**(p)

        height = [0.04, 0.0004] 
        lower  = [19, 40]
        upper  = [49, 45]

        data = []
        for img in self.image_paths:
            data.append(np.load(img))
        freq = data[1] * 2 * np.pi
        power = data[0]

        plt.figure()
        plt.loglog(freq, power, '.-')
        plt.loglog(freq[lower[0]:upper[0]], power_law(height[0], freq[lower[0]:upper[0]], -5), '-', label='$\omega^{-5}$', color='red')
        plt.axvline(x = int(4 * 2 * np.pi), linestyle='--', color='black', label='$\omega_{max} = 4\cdot 2\pi\, (rad\, Hz)$')
        plt.title('Welch PSD $f=4\, Hz$, $A=20\, mm$, $h_0 = 3\, cm$')
        plt.xlabel('$\omega\, (rad\, Hz)$')
        plt.ylabel('$PSD_{\mathcal{W}}\, [h(t)]$')
        plt.legend()
        plt.grid()

        plt.figure()
        plt.loglog(freq, power / freq**(-4), '.-')
        plt.grid()
        plt.show()

    def plot_dispersion_rel(self, h0, conver_factor):
        '''
        Plot the dispersion relation using as well as the analytical gravity capillary solution
        '''
        from numpy.fft import rfftfreq

        kana  = np.linspace(0, 1000, 2000)
        omega = np.sqrt(HelperFunctions.gravcap_dispersion_sq(kana, h0))  
    
        for image in self.image_paths:
            img = np.load(image)
            omega_space = rfftfreq(np.size(img, axis=0) * 2, d=1/125) * 2 * np.pi
            kspace = 2 * np.pi * rfftfreq(np.size(img, axis=1) * 2 + 1, d=conver_factor) 
        
            plt.figure()
            plt.plot(omega, kana, '--', color='black', label='analytical dispersion') 
            plt.pcolor(omega_space, kspace[1:], np.log(img[:, 1:].T))
            plt.xlabel('$\omega (rad/s)$')
            plt.ylabel('$k (rad/m)$')
            plt.legend()
            plt.colorbar()
        plt.show() 

    def plot_slices(self):
        '''
        Plot a few slices length wise for a few pixels to see what is going on in the time domain
        ''' 
        step_size = self.image_size // 10
        npoints = np.arange(step_size, self.image_size - step_size, step_size) 
        xidx, yidx = np.meshgrid(npoints, npoints)

        height  = np.zeros((self.frames, len(npoints), len(npoints)))
        spectra = np.zeros((self.frames // 2 + 1, len(npoints), len(npoints)))
        mean_intensity = np.zeros(self.frames)
        for i, image in enumerate(self.image_paths):
            img = np.load(image)
            height[i] = img[xidx, yidx]
            mean_intensity[i] = np.mean(img)
            if (i % int(self.frames / 10) == 0):
                HelperFunctions.print(f'  {int(np.ceil(i / self.frames * 100))}% ...', 'o')
        
        plt.figure()
        plt.plot(height[:, 7, 7], '.-')
        plt.xlabel('frame')
        plt.ylabel('h(frame)')
        plt.grid()
        plt.show()

    def convert_to_npy(self, output_folder, crop = [None, None, None, None]):
        '''
        Function used to convert raw image data to npy files
        '''
        HelperFunctions.create_output_folder(output_folder)
        
        nx1, nx2, ny1, ny2 = crop

        for i, image in enumerate(self.image_paths):
            img = sk.io.imread(image)[nx1:nx2, ny1:ny2]
            filename = self.image_names[i].split('.tif')[0] + '.npy'
            file_path = os.path.join(output_folder, filename)
            np.save(file_path, img)

    def square_images(self, output_folder):
        '''
        Function used to square the images
        '''
        HelperFunctions.create_output_folder(output_folder)

        for i, image in enumerate(self.image_paths):
            img = np.load(image)
            img = HelperFunctions.square_img(img)
            file_path = os.path.join(output_folder, self.image_names[i])
            np.save(file_path, img)

    def __generate_chunk_data(self, chunk_amount, doPad = True):
        '''
        Helper function used to generate chunk data for the run
        '''
        # Code assumes an 1024 x 1024 image, the chunk parameter determines how many blocks of an image are taken
        if (self.image_size % chunk_amount != 0):
            raise Exception(f'Cannot divide {self.image_size} x {self.image_size} image into desired chunks')
        else:
            # Create padding around the dataset to ignore weird edge effects
            if doPad is True:
                padding = chunk_amount * 4 
            else:
                padding = 0

            chunk_length_px = (self.image_size - 2 * padding) // chunk_amount 
            xidx = np.arange(padding, self.image_size - padding, chunk_length_px, dtype=int).tolist()
            yidx = xidx.copy()
            return Chunks(chunk_amount, padding, chunk_length_px, xidx, yidx)
    
    def __load_chunk(self, chunk, dtype='float'):
        '''
        Helper function to load in the chunks using np.load
        '''
        if dtype == 'float':
            data_type = np.float64
        elif dtype == 'complex':
            data_type = np.complex128
        else:
            raise Exception('Data type not recognized')

        # Running through the list of images
        img_chunk = np.zeros((self.frames, chunk.chunk_length_px, chunk.chunk_length_px), dtype=data_type)
        for i, image in enumerate(self.image_paths):
            img = np.load(image)
            img_chunk[i] = img[self.ny:(self.ny + chunk.chunk_length_px), self.nx:(self.nx + chunk.chunk_length_px)]

            if (i % int(self.frames / 10) == 0):
                HelperFunctions.print(f'  {int(np.ceil(i / self.frames * 100))}% ...', mode='o')
        
        return img_chunk 
  
    




    
