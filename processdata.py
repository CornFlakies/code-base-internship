import os
import cv2
import time
import numpy as np
import skimage as sk
from typing import List
import matplotlib
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt 
from dataclasses import dataclass
from scipy.signal.windows import hann
from helper_functions import HelperFunctions

font = {'size'   : 12}

matplotlib.rc('font', **font)

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
        
        foldername = 'averages'
        output_folder_averages = os.path.join(output_folder, foldername, '')
        HelperFunctions.create_output_folder(output_folder_averages)

        shape = np.load(self.image_paths[0]).shape
        box_size = 300 #px
        ctr_idx = [[(shape[0]- box_size) // 2, (shape[0] + box_size) // 2],[(shape[1] - box_size) // 2, (shape[1] + box_size) // 2]]
        # Generate a chunk containing the center of the image 
        chunk = np.zeros((len(self.image_paths), box_size, box_size))
        for i, img in enumerate(self.image_paths):
            image = np.load(img)
            image = image[ctr_idx[0][0]:ctr_idx[0][1], ctr_idx[1][0]:ctr_idx[1][1]]
            chunk[i] = image

        # Spatially fourier transform the image
        chunk_fft = np.zeros((np.size(chunk, axis = 0) // 2 + 1, chunk.shape[1], chunk.shape[2]))
        for ii in range(np.size(chunk, axis=1)):
            for jj in range (np.size(chunk, axis=2)):
                chunk_fft[:, ii, jj] = HelperFunctions.process_kspace_fft(chunk[:, ii, jj])
        
        height, width = chunk[0].shape
        pol_img = np.zeros((len(chunk_fft), box_size // 2))
        for i, slice in enumerate(chunk_fft): 
            # Get angular average
            center = (width // 2, height // 2)
            value = np.sqrt((center[0]**2 + center[1]**2) / 2)
            transformed_image = cv2.linearPolar(slice, center, value, cv2.WARP_FILL_OUTLIERS)
            pol_img[i] = np.mean(transformed_image[:, ::2], axis=0)                       
            

        # Get kx and ky slice
        hor_img_slice = chunk_fft[:, box_size // 2, box_size // 2:]
        ver_img_slice = chunk_fft[:, box_size // 2:, box_size // 2]
        
        print(pol_img.shape)
        print(hor_img_slice.shape)
    
        # Save the angular avgs and the chunk
        filename = 'angular_avg.npy'
        np.save(os.path.join(output_folder_averages, filename), pol_img)
        filename = 'kx_slice.npy'
        np.save(os.path.join(output_folder_averages, filename), hor_img_slice)
        filename = 'ky_slice.npy'
        np.save(os.path.join(output_folder_averages, filename), ver_img_slice) 
        filename = 'chunk'
        np.save(os.path.join(output_folder_chunks, filename), chunk)

    
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
    
    def compute_grad(self, output_folder):
        from scipy.ndimage import gaussian_filter
        HelperFunctions.create_output_folder(output_folder)

        padding = 25
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
        plt.title('Welch PSD $f=4\, Hz$, $A=20\, mm$, $h_0 = 5\, cm$')
        plt.xlabel('$\omega\, (rad\, Hz)$')
        plt.ylabel('$PSD_{\mathcal{W}}\, [h(t)]$')
        plt.legend()
        plt.grid()

        plt.figure()
        plt.loglog(freq, power / freq**(-6), '.-')
        plt.grid()
        plt.show()

    def plot_dispersion_rel(self, h0, conver_factor, fps):
        '''
        Plot the dispersion relation using as well as the analytical gravity capillary solution

        - h0            : water height in meter   cm      
        - conver_factor : pixel conversion factor cm/px  
        - fps           : frames per second       frames/s

        '''
        from numpy.fft import rfftfreq

        def power_law(A, x, p):
            return A * (x/x[0])**(p)
        
        h0 *= 1E-2             # m
        conver_factor *= 1E-2  # m/px

        labels = ['10mm', '15mm', '20mm', '5mm']
        markers = ['o', 'D', 'v', '>']
        
        plt.figure()
        plt.title('')
        plt.grid()
        plt.xlabel(r'$k\. (rad/m)$')
        plt.ylabel(r'$E(k)$')

        # Plot the analytical power laws
        height = 3.6E7
        start = 730
        finish = 5000
        Ekana = np.linspace(start, finish, 100)
        plt.loglog(Ekana, power_law(height, Ekana, -5/2), '--',color='black', label=r"$E_k \sim k^{-5/2}$")

        ## Plot the analytical power laws
        #height = 520000
        #start = 2000
        #finish = 5500
        #Ekana = np.linspace(start, finish, 100)
        #plt.loglog(Ekana, power_law(height, Ekana, -5/2), '--',color='black')

        # Plot the experimental power law in k-space
        for i, image in enumerate(self.image_paths[:]):
            img = np.load(image)
            E_k = np.sum(img, axis=0) / len(img)
            kspace = np.pi * rfftfreq(np.size(img, axis=1) * 2 + 1, d=conver_factor) 
            plt.loglog(kspace[1:], E_k**2, marker=markers[i], markersize=4, label=f'A = {labels[i]}')
        plt.legend()
        plt.xlabel(r'$k\. (rad/s)$')
        plt.ylabel(r'$PSD(k)$')

        # Plot the analytical power laws
        plt.figure()
        plt.grid() 
        height = 7.2E8
        start = 65
        finish = 220
        Eomegaana = np.linspace(start, finish, 100)
        plt.loglog(Eomegaana, power_law(height, Eomegaana, -4), '--',color='black', label=r'$\omega\sim -4$')

        # Plot the analytical power laws
        #height = 3.7E7
        #start = 66
        #finish = 120
        #Eomegaana = np.linspace(start, finish, 100)
        #plt.loglog(Eomegaana, power_law(height, Eomegaana, -4), '--',color='black', label=r'$\omega\sim -4$')
        
        # Plot the experimental power law in omega-space
        for i, image in enumerate(self.image_paths[:]):
            img = np.load(image)
            E_omega = np.sum(img, axis=1) / len(img[0, :])
            omegaspace = 2 * np.pi * rfftfreq(np.size(img, axis=0) * 2 + 1, d=1/fps)
            plt.loglog(omegaspace[2:], E_omega[1:]**2, marker=markers[i], markersize=4, label=f'A = {labels[i]}')
        plt.axvline(x=int(4*2*np.pi), color='black', linestyle='--')
        plt.legend()
        plt.xlabel(r'$\omega\. (rad/s)$')
        plt.ylabel(r'$PSD(\omega)$')
        
        # Plot the dispersion relation with the gravity capillary dispersion relation
        kana  = np.linspace(0, 1200, 500)
        omega = np.sqrt(HelperFunctions.gravcap_dispersion_sq(kana, h0))  
       
        for i, image in enumerate(self.image_paths):
            print(image)
            img = np.load(image)
            omegaspace = 2 * np.pi * rfftfreq(np.size(img, axis=0) * 2 + 1, d=1/fps)
            #kspace = 2 * np.pi * rfftfreq(1024 // 2 + 1, d=conver_factor)
            kspace = np.pi * rfftfreq(np.size(img, axis=1) * 2 + 1, d=conver_factor)  

            plt.figure()
            plt.plot(omega, kana, '--', linewidth=1, color='black', label=r'Gravity Capillary Disp. Relation') 
            plt.pcolor(omegaspace[1:], kspace[1:], np.log(img[1:, 1:].T), vmin=5, vmax=15) 
            plt.xlim([0, 300])
            plt.ylim([0, 1500])
            plt.xlabel('$\omega (rad/s)$')
            plt.ylabel('$k (rad/m)$')
            plt.legend()
            cbar = plt.colorbar()
            cbar.ax.set_ylabel(r'$PSD_{\omega, k} (arb. units)$')
        plt.legend()
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
        for i, image in enumerate(self.image_paths[:300]):
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

        for i, image in enumerate(self.image_paths[:]):
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
                padding = 10 
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
  
    




    
