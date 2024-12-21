import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift


  

"Fourier Transform"
def ift(image):
    shape = image.shape
    image = image[:, :, :3].mean(axis=2)  # Convert to grayscale

    # Calculate the 2D Fourier Transform
    ft = ifftshift(image)
    ft = ifft2(ft)
    ft = fftshift(ft)
    ft_image = np.reshape(ft, shape)
    return ft_image


"Inverse Fourier Transform"
def ft(image):
    shape = image.shape
    image = image[:, :, :3].mean(axis=2)  # Convert to grayscale

    # Calculate the 2D Fourier Transform
    ft = ifftshift(image)
    ft = fft2(ft)
    ft = fftshift(ft)
    ft_image = np.reshape(ft,  shape)
    return ft_image



"High pass filter"
def high_pass_filter(ft_image, cutoff_frequency):
    # Create a copy of the image in the frequency domain
    ft_image = ft_image.copy()
    
    # Determine the center of the image
    center = np.array(ft_image.shape) // 2
    
    # Create a grid matrix for the x and y coordinates
    y, x = np.ogrid[:ft_image.shape[0], :ft_image.shape[1]]
    
    # Calculate the distance of each point to the center
    distance = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Create the mask based on the cutoff_frequency
    mask = distance >= cutoff_frequency
    
    # Apply the mask to the Fourier image
    ft_image[~mask] = 0  # Set all frequencies outside the cutoff_frequency to 0
    
    return ft_image

def high_pass_dataset(dataset, cutoff_frequency):
    images = dataset['inputs']
    targets = dataset['targets']
    data_set_high_pass = []

    for i in range(len(images)):
        image = images[i]
        ft_image = ft(image)
        low_pass_ft_image = high_pass_filter(ft_image, cutoff_frequency)
        image = ift(low_pass_ft_image)
        data_set_high_pass.append(np.abs(image))

    high_pass_data_set = { 'inputs': np.array(data_set_high_pass), 'targets': targets }
    return high_pass_data_set



"Low pass filter"
def low_pass_filter(ft_image, cutoff_frequency):
    # Create a copy of the image in the frequency domain
    ft_image = ft_image.copy()
    
    # Determine the center of the image
    center = np.array(ft_image.shape) // 2
    
    # Create a grid matrix for the x and y coordinates
    y, x = np.ogrid[:ft_image.shape[0], :ft_image.shape[1]]
    
    # Calculate the distance of each point to the center
    distance = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Create the mask based on the cutoff_frequency
    mask = distance <= cutoff_frequency
    
    # Apply the mask to the Fourier image
    ft_image[~mask] = 0  # Set all frequencies outside the cutoff_frequency to 0
    
    return ft_image

def low_pass_dataset(dataset, cutoff_frequency):
    images = dataset['inputs']
    targets = dataset['targets']
    data_set_low_pass = []

    for i in range(len(images)):
        image = images[i]
        ft_image = ft(image)
        low_pass_ft_image = low_pass_filter(ft_image, cutoff_frequency)
        image = ift(low_pass_ft_image)
        data_set_low_pass.append(np.abs(image))

    low_pass_data_set = { 'inputs': np.array(data_set_low_pass), 'targets': targets }
    return low_pass_data_set



"Band pass filter"
def band_pass_filter(ft_image, cutoff_frequency_low, cutoff_frequency_high):
    # Create a copy of the image in the frequency domain
    ft_image = ft_image.copy()
    
    # Determine the center of the image
    center = np.array(ft_image.shape) // 2
    
    # Create a grid matrix for the x and y coordinates
    y, x = np.ogrid[:ft_image.shape[0], :ft_image.shape[1]]
    
    # Calculate the distance of each point to the center
    distance = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Create the mask based on the cutoff_frequency
    mask = (distance >= cutoff_frequency_low) & (distance < cutoff_frequency_high)
    
    # Apply the mask to the Fourier image
    ft_image[~mask] = 0  # Set all frequencies outside the cutoff_frequency to 0
    
    return ft_image

def band_pass_dataset(dataset, cutoff_frequency_low, cutoff_frequency_high):
    images = dataset['inputs']
    targets = dataset['targets']
    data_set_band_pass = []

    for i in range(len(images)):
        image = images[i]
        ft_image = ft(image)
        low_pass_ft_image = band_pass_filter(ft_image, cutoff_frequency_low, cutoff_frequency_high)
        image = ift(low_pass_ft_image)
        data_set_band_pass.append(np.abs(image))

    band_pass_data_set = { 'inputs': np.array(data_set_band_pass), 'targets': targets }
    return band_pass_data_set

    




