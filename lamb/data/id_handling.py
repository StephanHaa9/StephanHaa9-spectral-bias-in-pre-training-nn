
import znnl as nl 
import os 
import sys

from generel_handling import balance_data_bins
from generel_handling import translate_targetset
from generel_handling import copy_part_of_dataset
from generel_handling import mislabel_dataset
sys.path.append(os.path.join(os.getcwd(), '/tikhome/shaag/bachelor_arbeit/code/ZnNL/znnl/data'))
from fashion_mnist import FashionMNISTGenerator



def in_distribution_dataset_creator(
        bin_size: int,
        fluctuation_puffer: float,
        targetset_1: list[int],
        mislabel_rate: float,
        generator_flag: str,
        num_classes: int = 10,
    ):
    """
    Create two in-distribution datasets with the MNIST or FashionMNIST generator.
    The datasets are balanced and split into two datasets.
    The targets are translated to a lower dimensional one hot encoding.

    Parameters:
    -----------
    bin_size: int
        The size of the bins in the dataset.
        --> 10 * bin_size = dataset_size
    fluctuation_puffer: float
        The factor to increase the dataset size, to puffer the fluctuation.
    targetset_1: list[int]
        The targetset_1 for the first dataset.
         --> indexes the categories of targets whitch should be in the dataset_1
         example: [0, 2, 4, 6, 8] --> the one hot encodet labels [1, 0, 0, 0, 0, 0, 0, 0, 0 ,0], [0, 0, 1, 0, 0, 0, 0, 0, 0 ,0], ...
                                      are in the dataset_1
    mislabel_rate: float
        The rate of mislabeling the dataset.
        --> 0.0 <= mislabel_rate <= 1.0
    generator_flag: str
        The flag to choose the generator.
        --> 'mnist' or 'fashion_mnist'
    num_classes: int
        The number of classes in the dataset.
        --> default: 10
    """
    # create raw dataset
    dataset_size = fluctuation_puffer * (bin_size * num_classes)

    if generator_flag == 'mnist':
        generator = nl.data.MNISTGenerator(
                        ds_size = dataset_size,
                        one_hot_encoding = True,
                    )
    elif generator_flag == 'fashion_mnist':
        generator = FashionMNISTGenerator(
                        ds_size = dataset_size,
                        one_hot_encoding = True,
                    )
    else:
        raise ValueError('generator_flag must be either mnist or fashion_mnist')
    
    train_ds = generator.train_ds
    test_ds = generator.test_ds

    # balance the dataset
    train_ds_0 = balance_data_bins(
                    bin_size = bin_size,
                    dataset = train_ds,
                )
    test_ds_0 = balance_data_bins(
                    bin_size = bin_size,
                    dataset = test_ds,
                )
    
    # copy the targetset_1 from the dataset
    train_ds_1 = copy_part_of_dataset(
                    dataset = train_ds_0,
                    targetset = targetset_1,
                )
    test_ds_1 = copy_part_of_dataset(
                    dataset = test_ds_0,
                    targetset = targetset_1,
                )
    
    # mislabel the dataset
    train_ds_0 = mislabel_dataset(
                    dataset = train_ds_0,
                    mislabel_rate = mislabel_rate,
                )
    test_ds_0 = mislabel_dataset(
                    dataset = test_ds_0,
                    mislabel_rate = mislabel_rate,
                )
    
    # translates the target vectors into a lower dimentional target vectors 
    train_ds_1 = translate_targetset(
                    dataset = train_ds_1,
                    targetset = targetset_1,
                )
    test_ds_1 = translate_targetset(
                    dataset = test_ds_1,
                    targetset = targetset_1,
                )

    # create the final datasets
    dataset_0 = {'train': train_ds_0, 'test': test_ds_0}
    dataset_1 = {'train': train_ds_1, 'test': test_ds_1}
    id_datasets = {'set_0': dataset_0, 'set_1': dataset_1}
    return id_datasets