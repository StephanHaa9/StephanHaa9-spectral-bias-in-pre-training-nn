
import znnl as nl 
import os 
import sys
from generel_handling import balance_data_bins
from generel_handling import translate_targetset
from generel_handling import dataset_splitter
sys.path.append(os.path.join(os.getcwd(), '/tikhome/shaag/bachelor_arbeit/code/ZnNL/znnl/data'))
from fashion_mnist import FashionMNISTGenerator


def outof_distribution_dataset_creator(
        bin_size: int,
        fluctuation_puffer: float,
        targetset_0: list[int],
        targetset_1: list[int],
        generator_flag: str,
        num_classes: int = 10,
    ):
    """
    Create two out-of-distribution datasets with the MNIST or FashionMNIST generator.
    The datasets are balanced and split into two datasets.
    The targets are translated to a lower dimensional one hot encoding.

    Parameters:
    -----------
    bin_size: int
        The size of the bins in the dataset.
        --> 10 * bin_size = dataset_size
    fluctuation_puffer: float
        The factor to increase the dataset size, to puffer the fluctuation. 
    targetset_0: list[int]
        The targetset for the first dataset.
         --> indexes the categories of targets whitch should be in the dataset_0 
         example: [0, 2, 4, 6, 8] --> the one hot encodet labels [1, 0, 0, 0, 0, 0, 0, 0, 0 ,0], [0, 0, 1, 0, 0, 0, 0, 0, 0 ,0], ...
                                      are in the dataset_0
    targetset_1: list[int]
        The targetset for the second dataset.
         --> indexes the categories of targets whitch should be in the dataset_1
            example: [1, 3, 5, 7, 9] --> the one hot encodet labels [0, 1, 0, 0, 0, 0, 0, 0, 0 ,0], [0, 0, 0, 1, 0, 0, 0, 0, 0 ,0], ...
                                        are in the dataset_1
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
    train_ds = balance_data_bins(
                bin_size = bin_size,
                dataset = train_ds,
            )
    test_ds = balance_data_bins(
                bin_size = bin_size,
                dataset = test_ds,
            )
        
    # split the dataset into two ood datasets
    train_ds_0, train_ds_1 = dataset_splitter(
                                dataset = train_ds,
                                targetset_0 = targetset_0,
                                targetset_1 = targetset_1,
                            )
    test_ds_0, test_ds_1 = dataset_splitter(
                                dataset = test_ds,
                                targetset_0 = targetset_0,
                                targetset_1 = targetset_1,
                            )

    # translate the targetset to a new targetset
    train_ds_0 = translate_targetset(
                    dataset = train_ds_0,
                    targetset = targetset_0,
                )
    train_ds_1 = translate_targetset(
                    dataset = train_ds_1,
                    targetset = targetset_1,
                )
    test_ds_0 = translate_targetset(
                    dataset = test_ds_0,
                    targetset = targetset_0,
                )
    test_ds_1 = translate_targetset(
                    dataset = test_ds_1,
                    targetset = targetset_1,
                )
    # create the ood datasets
    dataset_0 = {'train': train_ds_0, 'test': test_ds_0}
    dataset_1 = {'train': train_ds_1, 'test': test_ds_1}
    ood_datasets = {'set_0': dataset_0, 'set_1': dataset_1}
    return ood_datasets