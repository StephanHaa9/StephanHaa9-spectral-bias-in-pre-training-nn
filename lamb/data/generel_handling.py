
import numpy as np

def balance_data_bins(
        bin_size:int,
        dataset: dict,
    ) -> dict:
    """
    Balances the data in the bins so that each bin contains the same number of samples.

    Parameters
    ----------
    bin_size : int
        The number of samples for each bin to return. The total size will be 10 * bin_size, and each subset will be 5 * bin_size.
    dataset : dict
        The dataset to balance. Expected keys: 'inputs' and 'targets'.

    Returns
    -------
    inputs : np.ndarray
        The balanced input data.
    """

    # Convert inputs and labels to NumPy arrays for efficient processing
    inputs = np.array(dataset['inputs'])
    targets = np.array(dataset['targets'])

    # Create a boolean mask where label is 1
    mask = (targets == 1)

    # Get the indices where the label is 1
    indices = np.where(mask)[1]  # Assuming label is a 2D array (samples x classes)

    # Determine the original positions of the samples
    original_positions = np.where(mask)[0]  # Get the original indices in the input array

    mask_0 = (indices == 0)
    mask_1 = (indices == 1)
    mask_2 = (indices == 2)
    mask_3 = (indices == 3)
    mask_4 = (indices == 4)
    mask_5 = (indices == 5)
    mask_6 = (indices == 6)
    mask_7 = (indices == 7)
    mask_8 = (indices == 8)
    mask_9 = (indices == 9)

    # sorting in bins

    inputs_0 = inputs[original_positions[mask_0]][:bin_size]
    targets_0 = targets[original_positions[mask_0]][:bin_size]

    inputs_1 = inputs[original_positions[mask_1]][:bin_size]
    targets_1 = targets[original_positions[mask_1]][:bin_size]

    inputs_2 = inputs[original_positions[mask_2]][:bin_size]
    targets_2 = targets[original_positions[mask_2]][:bin_size]

    inputs_3 = inputs[original_positions[mask_3]][:bin_size]
    targets_3 = targets[original_positions[mask_3]][:bin_size]

    inputs_4 = inputs[original_positions[mask_4]][:bin_size]
    targets_4 = targets[original_positions[mask_4]][:bin_size]

    inputs_5 = inputs[original_positions[mask_5]][:bin_size]
    targets_5 = targets[original_positions[mask_5]][:bin_size]

    inputs_6 = inputs[original_positions[mask_6]][:bin_size]
    targets_6 = targets[original_positions[mask_6]][:bin_size]

    inputs_7 = inputs[original_positions[mask_7]][:bin_size]
    targets_7 = targets[original_positions[mask_7]][:bin_size]

    inputs_8 = inputs[original_positions[mask_8]][:bin_size]
    targets_8 = targets[original_positions[mask_8]][:bin_size]

    inputs_9 = inputs[original_positions[mask_9]][:bin_size]
    targets_9 = targets[original_positions[mask_9]][:bin_size]

    # Test the length of the bins with assert
    assert len(inputs_0) == bin_size, f'inputs_0: {len(inputs_0)} != {bin_size} --> increase the bin size'
    assert len(inputs_1) == bin_size, f'inputs_1: {len(inputs_1)} != {bin_size} --> increase the bin size'
    assert len(inputs_2) == bin_size, f'inputs_2: {len(inputs_2)} != {bin_size} --> increase the bin size'
    assert len(inputs_3) == bin_size, f'inputs_3: {len(inputs_3)} != {bin_size} --> increase the bin size'
    assert len(inputs_4) == bin_size, f'inputs_4: {len(inputs_4)} != {bin_size} --> increase the bin size'
    assert len(inputs_5) == bin_size, f'inputs_5: {len(inputs_5)} != {bin_size} --> increase the bin size'
    assert len(inputs_6) == bin_size, f'inputs_6: {len(inputs_6)} != {bin_size} --> increase the bin size'
    assert len(inputs_7) == bin_size, f'inputs_7: {len(inputs_7)} != {bin_size} --> increase the bin size'
    assert len(inputs_8) == bin_size, f'inputs_8: {len(inputs_8)} != {bin_size} --> increase the bin size'
    assert len(inputs_9) == bin_size, f'inputs_9: {len(inputs_9)} != {bin_size} --> increase the bin size' 

    # Combine ans random shuffle the bins again but so that thetargets and inputs are still in the same order
    inputs = np.concatenate([inputs_0, inputs_1, inputs_2, inputs_3, inputs_4, inputs_5, inputs_6, inputs_7, inputs_8, inputs_9])
    targets = np.concatenate([targets_0, targets_1, targets_2, targets_3, targets_4, targets_5, targets_6, targets_7, targets_8, targets_9])

    # Shuffle the data
    idx = np.random.permutation(len(inputs))
    inputs = inputs[idx]
    targets = targets[idx]

    balanced_dataset = {'inputs': inputs, 'targets': targets}

    return balanced_dataset


def translate_targetset(
        dataset: dict,
        targetset: list,
    ) -> dict:
    """
    Translates the targetset to a new targetset.

    Parameters
    ----------
    dataset : dict
        The dataset to translate.
    targetset : list
        --> indexes the categories of targets whitch should be translated.
        example: targetset = [5, 7, 9] --> the first three categories are translated to [0, 1, 2]
                 That means in one hot encoding 5 : [0, 0, 0, 0, 0, 1, 0, 1, 0, 0] --> [1, 0, 0] : 5
        The new targetset.
    
    Returns
    -------
    new_dataset : dict
        The new dataset with the new targetset.
    """
    # create mapping for the new targetset
    new_targetset = np.arange(len(targetset))
    mapping = {targetset[i]: new_targetset[i] for i in range(len(targetset))}
    targets = np.array(dataset['targets'])
    index_targets = np.where(targets == 1)[1]
    mask = np.isin(index_targets, targetset)
    inputs = np.array(dataset['inputs'])[mask]
    targets = np.array(dataset['targets'])[mask]
    new_targets = np.zeros((len(targets), len(targetset)))
    for i in range(len(targets)):
        new_targets[i][mapping[index_targets[i]]] = 1
    new_dataset = {'inputs': inputs, 'targets': new_targets}
    return new_dataset


def dataset_splitter(
        dataset: dict,
        targetset_0: list,
        targetset_1: list,
    ) -> dict:
    """
    Splits the dataset into two datasets based on the labels.
    Parameters
    ----------
    dataset : dict
        A dictionary containing the training and test datasets.
    targetset_0 : list
        A list of labels for the first dataset.
        --> indexes the categories of target whitch schould be in the first dataset.
        example: [0, 5, 9] --> the first three categories are in the first dataset
                 --> the fist dataset contains datapoints with the one hot encodet labels 0:[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 5:[0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 9:[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    targetset_1 : list
        A list of labels for the second dataset.
        --> indexes the categories of target whitch schould be in the second dataset.
        example: [1, 3, 7] --> the first three categories are in the second dataset
                --> the second dataset contains datapoints with the one hot encodet labels 1:[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 3:[0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 7:[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

    Returns
    -------
    dataset_0 : dict
        A dictionary containing the first dataset.
    dataset_1 : dict
        A dictionary containing the second dataset.
  
    """
    targets = np.array(dataset['targets'])
    # transform the one-hot encoded targets to the labels into integers
    index_targets = np.where(targets == 1)[1]
    # create a masks for each labelset
    mask_0 = np.isin(index_targets, targetset_0)
    mask_1 = np.isin(index_targets, targetset_1)
    # apply the masks to the inputs and targets
    inputs_0 = np.array(dataset['inputs'])[mask_0]
    targets_0 = np.array(dataset['targets'])[mask_0]
    inputs_1 = np.array(dataset['inputs'])[mask_1]
    targets_1 = np.array(dataset['targets'])[mask_1]
    # create the new datasets
    dataset_0 = {'inputs': inputs_0, 'targets': targets_0}
    dataset_1 = {'inputs': inputs_1, 'targets': targets_1}
    return dataset_0, dataset_1




def mislabel_dataset(
        dataset: dict, 
        mislabel_rate: float
    ) -> dict:
    """
    Mislabels a given dataset with a given mislabel rate.

    Parameters
    ----------
    dataset : dict
        The dataset to mislabel. Expected keys: 'inputs' and 'targets'.
    mislabel_rate : float
        The rate at which to mislabel the dataset (0 <= mislabel_rate <= 1).

    Returns
    -------
    mislabeled_dataset : dict
        The mislabeled dataset.
    """
    inputs = np.array(dataset['inputs'])
    targets = np.array(dataset['targets'])
    
    # Calculate the number of data points to mislabel
    number_of_mislabeld_datapoints = int(len(targets) * mislabel_rate)
    
    # Randomly select rows to mislabel
    indices_to_mislabel = np.random.choice(len(targets), size=number_of_mislabeld_datapoints, replace=False)
    
    for i in indices_to_mislabel:
        # Get the current label (assumes one-hot encoding)
        current_label = np.argmax(targets[i])
        
        # Select a new label that is different from the current label
        possible_labels = [label for label in range(targets.shape[1]) if label != current_label]
        new_label = np.random.choice(possible_labels)
        
        # Update the target with the new label
        targets[i] = 0
        targets[i, new_label] = 1
    
    # Create and return the mislabeled dataset
    mislabeled_dataset = {'inputs': inputs, 'targets': targets}
    return mislabeled_dataset


def copy_part_of_dataset(
        dataset,
        targetset
    ) -> dict:
    """
    Copies a part of the dataset based on the targetset.

    Parameters
    ----------
    dataset : dict
        The dataset to copy.
    targetset : list
        The targetset to copy.
        --> indexes the categories of target whitch schould be copied.
        example: targetset = [0, 5, 9] --> three categories are copied
                 --> the copied dataset contains datapoints with the one hot encodet labels 0:[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 5:[0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 9:[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    Returns
    -------
    new_dataset : dict
        The new dataset witch contains all datapoints with the to the targetset coresponing targets
    """
    # create mapping for the new targetset
    targets = np.array(dataset['targets'])
    index_targets = np.where(targets == 1)[1]
    mask = np.isin(index_targets, targetset)
    # apply the mask to the inputs and targets
    inputs = np.array(dataset['inputs'])[mask]
    targets = np.array(dataset['targets'])[mask]
    # create the new dataset
    new_dataset = {'inputs': inputs, 'targets': targets}
    return new_dataset