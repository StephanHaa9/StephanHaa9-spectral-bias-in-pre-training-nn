from papyrus.measurements import (Loss, Accuracy, NTKTrace, NTKEntropy, NTK, NTKSelfEntropy, NTKEigenvalues, LossDerivative,)
import znnl as nl 
from typing import List, Union



class RecorderBuilder:
    """
    This class is used to create recorders for tracking loss, accuracy, and NTK metrics.
    """
    @staticmethod
    def create_loss_accuracy_recorder( 
            name: str,
            model: nl.models.JaxModel,
            dataset: dict,
            recording_schedule: Union[int, List[int]],
        ):
        """
        Creates a recorder to track loss and accuracy.

        Parameters:
        ----------
        name: str
            The name of the recorder.

        dataset: dict
            The dataset to be used for the recorder.
                -The dataset should be a dictionary with keys 'inputs' and 'targets'.

        recording_schedule: Union[int, List[int]]
            The schedule for recording the measurements.
                -if int is provided, the measurements will be recorded every `recording_schedule` epochs.'
                -if List[int] is provided, the measurements will be recorded at the epochs specified in the list.

        model: nl.models.JaxModel
            The model to be used for the recorder.
        
        Returns:
        -------
        recorder: nl.training_recording.JaxRecorder
            The instantiated recorder object.
        """
        # Create a recorder to track loss and accuracy
        recorder = nl.training_recording.JaxRecorder(
            name=name,
            measurements=[
                Loss(name="loss", apply_fn=nl.loss_functions.CrossEntropyLoss()),
                Accuracy(name="accuracy", apply_fn=nl.accuracy_functions.LabelAccuracy()),
            ],
            storage_path=".",
            recording_schedule=recording_schedule,
            chunk_size=1e5
        )
        # Instantiate the recorder
        recorder.instantiate_recorder(
            data_set=dataset, 
            model=model
        )
        return recorder

    @staticmethod
    def create_entropy_trace_loss_accuracy_recorder(
            name: str,
            model: nl.models.JaxModel,
            dataset: dict,
            recording_schedule: Union[int, List[int]],
            ntk_batch_size: int
        ):
        """
        Creates a recorder to track loss, accuracy, and NTK metrics.

        Parameters:
        ----------
        name: str
            The name of the recorder.

        dataset: dict
            The dataset to be used for the recorder.
                -The dataset should be a dictionary with keys 'inputs' and 'targets'.

        recording_schedule: Union[int, List[int]]
            The schedule for recording the measurements.
                -if int is provided, the measurements will be recorded every `recording_schedule` epochs.'
                -if List[int] is provided, the measurements will be recorded at the epochs specified in the list.

        model: nl.models.JaxModel
            The model to be used for the recorder.

        ntk_batch_size: int
            The batch size to be used for the NTK computation.

        Returns:
        -------
        recorder: nl.training_recording.JaxRecorder
            The instantiated recorder object.
        """
        # Create the ntk computation object
        ntk_computation = nl.analysis.JAXNTKComputation(
            apply_fn=model.ntk_apply_fn, 
            batch_size=ntk_batch_size
        )
        # Create the recorder
        recorder = nl.training_recording.JaxRecorder(
            name=name,
            measurements=[
                Loss(name="loss", apply_fn=nl.loss_functions.CrossEntropyLoss()),
                Accuracy(name="accuracy", apply_fn=nl.accuracy_functions.LabelAccuracy()),
                NTKTrace(name="ntk_trace"),
                NTKEntropy(name="ntk_entropy"),
                NTK(name="ntk")
            ],
            storage_path=".",
            recording_schedule=recording_schedule,
            chunk_size=1e5
        )
        # Instantiate the recorder
        recorder.instantiate_recorder(
            data_set=dataset, 
            ntk_computation=ntk_computation, 
            model=model
        )
        return recorder

    @staticmethod
    def create_all_vars_recorder(
            name: str,
            model: nl.models.JaxModel,
            dataset: dict,
            recording_schedule: Union[int, List[int]],
            ntk_batch_size: int
        ):
        """
        Creates a recorder to track all relevant variables, including NTK and loss derivatives.

        Parameters:
        ----------
        name: str
            The name of the recorder.

        dataset: dict
            The dataset to be used for the recorder.
                -The dataset should be a dictionary with keys 'inputs' and 'targets'.
                
        recording_schedule: Union[int, List[int]]
            The schedule for recording the measurements.
                -if int is provided, the measurements will be recorded every `recording_schedule` epochs.'
                -if List[int] is provided, the measurements will be recorded at the epochs specified in the list.
        
        model: nl.models.JaxModel
            The model to be used for the recorder.

        ntk_batch_size: int
            The batch size to be used for the NTK computation.

        Returns:
        -------
        recorder: nl.training_recording.JaxRecorder
            The instantiated recorder object.
        """
        # Create the ntk computation object
        ntk_computation = nl.analysis.JAXNTKComputation(
            apply_fn=model.ntk_apply_fn, 
            batch_size=ntk_batch_size
        )
        # Create the loss derivative computation object
        loss_derivative_computation = nl.analysis.LossDerivative(
            loss_fn=nl.loss_functions.CrossEntropyLoss()
            )
        # Create the recorder
        recorder = nl.training_recording.JaxRecorder(
            name=name,
            measurements=[
                Loss(name="loss", apply_fn=nl.loss_functions.CrossEntropyLoss()),
                Accuracy(name="accuracy", apply_fn=nl.accuracy_functions.LabelAccuracy()),
                NTKTrace(name="ntk_trace"),
                NTKEntropy(name="ntk_entropy"),
                NTK(name="ntk"),
                NTKSelfEntropy(name="ntk_self_entropy"),
                NTKEigenvalues(name="ntk_eigenvalues"),
                LossDerivative(name="loss_derivative", apply_fn=loss_derivative_computation.calculate)
            ],
            storage_path=".",
            recording_schedule=recording_schedule,
            chunk_size=1e5  
        )
        # Instantiate the recorder
        recorder.instantiate_recorder(
            data_set=dataset, 
            ntk_computation=ntk_computation, 
            model=model
        )
        return recorder



class ModelTraining:
    """
    This class is used to run the training process with the provided recorders.
    """
    @staticmethod
    def run_model_training( 
            recorders,
            model: nl.models.JaxModel,
            epochs: int,
            train_dataset: dict,
            test_dataset: dict,
            batch_size: int,
        ):
        """
        Runs the train process using the provided recorders.
        And creates reports for the training process.
        The reports contain the recorded measurements.

        Parameters:
        ----------
        recorders: List[nl.training_recording.JaxRecorder]
            The list of recorders to be used for the training process.

        epochs: int
            The number of epochs to train the model.

        batch_size: int
            The batch size to be used for training.

        train_dataset: dict
            The training dataset to be used for training.
                -The dataset should be a dictionary with keys 'inputs' and 'targets'.
        
        test_dataset: dict
            The testing dataset to be used for evaluation.
                -The dataset should be a dictionary with keys 'inputs' and 'targets'.
        
        Returns:
        -------
        reports: List[dict]
        """
        trainer = nl.training_strategies.SimpleTraining(
            model=model,
            loss_fn=nl.loss_functions.CrossEntropyLoss(),
            recorders=recorders
        )
        trainer.train_model(
            train_ds=train_dataset, 
            test_ds=test_dataset,
            batch_size=batch_size,
            epochs=epochs
        )
        reports = [recorder.gather() for recorder in recorders]
        
        # Clear recorders to free memory
        for recorder in recorders:
            del recorder
        
        return reports






class Single_One_Model__Training_Testing__Recording:
    """
    This class is used to run a single training and testing process with multiple recorders.
    If flagt it will also record the NTK metrics during training and testing.

    Parameters:
    ----------
    model: nl.models.JaxModel
        The model to be trained and tested.
    
    epochs: int
        The number of epochs to train the model.
    
    train_dataset: dict
        The training dataset to be used for training.
            -The dataset should be a dictionary with keys 'inputs' and 'targets'.
    
    test_dataset: dict
        The testing dataset to be used for evaluation.
            -The dataset should be a dictionary with keys 'inputs' and 'targets'.
    
    ntk_dataset_size: int
        The size of the dataset to be used for NTK computation.
    
    ntk_dataset_start_index: int
        The index where the NTK datasets are pulled from the training and testing datasets.
    
    ntk_batch_size: int
        The batch size to be used for NTK computation.
    
    batch_size: int
        The batch size to be used for training and testing.
    
    recording_schedule: Union[int, List[int]]
        The schedule for recording the measurements.
            -if int is provided, the measurements will be recorded every `recording_schedule` epochs.'
            -if List[int] is provided, the measurements will be recorded at the epochs specified in the list.
    
    ntk_recording_schedule: Union[int, List[int]]
        The schedule for recording the NTK measurements.
            -if int is provided, the measurements will be recorded every `ntk_recording_schedule` epochs.'
            -if List[int] is provided, the measurements will be recorded at the epochs specified in the list.
    
    ntk_train_bool: bool
        A flag to indicate if the NTK metrics should be recorded during training.
    
    ntk_test_bool: bool
        A flag to indicate if the NTK metrics should be recorded during testing.
    """

    def __init__(self,
                 
            model: nl.models.JaxModel, 

            epochs: int,

            train_dataset: dict,
            test_dataset: dict,

            ntk_dataset_size: int,
            ntk_dataset_start_index: int,

            ntk_batch_size: int,
            batch_size: int,

            recording_schedule: Union[int, List[int]],
            ntk_recording_schedule: Union[int, List[int]],

            ntk_train_bool: bool,
            ntk_test_bool: bool,

        ):
        """
        """

        self.model = model 

        self.epochs = epochs

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.ntk_dataset_size = ntk_dataset_size
        self.ntk_dataset_start_index = ntk_dataset_start_index

        self.ntk_batch_size = ntk_batch_size
        self.batch_size = batch_size

        self.recording_schedule = recording_schedule
        self.ntk_recording_schedule = ntk_recording_schedule

        self.ntk_train_bool = ntk_train_bool
        self.ntk_test_bool = ntk_test_bool

        # Building the ntk data sets:
        # Training 
        ntk_inputs_train_dataset = self.train_dataset['inputs'][ntk_dataset_start_index:ntk_dataset_start_index+ntk_dataset_size]
        ntk_targets_train_dataset = self.train_dataset['targets'][ntk_dataset_start_index:ntk_dataset_start_index+ntk_dataset_size]
        self.ntk_train_dataset = {'inputs':ntk_inputs_train_dataset, 'targets':ntk_targets_train_dataset}
        # Testing 
        ntk_inputs_test_dataset = self.test_dataset['inputs'][ntk_dataset_start_index:ntk_dataset_start_index+ntk_dataset_size]
        ntk_targets_test_dataset = self.test_dataset['targets'][ntk_dataset_start_index:ntk_dataset_start_index+ntk_dataset_size]
        self.ntk_test_dataset = {'inputs':ntk_inputs_test_dataset, 'targets':ntk_targets_test_dataset}
        

    
    def run(self
        ):
        """
        Run the train with configured recorders.
        """

        # Create list for recorders and append the train recorder
        recorders = [
            RecorderBuilder().create_loss_accuracy_recorder(
                                    name='train_recorder',
                                    model=self.model,
                                    dataset=self.train_dataset,
                                    recording_schedule=self.recording_schedule
                                )
        ]


        # Add a conditional recorder for recording the NTK metrics during training
        if self.ntk_train_bool == True:
            recorders.append(RecorderBuilder().create_entropy_trace_loss_accuracy_recorder(
                                                    name='ntk_entropy_trace_train_recorder',
                                                    model=self.model,
                                                    dataset=self.ntk_train_dataset,
                                                    recording_schedule=self.ntk_recording_schedule,
                                                    ntk_batch_size=self.ntk_batch_size
                                                )
                            )


        # Append the test recorder
        recorders.append(RecorderBuilder().create_loss_accuracy_recorder(
                                                    name='test_recorder',
                                                    model=self.model,
                                                    dataset=self.test_dataset,
                                                    recording_schedule=self.recording_schedule
                                                )
                        )


        # Add a conditional recorder for recording the NTK metrics during testing
        if self.ntk_test_bool == True:
            recorders.append(RecorderBuilder().create_entropy_trace_loss_accuracy_recorder(
                                                    name='ntk_entropy_trace_test_recorder',
                                                    model=self.model,
                                                    dataset=self.ntk_test_dataset,
                                                    recording_schedule=self.ntk_recording_schedule,
                                                    ntk_batch_size=self.ntk_batch_size
                                                )
                        )


        # Run the train with the configured recorders
        reports = ModelTraining.run_model_training(
            model=self.model,
            recorders=recorders,
            epochs=self.epochs,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            batch_size=self.batch_size
        )

        # Clear memory
        for recorder in recorders:
            del recorder

        return reports
