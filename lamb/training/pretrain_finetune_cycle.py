import sys
import os
sys.path.append(os.path.join(os.getcwd(), '/tikhome/shaag/bachelor_arbeit/code/NaLamb/lamb'))
from model.model_manipulation  import frankensteins_model
from data.dict_handling import sorting_reports
sys.path.append(os.path.join(os.getcwd(), '/tikhome/shaag/bachelor_arbeit/code/NaLamb/lamb/training'))
from recorder_training import Single_One_Model__Training_Testing__Recording
import time
from typing import List, Union



class Single_PreTrain_And_FineTune_Cycle:
    """
    This class is used to track the behavior tow models. One model is pre-trained and then fine-tuned on a new task.
    The other model is just trained on the new task. The created data is saved in the given dicts. 

    -- What data is created and saved in the specific parts of the cycle: 
    ---- Pre-Training:
    ------ 1. loss_train, accuracy_train
    ------ 2. loss_test, accuracy_test
    ---- Fine-Tuning:
    ------ 1. ntk_entropy_train, ntk_trace_train, ntk, loss_train, accuracy_train
    ------ 2. ntk_entropy_test, ntk_trace_test, ntk, loss_test, accuracy_test
    ---- Just Training:
    ------ 1. ntk_entropy_train, ntk_trace_train, ntk, loss_train, accuracy_train
    ------ 2. ntk_entropy_test, ntk_trace_test, ntk, loss_test, accuracy_test

    
    Parameters
    ----------
    pretrain_model : Model
        The models architectur schould fit the first dataset (pretrain_train_ds) and the first task.
        --> The model gives his pre-trained body to the pretrain_finetune_model.
    
    just_train_on_finetune_ds_model : Model
        The models architectur schould fit the second dataset (finetune_train_ds) and the second task.
        --> The model gives his head to the pretrain_finetune_model.
    
    model_layer_key_to_update : str
        The key of the layer that should be updated in the pretrain_finetune_model.
        --> The key is used to define the head of the pretrain_finetune_model.
    
    epochs : int
        The number of epochs that should be used for the training.

    pretrain_train_ds : dict
        The training dataset for the pre-training

    pretrain_test_ds : dict
        The test dataset for the pre-training

    finetune_train_ds : dict
        The training dataset for the fine-tuning

    finetune_test_ds : dict
        The test dataset for the fine-tuning

    ntk_dataset_size : int
        The size of the dataset that should be used for the NTK computation.

    ntk_dataset_start_index : int
        The start index of the dataset that should be used for the NTK computation.

    ntk_batch_size : int
        The batch size that should be used for the NTK computation.

    batch_size : int
        The batch size that should be used for the training.

    recording_schedule : Union[int, List[int]]
        The schedule that should be used for the recording of the training data.

    ntk_recording_schedule : Union[int, List[int]]
        The schedule that should be used for the recording of the NTK data.

    pretrain_data_saving_dict : dict
        The dict that should be used to save the data created during the pre-training.

    finetune_data_saving_dict : dict
        The dict that should be used to save the data created during the fine-tuning.

    just_train_data_saving_dict : dict
        The dict that should be used to save the data created during the just training.

        


    Returns
    -------
    pretrain_data_saving_dict : dict, finetune_data_saving_dict : dict, just_train_data_saving_dict : dict
    """
    def __init__(self,
                    
            pretrain_model,
            just_train_on_finetune_ds_model,
            model_layer_key_to_update: str,

            epochs: int,

            pretrain_train_ds: dict,
            pretrain_test_ds: dict,
            finetune_train_ds: dict,
            finetune_test_ds: dict,

            ntk_dataset_size: int,
            ntk_dataset_start_index: int,

            ntk_batch_size: int,
            batch_size: int,

            recording_schedule: Union[int, List[int]],
            ntk_recording_schedule: Union[int, List[int]],

            pretrain_data_saving_dict: dict,
            finetune_data_saving_dict: dict,
            just_train_data_saving_dict: dict,

    
                
        ):

        self.pretrain_model = pretrain_model
        self.just_train_on_finetune_ds_model = just_train_on_finetune_ds_model
        self.model_layer_key_to_update = model_layer_key_to_update

        self.epochs = epochs

        self.pretrain_train_ds = pretrain_train_ds
        self.pretrain_test_ds = pretrain_test_ds
        self.finetune_train_ds = finetune_train_ds
        self.finetune_test_ds = finetune_test_ds

        self.ntk_dataset_size = ntk_dataset_size
        self.ntk_dataset_start_index = ntk_dataset_start_index

        self.ntk_batch_size = ntk_batch_size
        self.batch_size = batch_size

        self.recording_schedule = recording_schedule
        self.ntk_recording_schedule = ntk_recording_schedule

        self.pretrain_data_saving_dict = pretrain_data_saving_dict
        self.finetune_data_saving_dict = finetune_data_saving_dict
        self.just_train_data_saving_dict = just_train_data_saving_dict


    def run(self
        ):
        print(f'###########################################################################################################')
        print(f'############################# START PRE_TRAINING AND FINETUNING TRAINER ###################################')
        print(f'###########################################################################################################')
        print(f'PRE_TRAINING')
        "-------------------------------------------------------------------------------------PRE_TRAINING-----------------------------------------------------------------------------------------"
        pretrining_reports = Single_One_Model__Training_Testing__Recording(
            model = self.pretrain_model,

            epochs = self.epochs,

            train_dataset = self.pretrain_train_ds,
            test_dataset = self.pretrain_test_ds,

            ntk_dataset_size=self.ntk_dataset_size,
            ntk_dataset_start_index=self.ntk_dataset_start_index,

            ntk_batch_size=self.ntk_batch_size,
            batch_size=self.batch_size,

            recording_schedule=self.recording_schedule,
            ntk_recording_schedule=self.ntk_recording_schedule,

            ntk_train_bool=False,
            ntk_test_bool=False,
        )
        sorting_reports(
            reports=pretrining_reports,

            loss_train=self.pretrain_data_saving_dict['loss_pretrain_train'],
            cv_loss_train=self.pretrain_data_saving_dict['cv_loss_pretrain_train'],
            accuracy_train=self.pretrain_data_saving_dict['acc_pretrain_train'],
            cv_accuracy_train=self.pretrain_data_saving_dict['cv_acc_pretrain_train'],
            ntk_train=self.pretrain_data_saving_dict['ntk_pretrain_train'],
            ntk_entropy_train=self.pretrain_data_saving_dict['ntk_entropy_pretrain_train'],
            ntk_trace_train=self.pretrain_data_saving_dict['ntk_trace_pretrain_train'],
            loss_test=self.pretrain_data_saving_dict['loss_pretrain_test'],
            cv_loss_test=self.pretrain_data_saving_dict['cv_loss_pretrain_test'],
            accuracy_test=self.pretrain_data_saving_dict['acc_pretrain_test'],
            cv_accuracy_test=self.pretrain_data_saving_dict['cv_acc_pretrain_test'],
            ntk_test=self.pretrain_data_saving_dict['ntk_pretrain_test'],
            ntk_entropy_test=self.pretrain_data_saving_dict['ntk_entropy_pretrain_test'],
            ntk_trace_test=self.pretrain_data_saving_dict['ntk_trace_pretrain_test'],
        )
    
        print(f'CREATING THE PRETRAIN_FINETUNE_MODEL')
        "--------------------------------------------------------------------------------------------CREATING THE PRETRAIN_FINETUNE_MODEL---------------------------------------------------------------"
        self.pretrain_finetune_model = frankensteins_model(body_model=self.pretrain_model, head_model=self.just_train_on_finetune_ds_model, head_location=self.model_layer_key_to_update)
    
        print(f'FINETUNING')
        "--------------------------------------------------------------------------------------------FINETUNING-----------------------------------------------------------------------------------------"
        finetune_reports= Single_One_Model__Training_Testing__Recording(
            model=self.pretrain_finetune_model,

            epochs=self.epochs,

            train_dataset=self.finetune_train_ds,
            test_dataset=self.finetune_test_ds,

            ntk_dataset_size=self.ntk_dataset_size,
            ntk_dataset_start_index=self.ntk_dataset_start_index,

            ntk_batch_size=self.ntk_batch_size,
            batch_size=self.batch_size,

            recording_schedule=self.recording_schedule,
            ntk_recording_schedule=self.ntk_recording_schedule,

            ntk_train_bool=True,
            ntk_test_bool=True,
        )
        sorting_reports(
            reports=finetune_reports,

            loss_train=self.finetune_data_saving_dict['loss_finetune_train'],
            cv_loss_train=self.finetune_data_saving_dict['cv_loss_finetune_train'],
            accuracy_train=self.finetune_data_saving_dict['acc_finetune_train'],
            cv_accuracy_train=self.finetune_data_saving_dict['cv_acc_finetune_train'],
            ntk_train=self.finetune_data_saving_dict['ntk_finetune_train'],
            ntk_entropy_train=self.finetune_data_saving_dict['ntk_entropy_finetune_train'],
            ntk_trace_train=self.finetune_data_saving_dict['ntk_trace_finetune_train'],
            loss_test=self.finetune_data_saving_dict['loss_finetune_test'],
            cv_loss_test=self.finetune_data_saving_dict['cv_loss_finetune_test'],
            accuracy_test=self.finetune_data_saving_dict['acc_finetune_test'],
            cv_accuracy_test=self.finetune_data_saving_dict['cv_acc_finetune_test'],
            ntk_test=self.finetune_data_saving_dict['ntk_finetune_test'],
            ntk_entropy_test=self.finetune_data_saving_dict['ntk_entropy_finetune_test'],
            ntk_trace_test=self.finetune_data_saving_dict['ntk_trace_finetune_test'],
        )
        
        print(f'JUST TRAINING')
        "----------------------------------------------------------------------------------------------JUST TRAINING-----------------------------------------------------------------------------------------"
        just_train_reports = Single_One_Model__Training_Testing__Recording(
            model=self.just_train_on_finetune_ds_model,

            epochs=self.epochs,

            train_dataset=self.finetune_train_ds,
            test_dataset=self.finetune_test_ds,

            ntk_dataset_size=self.ntk_dataset_size,
            ntk_dataset_start_index=self.ntk_dataset_start_index,

            ntk_batch_size=self.ntk_batch_size,
            batch_size=self.batch_size,

            recording_schedule=self.recording_schedule,
            ntk_recording_schedule=self.ntk_recording_schedule,

            ntk_train_bool=True,
            ntk_test_bool=True,
        )

        sorting_reports(
            reports=just_train_reports,

            loss_train=self.just_train_data_saving_dict['loss_train_train'],
            cv_loss_train=self.just_train_data_saving_dict['cv_loss_train_train'],
            accuracy_train=self.just_train_data_saving_dict['acc_train_train'],
            cv_accuracy_train=self.just_train_data_saving_dict['cv_acc_train_train'],
            ntk_train=self.just_train_data_saving_dict['ntk_train_train'],
            ntk_entropy_train=self.just_train_data_saving_dict['ntk_entropy_train_train'],
            ntk_trace_train=self.just_train_data_saving_dict['ntk_trace_train_train'],
            loss_test=self.just_train_data_saving_dict['loss_train_test'],
            cv_loss_test=self.just_train_data_saving_dict['cv_loss_train_test'],
            accuracy_test=self.just_train_data_saving_dict['acc_train_test'],
            cv_accuracy_test=self.just_train_data_saving_dict['cv_acc_train_test'],
            ntk_test=self.just_train_data_saving_dict['ntk_train_test'],
            ntk_entropy_test=self.just_train_data_saving_dict['ntk_entropy_train_test'],
            ntk_trace_test=self.just_train_data_saving_dict['ntk_trace_train_test'],
        )
        
        "--------------------------------------------------------------------------------------------RETURNING THE SAVING DICTS-------------------------------------------------------------------------"
        return self.pretrain_data_saving_dict, self.finetune_data_saving_dict, self.just_train_data_saving_dict
        
    













































































































# class PreTrain_And_FineTune_Trainer:
#     """
#     """
#     def __init__(self, 
                 
            # pretrain_model,
            # finetune_model,
            # model_layer_key_to_update: str,

            # pretrain_train_ds: dict,
            # pretrain_test_ds: dict,
            # finetune_train_ds: dict,
            # finetune_test_ds: dict,

            # pretrain_dict: dict,
            # finetune_dict: dict,
            # train_dict: dict,

            # epochs: int,
            # recording_schedule: Union[int, List[int]], 
            # batch_size: int, 

            # ntk_entropy_trace_recording_schedule: Union[int, List[int]],
            # ntk_entropy_trace_batch_size: int, 
            # ntk_entropy_trace_data_set_size: int,
            # ntk_entropy_trace_data_set_start_index: int,
            # ntk_entropy_trace_matrix_saveing_bool: bool,
 
        # ):
        

        # self.pretrain_model = pretrain_model
        # self.finetune_model = finetune_model
        # self.model_layer_key_to_update = model_layer_key_to_update

        # self.finetune_train_ds = finetune_train_ds
        # self.pretrain_train_ds = pretrain_train_ds
        # self.finetune_test_ds = finetune_test_ds
        # self.pretrain_test_ds = pretrain_test_ds

        # self.pretrain_dict = pretrain_dict
        # self.finetune_dict = finetune_dict
        # self.train_dict = train_dict

        # self.epochs = epochs
        # self.recording_schedule = recording_schedule
        # self.batch_size = batch_size

        # self.ntk_entropy_trace_recording_schedule = ntk_entropy_trace_recording_schedule
        # self.ntk_entropy_trace_batch_size = ntk_entropy_trace_batch_size
        # self.ntk_entropy_trace_data_set_size = ntk_entropy_trace_data_set_size
        # self.ntk_entropy_trace_data_set_start_index = ntk_entropy_trace_data_set_start_index
        # self.ntk_matrix_saveing_bool = ntk_entropy_trace_matrix_saveing_bool



























































    # def _run_basic_train_block(self, 
    #         model, 
    #         train_data_set, 
    #         test_data_set, 
    #         ntk_entropy_trace_train_bool,
    #         ntk_entropy_trace_test_bool,
    #     ):
    #     """
    #     Run a basic train process with optional cross-validation.
    #     """
    #     basic_train_block = Single_One_Model__Training_Testing__Recording(
    #         model=model,

    #         train_dataset=train_data_set,
    #         test_dataset=test_data_set,

    #         recording_schedule=self.recording_schedule,
    #         batch_size=self.batch_size,
    #         epochs=self.epochs,

    #         ntk_entropy_trace_train_bool=ntk_entropy_trace_train_bool,
    #         ntk_entropy_trace_test_bool=ntk_entropy_trace_test_bool,

    #         ntk_recording_schedule=self.ntk_entropy_trace_recording_schedule,
    #         ntk_batch_size=self.ntk_entropy_trace_batch_size,
    #         ntk_dataset_size=self.ntk_entropy_trace_data_set_size,
    #         ntk_dataset_start_index=self.ntk_entropy_trace_data_set_start_index,
    #     )
    #     reports = basic_train_block.run()
    #     return reports







    
    # def _run_single_train(self, 
    #     ):
    #     # Pretrain pretrain_finetune_model on even numbers
    #     print(f'###########################################################################################################')
    #     print(f'############################# START PRE_TRAINING AND FINETUNING TRAINER ###################################')
    #     print(f'###########################################################################################################')
    #     print()
    #     print()
    #     print(f'***********************************************************************************************************')
    #     print(f'----PRE_TRAINING:------------------------------------------------------------------------------------------')
    #     print(f'-----------------------------------------------------------------------------------------------------------')
    #     print(f'-------------------MODEL:      pretrain_finetune_model-----------------------------------------------------')
    #     print(f'-------------------DATASET:    pretrain_train_ds-----------------------------------------------------------')
    #     print(f'-------------------RECORDING:------------------------------------------------------------------------------')
    #     print(f'-------------------------------1. loss_train, accuracy_train-----------------------------------------------')
    #     print(f'-------------------------------2. loss_test, accuracy_test-------------------------------------------------')
    #     print(f'-----------------------------------------------------------------------------------------------------------')
    #     start_time = time.time()
    #     pretrining_report = self._run_basic_train_block(
    #         model=self.pretrain_finetune_model,
    #         train_data_set=self.pretrain_train_ds,
    #         test_data_set=self.pretrain_test_ds,
    #         ntk_entropy_trace_train_bool=False,
    #         ntk_entropy_trace_test_bool=False
    #     )
    #     end_time = time.time()









        
    #     print(f'-----------------------------------------------------------------------------------------------------------')
    #     print(f"---------------TIME:       {end_time - start_time}--------------------------------------------------------------")
    #     print("-------------------SORTING of the pretraining_reports------------------------------------------------------")
    #     sorting_reports(
    #         reports=pretrining_report,
    #         loss_train=self.pretrain_dict['loss_pretrain_train'],
    #         cv_loss_train=self.pretrain_dict['cv_loss_pretrain_train'],
    #         accuracy_train=self.pretrain_dict['acc_pretrain_train'],
    #         cv_accuracy_train=self.pretrain_dict['cv_acc_pretrain_train'],
    #         ntk_train=self.pretrain_dict['ntk_pretrain_train'],
    #         ntk_entropy_train=self.pretrain_dict['ntk_entropy_pretrain_train'],
    #         ntk_trace_train=self.pretrain_dict['ntk_trace_pretrain_train'],
    #         loss_test=self.pretrain_dict['loss_pretrain_test'],
    #         cv_loss_test=self.pretrain_dict['cv_loss_pretrain_test'],
    #         accuracy_test=self.pretrain_dict['acc_pretrain_test'],
    #         cv_accuracy_test=self.pretrain_dict['cv_acc_pretrain_test'],
    #         ntk_test=self.pretrain_dict['ntk_pretrain_test'],
    #         ntk_entropy_test=self.pretrain_dict['ntk_entropy_pretrain_test'],
    #         ntk_trace_test=self.pretrain_dict['ntk_trace_pretrain_test'],
    #     )
    #     print(f'***********************************************************************************************************')
    #     print()


    #     # Updating the last layer of the not_pretrain_finetune_model with the last layer of the pretrain_finetune_model
    #     print(f'***********************************************************************************************************')
    #     print(f'----UPDATING LAYER-----------------------------------------------------------------------------------------')  
    #     self.pretrain_finetune_model = frankensteins_model(body_model=self.pretrain_model, head_model=self.finetune_model, head_location=self.model_layer_key_to_update)
    #     print(f'***********************************************************************************************************')
    #     print()




    #     # Training pretrain_finetune_model on odd numbers tracking cv
        
    #     print(f'***********************************************************************************************************')
    #     print(f'----FINETUNING:--------------------------------------------------------------------------------------------')
    #     print(f'-----------------------------------------------------------------------------------------------------------')
    #     print(f'-----------------MODEL:      pretrain_finetune_model-------------------------------------------------------')
    #     print(f'-----------------DATASET:    finetune_train_ds-------------------------------------------------------------')
    #     print(f'-----------------RECORDING:--------------------------------------------------------------------------------')
    #     print(f'----------------------------1. ntk_entropy_train, ntk_trace_train, ntk, loss_train, accuracy_train---------')
    #     print(f'----------------------------2. ntk_entropy_test, ntk_trace_test, ntk, loss_test, accuracy_test-------------')
    #     print(f'-----------------------------------------------------------------------------------------------------------')
    #     start_time = time.time()
    #     finetune_report= self._run_basic_train_block(
    #         model=self.pretrain_finetune_model, 
    #         train_data_set=self.finetune_train_ds, 
    #         test_data_set=self.finetune_test_ds, 
    #         ntk_entropy_trace_test_bool=True,
    #         ntk_entropy_trace_train_bool=True
    #     )
    #     end_time = time.time()
    #     print(f'-----------------------------------------------------------------------------------------------------------')
    #     print(f"------------TIME:       {end_time - start_time}--------------------------------------------------------------")
    #     print("-----------------SORTING of the finetune_reports--------------------------------------------------------")
        
    #     sorting_reports(
    #         reports=finetune_report,
    #         loss_train=self.finetune_dict['loss_finetune_train'],
    #         cv_loss_train=self.finetune_dict['cv_loss_finetune_train'],
    #         accuracy_train=self.finetune_dict['acc_finetune_train'],
    #         cv_accuracy_train=self.finetune_dict['cv_acc_finetune_train'],
    #         ntk_train=self.finetune_dict['ntk_finetune_train'],
    #         ntk_entropy_train=self.finetune_dict['ntk_entropy_finetune_train'],
    #         ntk_trace_train=self.finetune_dict['ntk_trace_finetune_train'],
    #         loss_test=self.finetune_dict['loss_finetune_test'],
    #         cv_loss_test=self.finetune_dict['cv_loss_finetune_test'],
    #         accuracy_test=self.finetune_dict['acc_finetune_test'],
    #         cv_accuracy_test=self.finetune_dict['cv_acc_finetune_test'],
    #         ntk_test=self.finetune_dict['ntk_finetune_test'],
    #         ntk_entropy_test=self.finetune_dict['ntk_entropy_finetune_test'],
    #         ntk_trace_test=self.finetune_dict['ntk_trace_finetune_test'],
    #     )
    #     print(f'***********************************************************************************************************')
    #     print()
        


    #     # Training just_train_model on odd numbers tracking cv
    #     print(f'***********************************************************************************************************')
    #     print(f'----TRAINING:----------------------------------------------------------------------------------------------')
    #     print(f'-----------------------------------------------------------------------------------------------------------')
    #     print(f'---------------MODEL:      just_train_model----------------------------------------------------------------')
    #     print(f'---------------DATASET:    finetune_train_ds---------------------------------------------------------------')
    #     print(f'---------------RECORDING:----------------------------------------------------------------------------------')
    #     print(f'---------------------------1. ntk_entropy_train, ntk_trace_train, ntk, loss_train, accuracy_train----------')
    #     print(f'---------------------------2. ntk_entropy_test, ntk_trace_test, ntk, loss_test, accuracy_test--------------')
    #     print(f'-----------------------------------------------------------------------------------------------------------')
    #     start_time = time.time()
    #     train_report= self._run_basic_train_block(
    #         model=self.just_train_model, 
    #         train_data_set=self.finetune_train_ds, 
    #         test_data_set=self.finetune_test_ds, 
    #         ntk_entropy_trace_test_bool=True,
    #         ntk_entropy_trace_train_bool=True
    #     )
    #     end_time = time.time()
    #     print(f'---------------RECORDING:----------------------------------------------------------------------------------')
    #     print(f'-----------------------------------------------------------------------------------------------------------')
    #     print(f"------------TIME:       {end_time - start_time}--------------------------------------------------------------")
    #     print(f"------------SORTING of the train_reports------------------------------------------------------------------")
    #     sorting_reports(
    #         reports=train_report,
    #         loss_train=self.train_dict['loss_train_train'],
    #         cv_loss_train=self.train_dict['cv_loss_train_train'],
    #         accuracy_train=self.train_dict['acc_train_train'],
    #         cv_accuracy_train=self.train_dict['cv_acc_train_train'],
    #         ntk_train=self.train_dict['ntk_train_train'],
    #         ntk_entropy_train=self.train_dict['ntk_entropy_train_train'],
    #         ntk_trace_train=self.train_dict['ntk_trace_train_train'],
    #         loss_test=self.train_dict['loss_train_test'],
    #         cv_loss_test=self.train_dict['cv_loss_train_test'],
    #         accuracy_test=self.train_dict['acc_train_test'],
    #         cv_accuracy_test=self.train_dict['cv_acc_train_test'],
    #         ntk_test=self.train_dict['ntk_train_test'],
    #         ntk_entropy_test=self.train_dict['ntk_entropy_train_test'],
    #         ntk_trace_test=self.train_dict['ntk_trace_train_test'],
    #     )
    #     print(f'***********************************************************************************************************')
    #     print()
    #     print()
    #     print(f'###########################################################################################################')
    #     print(f'################## ONE PRE_TRAINING AND FINE_TUNING ENSEMBLE IS CREATED ###################################')
    #     print(f'###########################################################################################################')
    #     print()

    #     return self.pretrain_dict, self.finetune_dict, self.train_dict
        