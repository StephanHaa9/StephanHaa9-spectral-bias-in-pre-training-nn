
def sorting_reports(
        reports,

        loss_train,
        cv_loss_train,
        accuracy_train,
        cv_accuracy_train,

        ntk_train,
        ntk_entropy_train,
        ntk_trace_train,

        loss_test,
        cv_loss_test,
        accuracy_test,
        cv_accuracy_test,

        ntk_test,
        ntk_entropy_test,
        ntk_trace_test,


        ntk_matrix_saveing_bool=False ##############################################################################################
    ):

    """Sorts the report dicts 'reports' and appends them on the lists:

        ------------------------Lists
        loss_train
        cv_loss_train
        accuracy_train
        cv_accuracy_train

        ntk_train
        ntk_entropy_train
        ntk_trace_train

        loss_test
        cv_loss_test
        accuracy_test
        cv_accuracy_test

        ntk_test
        ntk_entropy_test
        ntk_trace_test
        ------------------------

    """    
    
    number_of_report_dicts_in_reports = len(reports)
    if number_of_report_dicts_in_reports == 2:

        # Training report 
        train_report = reports[0]
        loss_train.append(train_report['loss'])
        accuracy_train.append(train_report['accuracy'])

        
        # Testing report
        test_report = reports[1]
        loss_test.append(test_report['loss'])
        accuracy_test.append(test_report['accuracy'])


    elif number_of_report_dicts_in_reports == 3:

        #Training report
        train_report = reports[0]
        loss_train.append(train_report['loss'])
        accuracy_train.append(train_report['accuracy'])
        

        # Checking the order of recorders: [cv_train_report, test_report] or [test_report, cv_train_report]
        second_report = reports[1]
        number_of_items_in_dict = len(second_report)

        if number_of_items_in_dict == 5:

            # CV train report
            cv_train_report = second_report
            cv_loss_train.append(cv_train_report['loss'])
            cv_accuracy_train.append(cv_train_report['accuracy'])
            if ntk_matrix_saveing_bool:
                ntk_train.append(cv_train_report['ntk'])
            
            ntk_entropy_train.append(cv_train_report['ntk_entropy'])
            ntk_trace_train.append(cv_train_report['ntk_trace'])
        
        
            # Testing report
            test_report = reports[2]
            loss_test.append(test_report['loss'])
            accuracy_test.append(test_report['accuracy'])

        elif number_of_items_in_dict == 2:

            # Testing report
            test_report = second_report
            loss_test.append(test_report['loss'])
            accuracy_test.append(test_report['accuracy'])

            # CV test report
            cv_test_report = reports[2]
            cv_loss_test.append(cv_test_report['loss'])
            cv_accuracy_test.append(cv_test_report['accuracy'])

            if ntk_matrix_saveing_bool:
                ntk_test.append(cv_test_report['ntk'])
        
            ntk_entropy_test.append(cv_test_report['ntk_entropy'])
            ntk_trace_test.append(cv_test_report['ntk_trace'])

        else:
            raise ValueError(f"Unexpected number of items in dict: {number_of_items_in_dict}")

    elif number_of_report_dicts_in_reports == 4:

        # Training report
        train_report = reports[0]
        loss_train.append(train_report['loss'])
        accuracy_train.append(train_report['accuracy'])

        # CV train report
        cv_train_report = reports[1]
        cv_loss_train.append(cv_train_report['loss'])
        cv_accuracy_train.append(cv_train_report['accuracy'])
        if ntk_matrix_saveing_bool:
            ntk_train.append(cv_train_report['ntk'])
        
        ntk_entropy_train.append(cv_train_report['ntk_entropy'])
        ntk_trace_train.append(cv_train_report['ntk_trace'])

        # Testing report
        test_report = reports[2]
        loss_test.append(test_report['loss'])
        accuracy_test.append(test_report['accuracy'])
        
        # CV test report
        cv_test_report = reports[3]
        cv_loss_test.append(cv_test_report['loss'])
        cv_accuracy_test.append(cv_test_report['accuracy'])
        if ntk_matrix_saveing_bool:
            ntk_test.append(cv_test_report['ntk'])
    
        ntk_entropy_test.append(cv_test_report['ntk_entropy'])
        ntk_trace_test.append(cv_test_report['ntk_trace'])

    else:
        # Unexpected number of report dicts, handle error
        raise ValueError(f"Unexpected number of reports: {number_of_report_dicts_in_reports}")
    