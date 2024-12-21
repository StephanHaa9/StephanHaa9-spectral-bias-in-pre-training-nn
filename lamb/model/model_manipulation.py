
def frankensteins_model(
        body_model, 
        head_model, 
        head_location : str
    ):
    """
    This function creates a new zombie model by combining the body (params) of one model with the head (part of params) of another model.
    The Idea is to take the pre-trained body of a model and combine it with the head of a model that is for example untrained.
    The new head is needed so that the pre-trained body can be used for a new task.

    Parameters
    ----------
    body_model : Model
        The model that should be used as the body for the new model.
        --> most of the state the new model will have will be taken from this model.
    head_model : Model
        The model that should be used as the head for the new model.
        --> the head is needed to create a new model that can be used for a new task.

    Returns
    -------
    Model
        A new model that is a combination of the body of the body_model and the head of the head_model.
        --> the new model can be used for a new task. But the body is already pre-trained.
    """
    # extract the params of the body and the head
    body_params = body_model.model_state.params
    head_params = head_model.model_state.params

    # remove old pre-trained head from body and add new head --> create zombie_params
    body_params[head_location] = head_params[head_location]
    zombie_params = body_params
    # create zombie state
    zombi_state = head_model._create_train_state(params=zombie_params)
    # create zombie model
    head_model.model_state = zombi_state
    zombie_model = head_model
    return zombie_model


