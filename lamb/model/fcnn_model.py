from flax import linen as nn 
import znnl as nl
import optax

class FCNN_Arcitecture(nn.Module): 
    """ Defines the architecture of the model 
    
    """
    width: int =  128 
    number_features_output: int =  10

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1)) # Flatten the 28 x 28 input into a 784 vector 
        x = nn.Dense(features=self.width)(x) # Creates the first hidden layer as a dens layer with 128 features 
        x = nn.relu(x) # Applies the ReLu activation function to the nodes of the first hidden layer 
        x = nn.Dense(features=self.width)(x) # Second hidden layer 
        x = nn.relu(x) # Relu to the second hidden layer 
        x = nn.Dense(features=self.number_features_output)(x) # Creates the output layer 

        return x
    


def create_n_fcnn_models(
        number_of_models, 
        learning_rate_models, 
        seed_models, 
        width_layers_models=128,
        number_features_output=10,  
        input_shape_models=(1, 28, 28, 1)
    ):
    """
    Create n models with the same architecture.

    Parameters
    ----------
    number_of_models : int
            Number of models to be created.
    learning_rate_models : float
            Learning rate for the optimizer.
    seed_models : int
            Seed for the random number generator.
    width_layers_models : int
            Width of the hidden layers.
    num_features_output_models : int
            Number of features in the output layer.
    input_shape_models : tuple  
            Shape of the input.


    Returns
    -------
    models : list
            List of models.
    """

    print('----------------FCNN MODELS--------------------')
    print('PARAMS:')
    print('--------')
    print(f'number_of_models : {number_of_models}')
    print(f'learning_rate : {learning_rate_models}')
    print(f'seed_models : {seed_models}')
    print(f'width_layers_models : {width_layers_models}')
    print(f'number_features_output : {number_features_output}')
    print(f'input_shape_models : {input_shape_models}')

    optimizer = optax.adam(learning_rate=learning_rate_models)

    models = []
    for _ in range(number_of_models):
        model_ = nl.models.FlaxModel(
            flax_module=FCNN_Arcitecture(
                width=width_layers_models,
                number_features_output=number_features_output
                ), 
            optimizer=optimizer, 
            input_shape=input_shape_models,
            seed=seed_models
            )
        models.append(model_)

    print(f'------------- {number_of_models} MODELS ARE CREATED -------------')
    

      

    return models
