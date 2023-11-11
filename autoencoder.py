

import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
from pdb import set_trace
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

# DO WE NEED TO STANDARDISE BEFORE AE ?????

def return_latent_spaces(X_train, X_test, y_train, y_test):
    """Get the encoded representations."""

    def create_autoencoder(encoding_dim=128, hidden_layers=1, hidden_units=64):
        # This is our input image
        input_data = Input(shape=(500,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim, activation='relu')(input_data)
        
        # Add hidden layers
        for _ in range(hidden_layers):
            encoded = Dense(hidden_units, activation='relu')(encoded)

        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(500, activation='sigmoid')(encoded)
        # This model maps an input to its reconstruction
        autoencoder = Model(input_data, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        return autoencoder
        
    X_train = np.array(X_train).reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = np.array(X_test).reshape((len(X_test), np.prod(X_test.shape[1:])))
    #X_train = np.array(X_train).reshape((len(X_train), np.prod(X_train.shape[1:])))
    #X_test = np.array(X_test).reshape((len(X_test), np.prod(X_test.shape[1:])))

    X_train_ae, X_val_ae, y_train_ae, y_val_ae = \
        train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create a KerasClassifier for parameter tuning
    autoencoder = KerasClassifier(build_fn=create_autoencoder, epochs=200, batch_size=16)

    # Define hyperparameter grid for tuning
    param_grid = {
        'encoding_dim': [64],  # , 128, 256encoding_dim
        'hidden_layers': [1],      # , 2, 3 number of hidden layers
        'hidden_units': [32]    # , 64, 128 number of hidden units
        }
    # Perform grid search to find the best hyperparameters
    grid = GridSearchCV(estimator=autoencoder, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
    grid_result = grid.fit(X_train_ae, X_train_ae)
  
    # Print the best hyperparameters
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # Retrain the autoencoder with the best hyperparameters and early stopping
    best_autoencoder = create_autoencoder(encoding_dim=grid_result.best_params_['encoding_dim'],
                                          hidden_layers=grid_result.best_params_['hidden_layers'],
                                          hidden_units=grid_result.best_params_['hidden_units']
                                          )
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)  # Adjust patience as needed
    best_autoencoder.fit(X_train_ae, X_train_ae, epochs=200, batch_size=16, shuffle=True, validation_data=(X_val_ae, X_val_ae), callbacks=[early_stopping])

    # Use the best autoencoder for encoding
    X_train = best_autoencoder.predict(X_train)
    X_test = best_autoencoder.predict(X_train)
    return X_train, X_test  


