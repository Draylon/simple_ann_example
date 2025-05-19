import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load data
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode no target | Nesse caso, o target é categórico, precisamos dele um formato numérico
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)


# Split treino / teste
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define uma ANN
model = Sequential([
    Dense(10, input_shape=(4,), activation='relu'), # deeply connected layer (dense) | entra  com 4 features
    Dense(3, activation='softmax') # 3 classes do problema
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n\nNum GPUs Available:", len(tf.config.list_physical_devices('GPU')),"\n\n")

import time
start_time = time.time()
model.fit(X_train, y_train, epochs=200,verbose=0)
end_time = time.time()
print(f"Training time: {end_time - start_time:.4f} seconds")

# Avaliar o accuracy
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)


# featurezinha para salvar o layout do modelo
#from tensorflow.keras.utils import plot_model
#plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)


def animate():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    X1 = iris.data[:, 2:4]  # Petal length and width
    X2 = iris.data[:, 0:2]  # SEPALA length and width
    y = iris.target
    labels = iris.target_names

    # Color mapping
    colors = ['#ACE1AF', 'green', 'purple']  # Celadon green, green, purple
    label_color_map = {0: colors[0], 1: colors[1], 2: colors[2]}

    fig, [ax1,ax2] = plt.subplots(1,2,figsize=(12, 6))

    sc1 = ax1.scatter([], [], s=60, edgecolor='black')
    ax1.set_xlim(0, 7)
    ax1.set_ylim(0, 3)
    ax1.set_xlabel("Petal Length (cm)")
    ax1.set_ylabel("Petal Width (cm)")
    ax1.set_title("Animated Iris Scatter Plot")
    for i, label in enumerate(iris.target_names):
        ax1.scatter([], [], color=colors[i], label=label)
    ax1.legend()

    sc2 = ax2.scatter([], [], s=60, edgecolor='black')
    ax2.set_xlim(4, 8)
    ax2.set_ylim(1.5, 4.5)
    ax2.set_xlabel("Sepal Length (cm)")
    ax2.set_ylabel("Sepal Width (cm)")
    ax2.set_title("Animated Iris Scatter Plot")
    for i, label in enumerate(iris.target_names):
        ax2.scatter([], [], color=colors[i], label=label)
    ax2.legend()

    X_points1, Y_points1, C_points1 = [], [], []
    X_points2, Y_points2, C_points2 = [], [], []

    def update(frame):
        X_points1.append(X1[frame, 0])
        Y_points1.append(X1[frame, 1])
        C_points1.append(label_color_map[y[frame]])
        sc1.set_offsets(list(zip(X_points1, Y_points1)))
        sc1.set_color(C_points1)
        
        X_points2.append(X2[frame, 0])
        Y_points2.append(X2[frame, 1])
        C_points2.append(label_color_map[y[frame]])
        sc2.set_offsets(list(zip(X_points2, Y_points2)))
        sc2.set_color(C_points2)
        return sc1, sc2,

    ani = FuncAnimation(fig, update, frames=len(X1), interval=1, blit=False)
    plt.show()




def test_random(n=10):
    """
    Test the model on n random samples and print prediction vs actual.
    
    Parameters:
    - model: trained Keras model
    - X: input features (e.g., X_test)
    - y: true labels (integer-encoded)
    - n: number of random samples to test
    """
    if n > len(X_test):
        print(f"Warning: n ({n}) is greater than the number of test samples ({len(X_test)}). Setting n to {len(X_test)}.")
        n = len(X_test)
    correct = 0
    indices = np.random.choice(len(X_test), size=n, replace=False)

    for i in indices:
        sample = X[i].reshape(1, -1)  # reshape for prediction
        prediction = model.predict(sample, verbose=0)
        predicted_class = np.argmax(prediction)
        actual_class = y[i]

        #print(f"Sample {i}: Predicted = {predicted_class}, Actual = {actual_class}")

        if predicted_class == actual_class:
            correct += 1

    print(f"\n✅ Correct predictions: {correct}/{n} | Total: {n}")
    
#test_random_predictions(n=100)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

def build_ann_model(input_dim, 
                    hidden_layers=[10],
                    activation='relu', 
                    dropout_rate=0.0,
                    use_batchnorm=False,
                    output_units=3,
                    output_activation='softmax',
                    l2_reg=0.0):
    """
    Construtor de Model de RNA com dropout, batch normalization, e L2 regularization.

    Parametros:
    - input_dim: número de recursos de entrada
    - hidden_layers: lista de tamanhos de camadas ocultas (por exemplo, [10])
    - activation: função de ativação para camadas ocultas
    - dropout_rate: taxa de desligamento de neuronio (0,0 para desabilitar)
    - use_batchnorm: se a normalização em lote deve ser aplicada | normalização de treino em batch
    - output_units: número de classes de saída
    - output_activation: ativação para a camada de saída
    - l2_reg: fator de regularização L2 (0,0 para desabilitar) | Penalização de pesos grandes

    Returns:
    - A compiled Keras model
    """
    model = Sequential()
    
    for i, units in enumerate(hidden_layers):
        if i == 0:
            model.add(Dense(units, input_dim=input_dim,
                            activation=activation,
                            kernel_regularizer=l2(l2_reg)))
        else:
            model.add(Dense(units,
                            activation=activation,
                            kernel_regularizer=l2(l2_reg)))

        if use_batchnorm:
            model.add(BatchNormalization())
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    model.add(Dense(output_units, activation=output_activation))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
