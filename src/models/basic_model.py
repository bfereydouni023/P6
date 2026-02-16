from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Step 4: define a small CNN for 3-class facial-expression recognition.
        # - use convolution + max pooling blocks with ReLU activations
        # - include a flatten layer and one or more dense layers
        # - end in a softmax for multi-class probabilities
        # - keep parameter count under 150,000
        # We gradually increase filters while shrinking spatial dimensions via
        # pooling. This keeps the representational capacity useful while
        # respecting the parameter budget for the assignment.
        self.model = Sequential([
            # Normalize image pixels from [0, 255] -> [0, 1].
            Rescaling(1.0 / 255, input_shape=input_shape),

            # Convolutional feature extractor.
            layers.Conv2D(8, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(24, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Flatten + fully-connected head for classification.
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(categories_count, activation='softmax')
        ])
    
    def _compile_model(self):
        # Step 5: configure optimization exactly for multi-class classification.
        # RMSprop with a small learning rate is a solid default for this task.
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
