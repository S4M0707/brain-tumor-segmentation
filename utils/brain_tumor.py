import tensorflow as tf
import nibabel as nib
import numpy as np
import cv2
import os
import imageio
import matplotlib.pyplot as plt

class BrainTumorModel:
    """
    A class used to handle brain tumor segmentation using a pre-trained U-Net model.
    
    Methods
    -------        
    make_prediction(flair, t1ce)
        Prepares input data and makes predictions using the model.
        
    create_gif(flair, mri_data)
        Creates an animated GIF overlaying segmentation results on MRI scans.
    """

    def __init__(self):
        """Initializes the BrainTumorModel by loading a pre-trained model and setting default values."""
        model_path = os.path.join('models', "3D_MRI_Brain_tumor_segmentation.keras")
        self.model = tf.keras.models.load_model(model_path, custom_objects={
            'combined_loss_multiclass': self.combined_loss_multiclass,
            'dice_coefficient_multiclass': self.dice_coefficient_multiclass,
            'precision': self.precision,
            'sensitivity': self.sensitivity,
            'specificity': self.specificity,
            'dice_coef_healthy': self.dice_coef_healthy,
            'dice_coef_core': self.dice_coef_core,
            'dice_coef_edema': self.dice_coef_edema,
            'dice_coef_enhancing': self.dice_coef_enhancing,
            'Adam': tf.keras.optimizers.Adam(learning_rate=0.001)
        })

        self.VOLUME_SLICES = 100
        self.VOLUME_START_AT = 22
        self.IMG_SIZE = 128
        self.SEGMENT_CLASSES = {
            0: 'NOT tumor',
            1: 'NECROTIC/CORE',
            2: 'EDEMA',
            3: 'ENHANCING'
        }

        print('---INITALIZATION DONE---')

    def dice_coef_healthy(self, y_true, y_pred, epsilon=tf.keras.backend.epsilon()):
        intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true[:,:,:,0] * y_pred[:,:,:,0]))
        return (2. * intersection + epsilon) / (tf.keras.backend.sum(tf.keras.backend.square(y_true[:,:,:,0])) + tf.keras.backend.sum(tf.keras.backend.square(y_pred[:,:,:,0])) + epsilon)

    def dice_coef_core(self, y_true, y_pred, epsilon=tf.keras.backend.epsilon()):
        intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
        return (2. * intersection + epsilon) / (tf.keras.backend.sum(tf.keras.backend.square(y_true[:,:,:,1])) + tf.keras.backend.sum(tf.keras.backend.square(y_pred[:,:,:,1])) + epsilon)

    def dice_coef_edema(self, y_true, y_pred, epsilon=tf.keras.backend.epsilon()):
        intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
        return (2. * intersection + epsilon) / (tf.keras.backend.sum(tf.keras.backend.square(y_true[:,:,:,2])) + tf.keras.backend.sum(tf.keras.backend.square(y_pred[:,:,:,2])) + epsilon)

    def dice_coef_enhancing(self, y_true, y_pred, epsilon=tf.keras.backend.epsilon()):
        intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
        return (2. * intersection + epsilon) / (tf.keras.backend.sum(tf.keras.backend.square(y_true[:,:,:,3])) + tf.keras.backend.sum(tf.keras.backend.square(y_pred[:,:,:,3])) + epsilon)

    def dice_coefficient_multiclass(self, y_true, y_pred):
        healthy = self.dice_coef_healthy(y_true, y_pred)
        core = self.dice_coef_core(y_true, y_pred)
        edema = self.dice_coef_edema(y_true, y_pred)
        enhancing = self.dice_coef_enhancing(y_true, y_pred)
        return (healthy + core + edema + enhancing) / 4

    def combined_loss_multiclass(self, y_true, y_pred):
        dice_loss = 1 - self.dice_coefficient_multiclass(y_true, y_pred)
        cross_entropy_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return 0.7 * dice_loss + 0.3 * cross_entropy_loss

    def precision(self, y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + tf.keras.backend.epsilon())

    def sensitivity(self, y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + tf.keras.backend.epsilon())

    def specificity(self, y_true, y_pred):
        true_negatives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        possible_negatives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(1 - y_true, 0, 1)))
        return true_negatives / (possible_negatives + tf.keras.backend.epsilon())

    def make_prediction(self, flair, t1ce):
        """
        Prepares input data and makes predictions using the model.
        
        Parameters:
        flair -- FLAIR MRI scan
        t1ce -- T1-contrast enhanced MRI scan
        
        Returns:
        The prediction result as a numpy array.
        """
        try:
            X = np.empty((self.VOLUME_SLICES, self.IMG_SIZE, self.IMG_SIZE, 2))
            for j in range(self.VOLUME_SLICES):
                X[j, :, :, 0] = cv2.resize(flair[:, :, j + self.VOLUME_START_AT], (self.IMG_SIZE, self.IMG_SIZE))
                X[j, :, :, 1] = cv2.resize(t1ce[:, :, j + self.VOLUME_START_AT], (self.IMG_SIZE, self.IMG_SIZE))
            prediction = self.model.predict(X / np.max(X), verbose=1)
            return prediction
        except Exception as e:
            print("Error:", e)
            return None

    def create_gif(self, flair, mri_data):
        try:
            flair_reshape = cv2.resize(flair[:, :, self.VOLUME_START_AT: self.VOLUME_START_AT + 100], (self.IMG_SIZE, self.IMG_SIZE))
            colors = {
                0: [255, 255, 255],  # White for 'NOT tumor'
                1: [0, 0, 255],      # Red for 'NECROTIC/CORE'
                2: [0, 255, 0],      # Green for 'EDEMA'
                3: [255, 0, 0]       # Blue for 'ENHANCING'
            }

            frames = []
            for i in range(mri_data.shape[0]):
                x = flair_reshape[:, :, i] * 10e9
                y = mri_data[i, :, :]

                x_normalized = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
                x_uint8 = np.uint8(x_normalized)
                              
                colored_segmentation = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8)
                for class_idx, color in colors.items():
                    colored_segmentation[y == class_idx] = color
                
                image_bgr = cv2.cvtColor(x_uint8, cv2.COLOR_GRAY2BGR)

                overlay = cv2.addWeighted(image_bgr, 0.6, colored_segmentation, 0.4, 0)

                frames.append(overlay)


            gif_filename = os.path.join('outputs', 'mri_3d.gif')
            imageio.mimsave(gif_filename, frames, duration=0.1, loop=0)


            return gif_filename
        
        except Exception as e:
            print("Error:", e)
            return None

if __name__ == '__main__':
    case = '010'
    flair = nib.load(os.path.join('test_data', f'BraTS20_Training_{case}_flair.nii')).get_fdata()
    t1ce = nib.load(os.path.join('test_data', f'BraTS20_Training_{case}_t1ce.nii')).get_fdata()

    obj = BrainTumorModel()
    prediction = obj.make_prediction(flair, t1ce)
    mri_data = np.argmax(prediction, axis=-1)
    obj.create_gif(flair, mri_data)