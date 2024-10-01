from flask import Flask, request, send_file
from flasgger import Swagger
from utils.brain_tumor import BrainTumorModel
import nibabel as nib
import numpy as np
import os

app = Flask(__name__)
swagger = Swagger(app)

brain_model = BrainTumorModel()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction on brain tumor segmentation
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: flair
        in: formData
        type: file
        required: true
        description: Nifti file for flair MRI.
      - name: t1ce
        in: formData
        type: file
        required: true
        description: Nifti file for t1ce MRI.
    responses:
      200:
        description: A gif showing the MRI slices with segmentation overlay.
        content:
          image/gif:
            schema:
              type: string
              format: binary
    """
    flair_file = request.files['flair']
    t1ce_file = request.files['t1ce']
    
    # Save files to disk
    flair_pth = f'data/{flair_file.filename}'
    t1ce_pth = f'data/{t1ce_file.filename}'
    flair_file.save(flair_pth)
    t1ce_file.save(t1ce_pth)
    
    try:
        # Load NIfTI files
        flair = nib.load(flair_pth).get_fdata()
        t1ce = nib.load(t1ce_pth).get_fdata()
        
        # Make prediction
        prediction = brain_model.make_prediction(flair, t1ce)
        mri_data = np.argmax(prediction, axis=-1)
        
        # Create GIF
        gif_filename = brain_model.create_gif(flair, mri_data)
        
        # GIF generation
        response = send_file(gif_filename, mimetype='image/gif')

        # Delete files after sending response
        os.remove(flair_pth)
        os.remove(t1ce_pth)
        os.remove(gif_filename)

        return response

    
    except Exception as e:
        return {'error': f'Error happened {e}'}, 500


if __name__ == '__main__':
    app.run(debug=True)