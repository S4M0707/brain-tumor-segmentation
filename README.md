# Brain Tumor Segmentation

This project uses a pre-trained U-Net model to segment brain tumors from MRI scans (FLAIR and T1CE). It is built with a FastAPI backend to handle API calls and a frontend interface that allows easy uploading and visualization of the segmented tumor. The final output is visualized as a GIF, displaying the segmented tumor regions.

## Features

- **Brain Tumor Segmentation** using FLAIR and T1CE MRI scans.
- **FastAPI Backend** for efficient and scalable API calls.
- **Frontend Interface** for easy MRI scan upload and visualization.
- **GIF Output** showing the segmented tumor over time.

## Requirements

- Python 3.10
- FastAPI
- Uvicorn
- NumPy
- NiBabel
- ImageIO
- Matplotlib
- Scikit-image
- TensorFlow

## Installation

1. Clone the repository:
    `git clone git@github.com:S4M0707 brain-tumor-segmentation.git`
2. Navigate to the project directory:
    `cd brain-tumor-segmentation`
3.  Install dependencies:
    `pip install -r requirements.txt`
4.  Run the FastAPI server: 
    `uvicorn main:app`


## Usage

1.  Navigate to the FastAPI interface (default: `http://localhost:8000`)
2.  Upload FLAIR and T1CE MRI scans.
3.  Click "Submit" to segment the tumor.
4.  Visualize the segmented tumor in the GIF output.


## Docker Usage

You can also pull the pre-built Docker image for easier deployment:

1.  Pull the Docker image:
    `docker pull s4m0707/brain-tumor-segment:v3.0`
2.  Run the Docker container:
    `docker run -p 8000:8000 s4m0707/brain-tumor-segment:v3.0`
3.  Access the application via `http://localhost:8000`.


## Contributing

Contributions are welcome! To contribute:

1.  Fork the repository.
2.  Create a new branch with your feature or fix.
3.  Commit your changes.
4.  Submit a pull request for review.

## License

This project is licensed under the MIT License.

## Future Work

*   Improve model accuracy
*   Add support for other MRI modalities
*   Integrate with clinical systems


## Project Structure

*   `main.py`: FastAPI application
*   `utils/brain_tumor.py`: Brain tumor segmentation model
*   `index.html`: Frontend interface
*   `requirements.txt`: Dependencies