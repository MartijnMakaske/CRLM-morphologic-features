# Beyond Tumor Size: A Deep Learning Framework for the Extraction of Morphologic Features Upon Systemic Treatment in Colorectal Cancer Liver Metastases

This Github contains the implementation of the proposed DL model, metric calculation and saliency map creation. 

For privacy reasons, all paths have been set to None. Fill in your own paths when reproducing the experiments. 

## 📂 Repository Structure

DL_model_training.py: This file trains the Siamese multi task DL model. 
multi_task_metrics.ipynb: Calculates the metrics for each test set
multi_task_prediction.py: Makes predictions given a pre-trained model
Scans_data_preparation.py: Prepares the scans and segmentations for training
integrated_gradients.py: Creates saliency maps using SmoothGrad and Integrated Gradients for Pathological response.
model_utils.py: Util_file containing the dataloader, model architecture (pytorch nn.module) and multi-task loss