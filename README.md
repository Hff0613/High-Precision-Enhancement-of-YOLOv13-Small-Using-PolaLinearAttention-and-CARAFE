High Precision Enhancement of YOLOv13 Small Using PolaLinearAttention and CARAFE

Overview
This paper proposes an enhanced YOLOv13-Small model tailored for microalgae detection. By integrating PolaLinearAttention and CARAFE modules, the model attains 97.4% detection accuracy, delivering an efficient solution for automated water environment monitoring.

Datasets
Two microalgae datasets are provided for model training and validation
algae_dataset.rar Open-source microalgae dataset
custom_algae_dataset.rar Self-constructed microalgae dataset

Download and extract the compressed files to the project root directory to run the training code directly.

Requirements
All dependencies are listed in requirements.txt. Install them with
pip install -r requirements.txt

Model Structure
The entire model architecture is implemented in the nn directory, which includes the backbone, neck, head, and customized advanced modules.

Key modules
pola_attention.py Implements the PolaLinearAttention mechanism to enhance feature representation and capture long-range dependencies efficiently.
carafe.py Implements the CARAFE upsampling operator to improve the resolution and quality of feature maps with rich contextual information.

The nn directory also contains basic components including convolution blocks, activation functions, transformer structures, model heads, and task pipelines to support end-to-end training and inference.

Test Code
3.4test.py Provides a ready-to-run test script for model inference and performance evaluation on the microalgae datasets.
