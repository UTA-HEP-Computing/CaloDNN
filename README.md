# CaloDNN

Get the packages:

    git clone https://bitbucket.org/anomalousai/DLKit
    cd DLKit
    git clone https://github.com/UTA-HEP-Computing/CaloDNN

Work from DLKit Directory:

    cd DLKit

Check out the arguments:

    python -m CaloDNN.ClassificationExperiment --help

Edit `CaloDNN/ClassificationScanConfig.py` to set input files and experiment configuration.

Run an experiment:

    python -m CaloDNN.ClassificationExperiment

Look at the results in `TrainedModels` directory.

You can load a trained model:

    python -im TrainedModels/<ModelName>
       
or all of your trained models:

    python -im TrainedModels/*

and use the model for inference, further training, or inspection of metadata (e.g. using `MyModel[0].MetaData`).

