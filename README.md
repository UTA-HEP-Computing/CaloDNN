# CaloDNN

Get the packages:

    git clone https://bitbucket.org/anomalousai/DLKit
    cd DLKit
    git clone https://github.com/UTA-HEP-Computing/CaloDNN

Work from DLKit Directory:

    cd DLKit

Check out the arguments:

    python -m CaloDNN.ClassificationExperiment --help

Edit `CaloDNN/ClassificationScanConfig.py` to set input files and
experiment configuration.

Run an experiment:

    python -m CaloDNN.ClassificationExperiment

Run a series of experiments with different parameters (as configured
in `Params` dictionary in `CaloDNN/ClassificationScanConfig.py`) by first seeing
how many possible experiments:

    python -m CaloDNN.ClassificationScanConfig

and then running N experiments with the different configurations using
`-s` option and an integer between 0 and N:

    python -m CaloDNN.ClassificationExperiment -s 5

The models and results are stored in `TrainedModels` directory.

For example, you can get a summary of results (all numbers stored in
the Model MetaData) using:

    python -m DLTools.ScanAnalysis TrainedModels/

You can load a trained model:

    python -im TrainedModels/<ModelName>
       
or all of your trained models:

    python -im TrainedModels/*

and use the model for inference, further training, or inspection of
metadata (e.g. using `MyModel[0].MetaData`).

