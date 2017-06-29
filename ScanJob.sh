#PBS -V
printenv
mkdir -p ScanLogs
output=ScanLogs/$PBS_ARRAYID.log

echo $output >> $output
echo Running on $HOSTNAME >> $output
echo Array Number: $PBS_ARRAYID >> $output
echo Queue: $PBS_QUEUE >> $output

cd ~/Tutorial/DLKit
source setup.sh

python -m CaloDNN.ClassificationExperiment_GammaPi0 -s $PBS_ARRAYID --GracefulExit --Test &>> $output



