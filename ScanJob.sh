#PBS -V
printenv
mkdir -p ScanLogs
output=ScanLogs/$PBS_ARRAYID.log

echo $output >> $output
echo Running on $HOSTNAME >> $output
echo Array Number: $PBS_ARRAYID >> $output
echo Queue: $PBS_QUEUE >> $output

cd ~/LCD/DLKit
source setup.sh

python -m CaloDNN.ClassificationExperiment -s $PBS_ARRAYID &>> $output



