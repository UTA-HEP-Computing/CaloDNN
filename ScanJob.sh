#PBS -V
cd ~/LCD/DLKit
source setup.sh


mkdir -p ScanLogs
output=ScanLogs/$PBS_ARRAYID.log

echo $output >> $output
echo Running on $HOSTNAME >> $output
echo Array Number: $PBS_ARRAYID >> $output
echo Queue: $PBS_QUEUE >> $output

python -m CaloDNN.ClassificationExperiment --GracefulExit -s $PBS_ARRAYID &>> $output



