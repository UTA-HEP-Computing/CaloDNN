import h5py,os
import glob
from subprocess import call

FileSearch="/data/afarbin/LCD/*/*.h5"

dtype       = 'float16' # 'float16' # Half-precision should be enough.
compression = 'gzip'  #'gzip'

print "Searching in :",FileSearch
Files = glob.glob(FileSearch)

for fname in Files:
    print "Compressing",fname
    outdir=os.path.dirname(fname)+"-Compress"
    basename=os.path.basename(fname)
    try:
        os.makedirs(outdir)
    except:
        pass

    call(["/usr/bin/h5repack","-f","images:GZIP=2",fname,outdir+"/"+basename])


    # inFile=h5py.File(fname)
    # outFile=h5py.File(outdir+"/"+basename,"w")

    # for ds in inFile:
    #     inData=inFile[ds]
    #     chunkshape  = inData.shape
    #     outData=outFile.create_dataset(ds, inData.shape, chunks=chunkshape, dtype=dtype, compression=compression)
    #     outData=inData
        
    # inFile.close()
    # outFile.close()
