import glob
import h5py
import numpy as np
import os
import random
import sys
from DLTools.ThreadedGenerator import DLMultiClassFilterGenerator, \
    DLMultiClassGenerator
from DLTools.ThreadedGenerator import DLh5FileGenerator
from time import time

GeneratorClasses = []


def ConstantNormalization(Norms):
    def NormalizationFunction(Ds):
        out = []
        for i, Norm in enumerate(Norms):
            Ds[i] /= Norm
            out.append(Ds[i])
        return out

    return NormalizationFunction


def LCDNormalization(Norms):
    def NormalizationFunction(Ds):
        out = []
        for i, Norm in enumerate(Norms):
            if Norm != 0.:
                if type(Norm) == float:
                    Ds[i] /= Norm
                if type(Norm) == str and Norm.lower() == "nonlinear":
                    Ds[i] = np.tanh(
                        np.sign(Ds[i]) * np.log(np.abs(Ds[i]) + 1.0) / 2.0)
                out.append(Ds[i])
        return out

    return NormalizationFunction


# PreMix Generator.
def MergeAndNormInputs(NormFunc):
    def f(X):
        X = NormFunc(X)
        return [X[0], X[1]], X[2:]

    return f


def MergeInputs():
    def f(X):
        return [X[0], X[1]], X[2:]

    return f


def DivideFiles(FileSearch="/data/LCD/*/*.h5", Fractions=[.9, .1],
                datasetnames=["ECAL", "HCAL"], Particles=[], MaxFiles=-1):
    print "Searching in :", FileSearch
    Files = glob.glob(FileSearch)

    print "Found", len(Files), "files."

    FileCount = 0
    Samples = {}
    for F in Files:
        FileCount += 1
        basename = os.path.basename(F)
        ParticleName = basename.split("_")[0].replace("Escan", "")

        if ParticleName in Particles:
            try:
                Samples[ParticleName].append((F, datasetnames, ParticleName))
            except:
                Samples[ParticleName] = [(F, datasetnames, ParticleName)]

        if MaxFiles > 0:
            if FileCount > MaxFiles:
                break

    out = []
    for j in range(len(Fractions)):
        out.append([])

    SampleI = len(Samples.keys()) * [int(0)]

    for i, SampleName in enumerate(Samples):
        Sample = Samples[SampleName]
        NFiles = len(Sample)

        for j, Frac in enumerate(Fractions):
            EndI = int(SampleI[i] + round(NFiles * Frac))
            out[j] += Sample[SampleI[i]:EndI]
            SampleI[i] = EndI + 1

    return out


def SetupData(FileSearch,
              ECAL, HCAL, target,
              NClasses, f, Particles,
              BatchSize,
              multiplier,
              ECALShape,
              HCALShape,
              ECALNorm,
              HCALNorm):
    datasets = []
    shapes = []
    Norms = []

    if ECAL:
        datasets.append("ECAL")
        shapes.append((BatchSize * multiplier,) + ECALShape[1:])
        Norms.append(ECALNorm)
    if HCAL:
        datasets.append("HCAL")
        shapes.append((BatchSize * multiplier,) + HCALShape[1:])
        Norms.append(HCALNorm)
    if target:
        datasets.append("target")
        shapes.append((BatchSize * multiplier,) + (1, 5))
        Norms.append(1.)

    # This is for the OneHot    
    shapes.append((BatchSize * multiplier, NClasses))
    Norms.append(1.)

    TrainSampleList, TestSampleList = DivideFiles(FileSearch, f,
                                                  datasetnames=datasets,
                                                  Particles=Particles)

    return TrainSampleList, TestSampleList, Norms, shapes


def FilterEnergy(ECut):
    def filterfunction(f):
        r = np.where(np.logical_and(
            f['HCALfeatures'][:, 0] / f['ECALfeatures'][:, 0] < ECut,
            f['ECALfeatures'][:, 0] > 0.))
        # r=np.where(f['ECALfeatures'][:,0]>0.)
        assert len(r[0]) > 0
        return r[0]

    return filterfunction


def MakeGenerator(ECAL, HCAL,
                  SampleList, NSamples, NormalizationFunction,
                  Merge=True, **kwargs):
    if ECAL and HCAL and Merge:
        post_f = MergeInputs()
    else:
        post_f = False

    pre_f = NormalizationFunction

    #    return DLMultiClassFilterGenerator(SampleList,FilterEnergy(1.0),
    #                                       max=NSamples,
    #                                       preprocessfunction=pre_f,
    #                                       postprocessfunction=post_f,
    #                                       **kwargs)

    return DLMultiClassGenerator(SampleList,
                                 max=NSamples,
                                 preprocessfunction=pre_f,
                                 postprocessfunction=post_f,
                                 **kwargs)
