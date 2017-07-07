import h5py
import numpy as np


def lcd_3Ddata():
    f = h5py.File("EGshuffled.h5", "r")
    data = f.get('ECAL')
    dtag = f.get('TAG')
    xtr = np.array(data)
    tag = np.array(dtag)
    # xtr=xtr[...,numpy.newaxis]
    # xtr=numpy.rollaxis(xtr,4,1)
    print xtr.shape

    return xtr, tag.astype(bool)


# ##################################################################################################################
# CaloDNN
# ##################################################################################################################
def LCDN(Norms):
    def NormalizationFunction(Ds):
        # converting the data from an ordered-dictionary format to a list
        Ds = [Ds[item] for item in Ds]
        print('Ds_len', len(Ds))
        out = []
        # print('DS', Ds)
        # TODO replace with zip function
        for i, Norm in enumerate(Norms):
            if Norm != 0.:
                if isinstance(Norm, float):
                    Ds[i] /= Norm
                if isinstance(Norm, str) and Norm.lower() == "nonlinear":
                    Ds[i] = np.tanh(
                        np.sign(Ds[i]) * np.log(np.abs(Ds[i]) + 1.0) / 2.0)
                out.append(Ds[i])
        return out

    return NormalizationFunction
