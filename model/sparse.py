
import os.path as path
import scipy.sparse
import numpy as np
import h5py as h5

#
# Save methods
#
def _save_formats_coordinate(h5f, array):
    h5f.create_dataset('row', data=array.row, compression=True)
    h5f.create_dataset('col', data=array.col, compression=True)

def _save_formats_compressed(h5f, array):
    h5f.create_dataset('indices', data=array.indices, compression=True)
    h5f.create_dataset('indptr', data=array.indptr, compression=True)

_save_formats = {
    'coo': _save_formats_coordinate,
    'csc': _save_formats_compressed,
    'csr': _save_formats_compressed
}

def save(filename, array):
    ext = path.splitext(filename)[1]

    if (ext == '.hd5'):
        h5f = h5.File(filename, mode='w')
        h5f.create_dataset('data', data=array.data, compression=True)
        h5f.attrs['shape'] = array.shape
        h5f.attrs['format'] = array.getformat()

        _save_formats[array.getformat()](h5f, array)

        h5f.close()
    elif (ext == '.npy'):
        np.save(filename, array)
    elif (ext == '.npz'):
        np.savez(filename, **array)
    else:
        raise NotImplementedError('filetype %s not supported for saveing' % ext)

#
# Load methods
#
def _load_formats_coordinate(h5f):
    return scipy.sparse.coo_matrix(
        (h5f['data'][:], (h5f['row'][:], h5f['col'][:])),
        shape=h5f.attrs['shape']
    )

def _load_formats_compressed(h5f):
    return getattr(scipy.sparse, h5f.attrs['format'] + '_matrix')(
        (h5f['data'][:], h5f['indices'][:], h5f['indptr'][:]),
        shape=h5f.attrs['shape']
    )

_load_formats = {
    'coo': _load_formats_coordinate,
    'csc': _load_formats_compressed,
    'csr': _load_formats_compressed
}

def load(filename):
    ext = path.splitext(filename)[1]

    if (ext == '.hd5'):
        h5f = h5.File(filename, mode='r')
        matrix = _load_formats[h5f.attrs['format']](h5f)
        h5f.close()
    elif (ext == '.npy'):
        matrix = np.load(filename)
    elif (ext == '.npz'):
        matrix = np.load(filename)
    else:
        raise NotImplementedError('filetype %s not supported for loading' % ext)

    return matrix
