
import run
import dataset
import model

import os.path as path

m = model.Connectivity(verbose=True)
connectivity = m.transform(dataset.news.dates(100000))
model.save(path.join(run.output_dir, 'builds/connectivity.hd5'), connectivity)
