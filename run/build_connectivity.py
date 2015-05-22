
import run
import dataset
import model

import os.path as path

m = model.Connectivity(verbose=True)
connectivity = m.transform(dataset.news.dates().data)
model.save(path.join(run.output_dir, 'builds/connectivity.hd5'), connectivity)