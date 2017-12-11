import scipy.io as sio
import numpy as np

print sio.whosmat('modified.mat')

matlab_contents = sio.loadmat('modified.mat')

depths = matlab_contents['depths']
np.save('nyu_dataset_depths', depths)

images = matlab_contents['images']
np.save('nyu_dataset_images', images)

labels = matlab_contents['labels']
np.save('nyu_dataset_labels', labels)

names = matlab_contents['names']
np.save('nyu_dataset_names', names)

scenes = matlab_contents['scenes']
np.save('nyu_dataset_scenes', scenes)