import os.path
from os import path
import shutil
from shutil import copyfile
if __name__ == "__main__":
    if (path.exists('results')):
        shutil.rmtree('results')
    if (path.exists('datasets\\test')):
        shutil.rmtree('datasets\\test')
    os.mkdir(os.path.join('datasets', 'test')  )
        # copyfile(request.form.get('file'), 'datasets\\test\\img.png')

