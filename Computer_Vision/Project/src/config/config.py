from pathlib import Path

url = "https://zenodo.org/record/3723295/files/candidates.csv?download=1"

exec = 'colab'

if exec == 'local':
    data_path = Path('drive/MyDrive/data/data_3D')
else:
    data_path = Path('/tmp/myapp/data')



