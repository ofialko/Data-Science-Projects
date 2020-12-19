import zipfile
import pathlib as pl

data_path = pl.Path('/tmp/myapp/data/')
archive_file = [f for f in list(data_path.glob('*.*')) if f.name.endswith('zip')][0]

def list_files(path_to_archive:pl.PosixPath = archive_file) -> list:
    with zipfile.ZipFile(path_to_archive, 'r') as zip_file:
        return zip_file.namelist()

def extract_files(
    list_of_files:list,
    path_to_archive:pl.PosixPath = archive_file,
    output_dir:pl.PosixPath = data_path) -> None:
    '''Extracts files from a given ziped file'''

    list_of_archived_files = list_files(path_to_archive)
    with zipfile.ZipFile(path_to_archive, 'r') as zip_file:
        for file_ in list_of_archived_files:

            # Check filename endswith csv
            if file_ in list_of_files:
                # Extract a single file from zip
                zip_file.extract(file_, output_dir)