import pathlib as pl
import pandas as pd

def getDirContents(path, columns = ['File_Name', 'Parent', 'Full_Path', 'Modified', 'File_Size', 'File', 'Directory']):
    if isinstance(path, str):
        path = pl.Path(path)
    all_files = []
    #for i in path.glob('**/*'): #only includes files not directories
    for i in path.glob('**/*'):
        if not i.name.startswith('.'): #exclude hidden files/folder
            all_files.append((i.name, str(i.parent), str(i.absolute()), i.lstat().st_mtime, i.stat().st_size, i.is_file(), i.is_dir()))
    df = pd.DataFrame(all_files, columns = columns)
    return df
