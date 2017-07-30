import os


def iter_files_with_ext(root_dir, ext):
    """
    Iterate recursively on all files in root dir, ending with <ext>.
    :param root_dir: Directory to list.
    :param ext: Extension of the files.
    :return: absolute path of all files.
    """
    for path, sub_dirs, files in os.walk(root_dir):
        for name in files:
            if not name.startswith(".") and name.endswith(ext):
                abs_path = os.path.abspath(os.path.join(path, name))
                if os.path.isfile(abs_path):
                    yield os.path.join(path, name)
