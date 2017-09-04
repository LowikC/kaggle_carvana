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


def num_lines(filename):
    """
    Return the number of lines in a file.
    """
    with open(filename, "r") as f:
        return sum(1 for _ in f)


def get_car_id_rot(filename):
    """
    Return the car id and the rotation of an image.
    :param filename: Path to the image.
    :return: car_id (str), rotation (int)
    """
    basename = os.path.basename(filename)
    basename, _ = os.path.splitext(basename)
    if "mask" in basename:
        car_id, rot, _ = basename.split("_")
    else:
        car_id, rot = basename.split("_")
    return car_id, int(rot)
