"""Read trusted training metadata across operating systems without global patches."""
import os
import pathlib
import pickle


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath" and os.name == "nt":
            return pathlib.PurePosixPath
        if module == "pathlib" and name == "WindowsPath" and os.name != "nt":
            return pathlib.PureWindowsPath
        return super().find_class(module, name)


load = pickle.load
loads = pickle.loads
dump = pickle.dump
dumps = pickle.dumps
Pickler = pickle.Pickler
