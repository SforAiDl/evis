import pathlib

import numpy as np
from torchvision.datasets.utils import download_and_extract_archive


class NMNIST:
    """
    N-MNIST dataset introduced in `Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades <https://www.frontiersin.org/articles/10.3389/fnins.2015.00437/full>` paper.

    Dataset `homepage <https://www.garrickorchard.com/datasets/n-mnist>`

    Parameters
    -----------
    root: string or Pathlib.Path object.
        Root directory where dataset will be downloaded and extracted.
    split: string
        Must be one of `train` or `test`
    donwload: bool
        Whether to download the dataset or not.

    """

    base_foler = "NMNIST"

    urls = {
        "train": "https://data.mendeley.com/public-files/datasets/468j46mzdv/files/39c25547-014b-4137-a934-9d29fa53c7a0/file_downloaded",
        "test": "https://data.mendeley.com/public-files/datasets/468j46mzdv/files/05a4d654-7e03-4c15-bdfa-9bb2bcbea494/file_downloaded",
    }

    md5s = {
        "train": "20959b8e626244a1b502305a9e6e2031",
        "test": "69ca8762b2fe404d9b9bad1103e97832",
    }

    def __init__(self, root, split="train", download=False):
        self.root = root
        assert split in ("train", "test"), TypeError
        self._split = split
        self._base_folder = pathlib.Path(self.root) / self.base_foler

        if download:
            self.download()

        if not self._check_exist():
            raise RuntimeError(
                "Dataset not found, you can use `download=True` to download it."
            )

    def download(self):
        """
        Downloads the dataset and extract using utility function from torchvision.
        """
        if not self._check_exist():

            download_and_extract_archive(
                url=self.urls[self._split],
                download_root=self._base_folder,
                filename=self._split + ".zip",
                md5=self.md5s[self._split],
            )

    def decode_binary_to_npz(self, file_decode_path, npz_file_path):
        """
        Decoding single binary file to npz file.
        """
        # ref - https://drive.google.com/drive/folders/16PYo5Jo3VlFC6-Lvw4c2hB-EAEf_egTL
        with open(file_decode_path, "rb") as bin_f:

            data = np.uint32(np.fromfile(bin_f, dtype=np.uint8))
            x = data[0::5]
            y = data[1::5]

            remaining_data = data[2::5]

            p = (remaining_data & 128) >> 7  #
            ts = ((remaining_data & 127) << 16) | (data[3::5] << 8) | (data[4::5])

            event_data = {"x": x, "y": y, "p": p, "ts": ts}
            np.savez(
                npz_file_path,
                ts=event_data["ts"],
                x=event_data["x"],
                y=event_data["y"],
                p=event_data["p"],
            )

    def decode(self):
        """
        Decodes all the binary files in npz format.
        """
        source_dir = self._base_folder / f"{self._split.capitalize()}"
        target_dir = self._base_folder / f"{self._split.capitalize()}_npz"

        for class_dir_name in source_dir.iterdir():
            target_class_dir = target_dir / class_dir_name.parts[-1]
            target_class_dir.mkdir(parents=True, exist_ok=True)
            for bin_file in class_dir_name.iterdir():
                target_file = target_class_dir / (
                    bin_file.parts[-1].split(".")[0] + ".npz"
                )

                self.decode_binary_to_npz(bin_file, target_file)
                bin_file.unlink()  # removes the binary file as the data is now stored in npz format.
        print("Decoded all the files. ")

    def _check_exist(self):
        return (self._base_folder / self._split.capitalize()).is_dir()

    def __getitem__(self, index):
        # TODO logic for indexing
        pass

    def __len__(self):
        # TODO this depends on __getitem__ method.

        pass
