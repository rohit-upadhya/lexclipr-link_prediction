import pickle

from datasets import load_dataset


class HFDatasetDownload:
    def __init__(
        self,
        dataset_name: str,
    ):
        self.dataset_name = dataset_name
        pass

    def download(
        self,
    ):
        dataset = load_dataset(self.dataset_name)
        return dataset


if __name__ == "__main__":
    hf_download_obj = HFDatasetDownload(dataset_name="rohit-upadhya/lexclipr")

    dataset = hf_download_obj.download()
    with open("dataset/lexclipr.pl", "w+") as f:
        pickle.dump(dataset)
    print(dataset)
