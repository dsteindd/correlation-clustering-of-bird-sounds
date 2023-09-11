import csv
import os
from collections import defaultdict
from typing import List

from pydub import AudioSegment

from torch_datasets import read_species_list

FRAME_RATE = 44100


def read_metadata_file(
        metadata_path: str
):
    meta = dict()
    with open(metadata_path, "r") as f:
        lines = csv.DictReader(f)

        for line in lines:
            meta[line["id"]] = {
                "gen": line["gen"],
                "sp": line["sp"],
                "type": line["type"],
                "duration": line["duration"]
            }

    return meta


def do_train_test_split_no_balancing(
        in_dirs: List[str],
        metadata_path: str,
        include_species: List[str],
        out_dir: str,
        min_duration: float = 1,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        min_duration_seconds: float = None
):
    species_dict = {
        species: [] for species in include_species
    }

    metadates = read_metadata_file(metadata_path=metadata_path)

    for in_dir in in_dirs:
        for species in include_species:
            species_dir = os.path.join(in_dir, species)

            if not os.path.isdir(species_dir):
                continue

            for file in os.listdir(species_dir):
                file_path = os.path.join(in_dir, species, file)
                file_id = os.path.splitext(file)[0]

                # print(in_dir, species, file_id)

                duration = float(metadates[file_id]["duration"])
                if duration < min_duration:
                    continue

                species_dict[species].append({
                    "path": file_path,
                    "duration": duration
                })

        cumulative_duration = {

        }

        # delete all species with less than required duration
        for (species, file_list) in species_dict.copy().items():
            cum_duration = sum([file_item["duration"] for file_item in file_list])
            if min_duration_seconds is not None and cum_duration < min_duration_seconds:
                print("Excluding Species: ", species)
                del species_dict[species]
            else:
                cumulative_duration[species] = cum_duration

        print("Remaining Species after deletion: ", len(species_dict))
        for species, file_list in species_dict.items():
            def get_file_list_cumulative_duration(cum_duration: float):
                current_duration = 0.0
                current_files = []
                while current_duration < cum_duration and file_list:
                    # best pick
                    current_file = min(file_list, key=lambda x: x["duration"] + current_duration - cum_duration)
                    current_file_index = file_list.index(current_file)
                    file_list.pop(current_file_index)
                    # print(current_file)

                    # current_file = file_list.pop()
                    current_files.append(current_file["path"])
                    current_duration += current_file["duration"]

                return current_files, current_duration

            # fill test up to test duration
            train_duration = train_split * cumulative_duration[species]
            val_duration = val_split * cumulative_duration[species]
            test_duration = test_split * cumulative_duration[species]

            test_files, current_test_duration = get_file_list_cumulative_duration(test_duration)
            val_files, current_val_duration = get_file_list_cumulative_duration(val_duration)

            train_files, current_train_duration = get_file_list_cumulative_duration(train_duration)

            assert (set(test_files).intersection(set(val_files)) == set())
            assert (set(val_files).intersection(set(train_files)) == set())
            assert (set(test_files).intersection(set(train_files)) == set())

            print(f"Test Duration for Species {species} in InDir {in_dir} is {current_test_duration}...")
            print(f"Val Duration for Species {species} in InDir {in_dir} is {current_val_duration}...")
            print(f"Train Duration for Species {species} in InDir {in_dir} is {current_train_duration}...")

            # copy to out dir
            out_train_species_dir = os.path.join(out_dir, species, "train")
            if not os.path.exists(out_train_species_dir):
                os.makedirs(out_train_species_dir)

            out_val_species_dir = os.path.join(out_dir, species, "val")
            if not os.path.exists(out_val_species_dir):
                os.makedirs(out_val_species_dir)

            out_test_species_dir = os.path.join(out_dir, species, "test")
            if not os.path.exists(out_test_species_dir):
                os.makedirs(out_test_species_dir)

            for test_file in test_files:
                file_name = test_file.split(os.path.sep)[-1]
                file_path = os.path.join(out_test_species_dir, file_name)
                # shutil.copy2(test_file, file_path)

                wav_file_path = file_path.replace(".mp3", ".wav")

                if not os.path.isfile(wav_file_path):
                    # convert
                    sound = AudioSegment.from_file(test_file)
                    sound.set_frame_rate(FRAME_RATE)
                    sound.export(wav_file_path, format='wav')

            for val_file in val_files:
                file_name = val_file.split(os.path.sep)[-1]
                file_path = os.path.join(out_val_species_dir, file_name)
                # shutil.copy2(val_file, file_path)

                wav_file_path = file_path.replace(".mp3", ".wav")
                if not os.path.isfile(wav_file_path):
                    # convert
                    sound = AudioSegment.from_file(val_file)
                    sound.set_frame_rate(FRAME_RATE)
                    sound.export(wav_file_path, format='wav')

            for train_file in train_files:
                file_name = train_file.split(os.path.sep)[-1]
                file_path = os.path.join(out_train_species_dir, file_name)
                # shutil.copy2(train_file, file_path)

                wav_file_path = file_path.replace(".mp3", ".wav")
                if not os.path.isfile(wav_file_path):
                    # convert
                    sound = AudioSegment.from_file(train_file)
                    sound.set_frame_rate(FRAME_RATE)
                    sound.export(wav_file_path, format='wav')


def save_species_list_with_min_duration(
        directory: str,
        meta_data_path: str,
        min_duration: float,
        out_path: str
):
    metadata = read_metadata_file(meta_data_path)

    cumulative_duration = defaultdict(float)

    for species in os.listdir(directory):
        species_dir = os.path.join(directory, species)

        for file_id in os.listdir(species_dir):
            file_id = os.path.splitext(file_id)[0]
            cumulative_duration[species] += float(metadata[file_id]["duration"])

    for species, duration in cumulative_duration.copy().items():
        if duration < min_duration:
            del cumulative_duration[species]

    species_list = list(cumulative_duration.keys())

    with open(out_path, "w") as f:
        f.writelines(species_list)


def main():
    # min_duration_qA = 600
    #
    # save_species_list_with_min_duration(
    #     directory="./data_q-All_germany/song/quality=A",
    #     meta_data_path="./data_q-All_germany/germany-meta-qAll.csv",
    #     min_duration=min_duration_qA,
    #     out_path="../species-3600.txt"
    # )

    species_list = read_species_list("../species-3600.txt")

    # ### QUALITY A
    do_train_test_split_no_balancing(
        in_dirs=[
            "../data/data_q-All_germany/song/quality=A/",
        ],
        metadata_path="../data/data_q-All_germany/germany-meta-qAll.csv",
        include_species=species_list,
        out_dir=f"../data/split-3600-A/",
        min_duration=1,
        min_duration_seconds=None,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1
    )

    do_train_test_split_no_balancing(
        in_dirs=[
            "./data_q-All_germany/song/quality=B/",
        ],
        metadata_path="../data/data_q-All_germany/germany-meta-qAll.csv",
        include_species=species_list,
        out_dir=f"../data/split-3600-B/",
        min_duration=1,
        min_duration_seconds=None,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1
    )


if __name__ == "__main__":
    main()
