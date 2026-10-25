import csv
import itertools
import os.path
from multiprocessing import Pool
from typing import Dict, List, Union

import wget
import requests

# xeno-canto API v3 requires a personal API key, see https://xeno-canto.org/account
API_KEY_ENV = "XENO_CANTO_API_KEY"


def get_generic_url(search="", area: Union[str, None] = None, quality: Union[str, None] = None, page: int = 1):
    query_string = f"query={search}" + (f"+area:{area}" if area is not None else "") + (
        f"+q:{quality}" if quality is not None else "")

    return f'https://xeno-canto.org/api/3/recordings?{query_string}&page={page}'


def query_api(url: str) -> Dict:
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise Exception(f"Set the environment variable {API_KEY_ENV} to your xeno-canto API key")

    # the key is passed separately so that it never appears in printed URLs
    response = requests.get(url, params={"key": api_key}).json()
    if "error" in response:
        raise Exception(f"xeno-canto API error: {response['message']}")

    return response


def get_duration_from_minute_str(length: str):
    split = length.split(":")
    split.reverse()
    total_seconds = 0
    for i in range(len(split)):
        d = int(split[i])
        total_seconds += d * 60 ** i
    return total_seconds


def process_downloads_for_page(config: Dict):
    # check that config
    page = config["page"]
    search = config["search"]
    area = config["area"]
    quality = config["quality"]
    out_dir = config["out_dir"]

    if page is None or search is None:
        raise Exception("Either page or search keyword not provided for download")

    url = get_generic_url(search=search, page=page, area=area, quality=quality)

    response = query_api(url)

    recordings = response["recordings"]

    metadata = []

    for recording in recordings:
        recording_id = recording["id"]
        metadata.append({
            "id": recording_id,
            "gen": recording["gen"],
            "sp": recording["sp"],
            "lat": recording["lat"],
            "lng": recording["lon"],
            "type": recording["type"],
            "quality": recording["q"],
            "cnt": recording["cnt"],
            "duration": get_duration_from_minute_str(recording["length"])
        })

        quality = recording["q"]
        quality_sub_dir = f"quality={quality}"

        sound_type = recording["type"]
        if isinstance(sound_type, list):
            sound_type = str.join(", ", sound_type).lower()

        if ("song" in sound_type or 'gesang' in sound_type) and ("call" in sound_type or "ruf" in sound_type):
            sound_type_subdir = "song+call"
        elif "song" in sound_type or 'gesang' in sound_type:
            sound_type_subdir = "song"
        elif "call" in sound_type or "ruf" in sound_type:
            sound_type_subdir = "call"
        else:
            sound_type_subdir = "unknown"

        species_name = (recording["gen"].strip() + " " + recording["sp"].strip())
        base_path = os.path.join(out_dir, sound_type_subdir, quality_sub_dir, species_name)

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        if os.path.exists(base_path + "/" + recording_id + ".mp3"):
            continue

        download_url = recording["file"]
        if download_url:
            wget.download(download_url, out=os.path.join(base_path, f"{recording_id}.mp3"))

    print(f"Finished downloading page {page}")

    return metadata


def unique_by(iterable: List, selector):
    seen = set()
    return [seen.add(selector(obj)) or obj for obj in iterable if selector(obj) not in seen]


def process_downloads_with_keyword(
        search: str = "",
        area: Union[str, None] = None,
        quality: Union[str, None] = None,
        out_dir: str = "./data",
        save: bool = False
) -> List:
    """

    Args:
        search: The keyword to search for
        area: The area key to search for
        quality: The quality to include, only single quality or None
        out_dir: The output directory for the files.

    Returns:

    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    response = query_api(get_generic_url(search=search, quality=quality, area=area))
    page_num = response["numPages"]

    print(f"Downloading {page_num} pages worth of data")

    is_empty = response["numRecordings"] == 0
    if is_empty:
        print(f"Query for {search} was empty")
        return []

    page_download_configs = [{
        "page": page,
        "area": area,
        "out_dir": out_dir,
        "quality": quality,
        "search": search
    } for page in range(1, page_num + 1)]

    with Pool(os.cpu_count()) as p:
        metadatas = p.map(process_downloads_for_page, page_download_configs)
        metadatas = unique_by(list(itertools.chain.from_iterable(metadatas)), lambda meta: meta["id"])

        if save:
            save_metadata(metadatas, os.path.join(out_dir, "metadata.csv"))

        return metadatas


def save_metadata(metadata: List, path: str):
    with open(path, "w", newline="") as f:
        title = "id,gen,sp,lat,lng,type,quality,cnt,duration".split(",")
        cw = csv.DictWriter(f, title, delimiter=",")
        cw.writeheader()
        cw.writerows(metadata)


def process_downloads_with_list_of_search_keywords(
        config: Union[List[str], None] = None,
        area: Union[str, None] = None,
        quality: Union[str, None] = None,
        out_dir: str = "./data"
):
    """

    Args:
        out_dir:
        quality:
        area:
        config: A list of search keywords. Any file matching the keyword will be downloaded.

    Returns: None

    """
    # process only downloads in parallel, without getting metadata, parallelize over pages

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    metadatas = []
    for search in config:
        metas = process_downloads_with_keyword(search, area=area, quality=quality, out_dir=out_dir)
        metadatas.extend(metas)
        metadatas = unique_by(metadatas, lambda meta: meta["id"])

    save_metadata(metadatas, os.path.join(out_dir, "metadata.csv"))


def process_metadata_download_with_keyword(
        search: str,
        area: Union[str, None],
        out_dir: str,
        quality: Union[str, None],
        fn: str = "metadata.csv"
):
    response = query_api(get_generic_url(search=search, quality=quality, area=area))
    page_num = response["numPages"]

    metadates = []
    for page in range(1, page_num + 1):
        url = get_generic_url(search=search, quality=quality, area=area, page=page)
        print(f"Calling URL: {url}")

        response = query_api(url)

        recordings = response["recordings"]

        for recording in recordings:
            metadates.append({
                "id": recording["id"],
                "gen": recording["gen"],
                "sp": recording["sp"],
                "lat": recording["lat"],
                "lng": recording["lon"],
                "type": recording["type"],
                "sex": recording["sex"],
                "license": recording["lic"],
                "quality": recording["q"],
                "time": recording["time"],
                "duration": get_duration_from_minute_str(recording["length"]),
                "recorder": recording["rec"],
                "other_birds": recording["also"]
            })

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, fn)

    with open(out_path, "w") as f:
        writer = csv.DictWriter(f,
                                fieldnames=[
                                    "id", "gen", "sp", "lat", "lng", "type", "sex", "license", "quality", "time",
                                    "duration", "recorder", "other_birds"
                                ])
        writer.writeheader()

        writer.writerows(metadates)


if __name__ == "__main__":
    # downloads all data from germany, sorted by quality and tag only containing of 'song', 'call' or both 'song+call'
    process_downloads_with_keyword(
        search="cnt:germany",
        area=None,
        out_dir="../data/data_q-All_germany",
        quality=None,
        save=True
    )
