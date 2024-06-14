import time
import torch
import zipfile
import requests
from tqdm import tqdm


def download(url: str, fname: str, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            resp = requests.get(url, stream=True)
            # Check the response status code (raise an exception if it's not in the range 200-299)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(fname, "wb") as file, tqdm(
                desc=f"Downloading {url} to {fname}...",
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in resp.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            return

        except requests.exceptions.HTTPError as errh:
            print(f"HTTP error occurred: {errh}")
            retry_count += 1
            continue
        except requests.exceptions.ConnectionError as errc:
            print(f"Connection error occurred: {errc}")
            retry_count += 1
            continue
        except requests.exceptions.Timeout as errt:
            print(f"Timeout error occurred: {errt}")
            retry_count += 1
            continue
        except Exception as err:
            print(f"Other error occurred: {err}")
            retry_count += 1
            continue

    else:
        print(f"The operation could not be completed after {max_retries} retries.")
        exit()


def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, "r")
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print("This is not zip")


def time_stamp(timestamp):
    if timestamp:
        return timestamp.strftime("%Y-%m-%d_%H-%M-%S")

    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))


def to_cuda(x):
    if hasattr(x, "cuda"):
        if torch.cuda.is_available():
            return x.cuda()

    return x
