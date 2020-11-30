from tqdm import tqdm
import requests
from pathlib import Path
import json
from config.logconfig import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def streaming(url, output_path):
    # Streaming, so we can iterate over the response.
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 8192
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(output_path, 'wb') as file:
            for data in response.iter_content(block_size):
                if data:
                    progress_bar.update(len(data))
                    file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            log.error("Something went wrong")


def main():
    with open('config/data_config.json','r') as fp:
        data_conf = json.load(fp)
        url = data_conf['url']
        file_name = url.split('/')[-1].split('?')[0]

    output_dir = Path(data_conf['output_dir'])
    log.info(f"Starting reading {file_name}")
    streaming(url, output_dir / file_name)
    

if __name__ == "__main__":
    main()