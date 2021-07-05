from tqdm import tqdm
import requests
from pathlib import Path
import json
from src.config.logconfig import logging


from src.config.config import data_path, url


log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


class getData:
    def __init__(self, url, output_dir):
        self.url = url
        self.file_name = self.url.split('/')[-1].split('?')[0]
        self.output_path = Path(output_dir) / self.file_name

    @classmethod
    def from_json(cls, json_config):
        with open(json_config,'r') as fp:
            data_conf = json.load(fp)
        url = data_conf['url']
        output_dir = data_conf['output_dir']
        return cls(url, output_dir)

    @classmethod
    def from_config(cls):
        return cls(url, data_path)

    
    def stream(self):
        # Streaming, so we can iterate over the response.
        log.info(f"Starting reading {self.file_name}")
        with requests.get(self.url, stream=True) as response:
            response.raise_for_status()
            total_size_in_bytes= int(response.headers.get('content-length', 0))
            block_size = 8192
            with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True) as progress_bar:
                with open(self.output_path, 'wb') as file:
                    for data in response.iter_content(block_size):
                        if data:
                            progress_bar.update(len(data))
                            file.write(data)
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    log.error("Something went wrong")        

                    
                    
                    

if __name__ == "__main__":
    #json_config = Path('src/config') / 'data_config.json'
    #getData.from_json(json_config).stream()
    getData.from_config().stream()
    