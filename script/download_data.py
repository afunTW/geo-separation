import click
import requests

from pathlib import Path
from tqdm import tqdm

DATA_URL = 'http://data.moi.gov.tw/MoiOD/System/DownloadFile.aspx?DATA=CD02C824-45C5-48C8-B631-98B205A2E35A'

@click.command()
@click.option('--save-dir', 'save_dir',
              required=True,
              default='./data',
              type=Path)
@click.option('--save-file', 'save_file',
              required=True,
              default='polygon.zip')
@click.option('--chunk-size', 'chunk_size',
              required=True,
              default=10485760,
              type=int)
def main(save_dir, save_file, chunk_size):
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    with requests.get(DATA_URL, stream=True) as r:
        r.raise_for_status()
        file_length = int(r.headers['Content-length'])
        save_file = str(save_dir / save_file)
        with open(save_file, 'wb') as f:
            progress_bar = tqdm(unit='B', total=file_length//chunk_size)
            for chunk in r.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update()
        print(f'Save - {save_file}')

if __name__ == '__main__':
    main()
