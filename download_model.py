import os
import sys
import requests
from tqdm import tqdm

if len(sys.argv) != 2:
    print('You must enter the model name as a parameter, e.g.: download_model.py 124M')
    sys.exit(1)

model = sys.argv[1]

subdir = os.path.join('models', model)
if not os.path.exists(subdir):
    os.makedirs(subdir)
subdir = subdir.replace('\\','/') # needed for Windows

name = 'model.ckpt'

for filename in ['checkpoint','hparams.json','encoder.json','vocab.bpe','model.ckpt.index', 'model.ckpt.meta', 'model.ckpt.data-00000-of-00001']:

    filename = filename.replace('model.ckpt', name)

    bucket = os.environ.get('BUCKET', 'gpt-2')
    path = os.environ.get('MODEL_DIR', 'gs://{bucket}/{subdir}'.format(bucket=bucket, subdir=subdir)).lstrip('gs:').strip('/')
    url = "https://openaipublic.blob.core.windows.net/" + path + "/" + filename
    r = requests.get(url, stream=True)
    if not r.ok and filename == 'checkpoint':
        raise FileNotFoundError(url)
    
    if not r.ok:
        continue

    with open(os.path.join(subdir, filename), 'wb') as f:
        file_size = int(r.headers["content-length"])
        chunk_size = 1000
        with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
            # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(chunk_size)
    if filename == 'checkpoint':
        with open(os.path.join(subdir, filename)) as f:
            for line in f:
                if line.startswith('model_checkpoint_path'):
                    name = line.split(':', 1)[1].strip().strip('"')

        
