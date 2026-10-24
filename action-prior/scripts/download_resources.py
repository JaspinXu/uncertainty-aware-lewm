"""Download pinned official resources, verify hashes, extract only the expected data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

HASHES = {
    'cube': ('2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89',
             '3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600'),
    'pusht': ('48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607',
              '7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318')}

def verify(path, expected):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024*1024), b''):h.update(chunk)
    if h.hexdigest() != expected:raise ValueError(f'Checksum mismatch: {path}')

def download(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_name(path.name+'.partial')
        subprocess.run(['curl','-fL','--retry','5','--connect-timeout','30','-C','-',
                        '-o',str(partial),url], check=True)
        if expected:verify(partial,expected)
        partial.rename(path)
    elif expected:verify(path,expected)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', choices=HASHES, required=True)
    args=parser.parse_args()
    task,root=args.task,args.root.resolve()
    metadata=json.loads((Path(__file__).resolve().parents[1]/'provenance/resources.json').read_text())[task]
    model=metadata['model'];data=metadata.get('dataset',metadata.get('data'))
    for name in ['weights.pt','config.json']:
        download(f'https://huggingface.co/{model["id"]}/resolve/{model["revision"]}/{name}',
                 root/'checkpoints'/task/'lewm'/name, HASHES[task][0] if name=='weights.pt' else None)
    name='cube_single_expert.tar.zst' if task=='cube' else 'pusht_expert_train.h5.zst'
    archive=root/'downloads'/name
    download(f'https://huggingface.co/datasets/{data["id"]}/resolve/{data["revision"]}/{name}',archive,HASHES[task][1])
    dest=root/('datasets/ogbench/cube_single_expert.h5' if task=='cube' else 'datasets/pusht_expert_train.h5')
    if dest.exists():
        print(f'Existing dataset left unchanged: {dest}; archive verified, extracted file not rehashed.')
        return
    dest.parent.mkdir(parents=True,exist_ok=True)
    partial=dest.with_name(dest.name+'.partial')
    if partial.exists():raise FileExistsError(f'Inspect unfinished extraction first: {partial}')
    import zstandard
    with archive.open('rb') as f, zstandard.ZstdDecompressor().stream_reader(f) as stream:
        if task=='cube':
            count=0
            with tarfile.open(fileobj=stream,mode='r|') as tar:
                for member in tar:
                    if not member.isfile():continue
                    if member.name.lstrip('./')!=dest.name or count:
                        raise ValueError(f'Unexpected archive member: {member.name}')
                    count+=1
                    with tar.extractfile(member) as source, partial.open('xb') as output:
                        shutil.copyfileobj(source,output,16*1024*1024)
                    if partial.stat().st_size!=member.size:raise ValueError('Incomplete extraction')
            if count!=1:raise ValueError('Expected one HDF5 file')
        else:
            with partial.open('xb') as output:shutil.copyfileobj(stream,output,16*1024*1024)
    import h5py,hdf5plugin
    with h5py.File(partial) as f:
        for key in ['pixels','action','ep_len','ep_offset']:
            if key not in f:raise ValueError(f'Missing dataset column: {key}')
        f['pixels'][0]
    partial.rename(dest)
    print('Verified archive and extracted',dest)

if __name__=='__main__':main()
