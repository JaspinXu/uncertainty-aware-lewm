"""Record verified upstream revisions in the workspace expected by cache scripts."""
import argparse
from pathlib import Path
import subprocess

REVISIONS = {'le-wm': ('paper-code.commit', '8edfeb336732b5f3ce7b8b210d0ba370a09e2cac'),
             'stable-worldmodel-paper': ('paper-dependency.commit', 'abdced49809d5eae38e24b27dc7b635c502c4812')}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    for directory, (name, expected) in REVISIONS.items():
        actual = subprocess.check_output(['git','-C',str(args.root/directory),'rev-parse','HEAD'], text=True).strip()
        if actual != expected:
            raise RuntimeError(f'{directory}: expected {expected}, got {actual}')
        path = args.root/'setup'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text().strip() != expected:
            raise FileExistsError(f'Conflicting revision record: {path}')
        path.write_text(expected+'\n')
    print('Upstream revisions verified.')

if __name__ == '__main__':
    main()
