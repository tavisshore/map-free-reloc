from pathlib import Path
from tqdm import tqdm

train = Path('data/MFR/train')
train_seqs = [x for x in train.iterdir() if x.is_dir()]

for seq in tqdm(train_seqs):
    sub_seqs = [x for x in seq.iterdir() if x.is_dir()]
    # check there are files within the subdirectories
    for sub_seq in sub_seqs:
        # check for jpg files
        assert len(list(sub_seq.glob('*.jpg'))) > 0, f'{sub_seq} does not have jpg files'

    # check there are intrinsics.txt and poses.txt files
    assert (seq / 'intrinsics.txt').exists(), f'{seq} does not have intrinsics.txt'
    assert (seq / 'poses.txt').exists(), f'{seq} does not have poses.txt'
