"""Get the mapped W values for Z values"""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import csv

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--outfilecsv', help='Where to save the w values csv', type=str, required=True, metavar='DIR')
def generate_w_values(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    outfilecsv: str,
):

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # os.makedirs(outdir, exist_ok=True)

    if seeds is None:
        ctx.fail('--seeds option is required to get the W values')
        
    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    
    with open(outfilecsv, 'w') as f:
        writer = csv.writer(f)

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        ws = G.mapping(z, label, truncation_psi=truncation_psi)
        unique_weights = ws[0][0].numpy()
        csv_row = ','.join([str(seed)] + [str(i) for i in unique_weights])
        writer.writerow(csv_row)
        print('Generated ws value for seed %d (' % (seed))
        print(f"ws Tensor: \n {ws} \n")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_w_values() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
