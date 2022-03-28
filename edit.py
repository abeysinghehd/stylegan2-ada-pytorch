import numpy as np
import os.path
import numpy as np
from tqdm import tqdm
import torch
import dnnlib
import legacy
import PIL.Image
import click
from typing import List, Optional

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def get_batch_inputs(latent_codes, batch_size=1):
    """Gets batch inputs from a collection of latent codes.

    This function will yield at most `self.batch_size` latent_codes at a time.

    Args:
      latent_codes: The input latent codes for generation. First dimension
        should be the total number.
    """
    total_num = latent_codes.shape[0]
    for i in range(0, total_num, batch_size):
      yield latent_codes[i:i + batch_size]
      
def linear_interpolate(latent_code,
                       boundary,
                       start_distance=-3.0,
                       end_distance=3.0,
                       steps=10):
  """Manipulates the given latent code with respect to a particular boundary.

  Basically, this function takes a latent code and a boundary as inputs, and
  outputs a collection of manipulated latent codes. For example, let `steps` to
  be 10, then the input `latent_code` is with shape [1, latent_space_dim], input
  `boundary` is with shape [1, latent_space_dim] and unit norm, the output is
  with shape [10, latent_space_dim]. The first output latent code is
  `start_distance` away from the given `boundary`, while the last output latent
  code is `end_distance` away from the given `boundary`. Remaining latent codes
  are linearly interpolated.

  Input `latent_code` can also be with shape [1, num_layers, latent_space_dim]
  to support W+ space in Style GAN. In this case, all features in W+ space will
  be manipulated same as each other. Accordingly, the output will be with shape
  [10, num_layers, latent_space_dim].

  NOTE: Distance is sign sensitive.

  Args:
    latent_code: The input latent code for manipulation.
    boundary: The semantic boundary as reference.
    start_distance: The distance to the boundary where the manipulation starts.
      (default: -3.0)
    end_distance: The distance to the boundary where the manipulation ends.
      (default: 3.0)
    steps: Number of steps to move the latent code from start position to end
      position. (default: 10)
  """
  assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
          len(boundary.shape) == 2 and
          boundary.shape[1] == latent_code.shape[-1])

  linspace = np.linspace(start_distance, end_distance, steps)
  if len(latent_code.shape) == 2:
    linspace = linspace - latent_code.dot(boundary.T)
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    return latent_code + linspace * boundary
  if len(latent_code.shape) == 3:
    linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
    return latent_code + linspace * boundary.reshape(1, 1, -1)
  raise ValueError(f'Input `latent_code` should be with shape '
                   f'[1, latent_space_dim] or [1, N, latent_space_dim] for '
                   f'W+ space in Style GAN!\n'
                   f'But {latent_code.shape} is received.')


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--steps', help='Number of steps', default=10, type=int, metavar='INT')
@click.option('--boundry', 'boundry_path', help='Path of the boundry file', metavar='PATH', required=True)
@click.option('--outdir', 'output_dir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--seeds', type=num_range, required=True, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
def linear_interpolate_images(boundry_path, output_dir, network_pkl, steps, truncation_psi, seeds):
  """Interpolate function."""

  print(f'Initializing generator.')
  print('Loading networks from "%s"...' % network_pkl)
  device = torch.device('cuda')
  with dnnlib.util.open_url(network_pkl) as f:
      G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

  print(f'Preparing boundary.')
  if not os.path.isfile(boundry_path):
    raise ValueError(f'Boundary `{boundry_path}` does not exist!')
  boundary = np.load(boundry_path)
  np.save(os.path.join(output_dir, 'boundary.npy'), boundary)

  print(f'Preparing latent codes.')
  # if os.path.isfile(input_latent_codes_path):
    # print(f'  Load latent codes from `{input_latent_codes_path}`.')
    # latent_codes = np.load(input_latent_codes_path)
    # # preprocess
    # latent_codes = latent_codes.reshape(-1, 512)
    # latent_codes = latent_codes.astype(np.float32)
  # else:
  print(f'  Sample latent codes randomly.')
  # latent_codes = model.easy_sample(args.num, **kwargs)
  # latent_codes = np.random.randn(10, 14, 512)
  # latent_codes = latent_codes.reshape(-1, 14, 512)
  # latent_codes = latent_codes.astype(np.float32)
  
  # Generate images.
  latent_codes = []
  label = torch.zeros([1, G.c_dim], device=device)
  # seeds = [101,102,103,104,105,106,107,108,109,110]
  for seed_idx, seed in enumerate(seeds):
      z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
      ws = G.mapping(z, label, truncation_psi=truncation_psi)
      ws = ws.to('cpu').numpy()
      latent_codes.append(ws)
  latent_codes = np.array(latent_codes)
  latent_codes = latent_codes.reshape(-1, 14, 512)
  latent_codes = latent_codes.astype(np.float32)

  np.save(os.path.join(output_dir, 'latent_codes.npy'), latent_codes)
  total_num = latent_codes.shape[0]

  print(f'Editing {total_num} samples.')
  for sample_id in tqdm(range(total_num), leave=False):
    interpolations = linear_interpolate(latent_codes[sample_id:sample_id + 1],
                                        boundary,
                                        # start_distance=args.start_distance,
                                        # end_distance=args.end_distance,
                                        steps)
    print(f'shape of interpolations {interpolations.shape}')

    interpolation_id = 0
    for interpolations_batch in get_batch_inputs(interpolations):
      assert interpolations_batch.shape[1:] == (G.num_ws, G.w_dim)
      
      save_path = os.path.join(output_dir,
                                 f'{sample_id:03d}_{interpolation_id:03d}.jpg')
      interpolations_batch = torch.from_numpy(interpolations_batch).to(device)
      # print(f'type of interpolations_batch {type(interpolations_batch)}')
      # print(f'shape of interpolations_batch {interpolations_batch.shape}')
      img = G.synthesis(interpolations_batch, noise_mode='const')
      img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
      img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(save_path)
      interpolation_id += 1
      
    assert interpolation_id == steps
    print(f'  Finished sample {sample_id:3d}.')
  print(f'Successfully edited {total_num} samples.')


if __name__ == '__main__':
  linear_interpolate_images() # pylint: disable=no-value-for-parameter
  # linear_interpolate_images(
    # boundry_path='/content/drive/MyDrive/colab_resources/sleeve_boundry_v83r73.npy',
    # output_dir = '/content/out',
    # network_pkl = '/content/network2.pkl', 
    # steps = 10
  # )



