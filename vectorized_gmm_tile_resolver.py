import sys
import torch


def cdiv(x, y):
    return (x + y - 1) // y


def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


device = "cpu"
gs_dtype = torch.int64
M = 267424
K = 1280
N = 2560
# original G is 32, testing with an odd / non-power-of-2 G
G = 13  # 32
G_POW2 = next_power_of_2(G)

gs_list = [M // G] * G
gs_list[-1] += M % G
group_sizes = torch.tensor(gs_list, dtype=gs_dtype, device=device)
assert torch.all(group_sizes >= 0)
assert torch.sum(group_sizes) == M

BLOCK_SIZE_M = 256
BLOCK_SIZE_K = 64
BLOCK_SIZE_N = 256

tile = int(sys.argv[1]) if len(sys.argv) > 1 else 1983
print("tile", tile)

if tile < 0:
    sys.exit("tile must be >= 0")

g_range = torch.arange(G_POW2, device=device)
g_mask = g_range < G
zeros = torch.zeros((G_POW2,), dtype=gs_dtype, device=device)

# emulate tl.load, with a power-of-2 tensor
loaded_group_sizes = zeros.clone()
loaded_group_sizes[:G] = group_sizes
assert torch.all(group_sizes >= 0)
assert torch.sum(loaded_group_sizes) == M
print("group_sizes", loaded_group_sizes)

num_m_tiles = cdiv(loaded_group_sizes, BLOCK_SIZE_M)
print("num_m_tiles", num_m_tiles)
num_n_tiles = cdiv(N, BLOCK_SIZE_N)
print("num_n_tiles", num_n_tiles)
num_tiles = num_m_tiles * num_n_tiles
print("num_tiles", num_tiles)
cumsum_tile = torch.where(g_mask, torch.cumsum(num_tiles, dim=0), zeros)
print("cumsum_tile", cumsum_tile)

max_tile = torch.max(cumsum_tile)
print("max_tile", max_tile.item())
if tile >= max_tile:
    sys.exit(f"tile must be < {max_tile}")

cumsum_m = torch.where(g_mask, torch.cumsum(loaded_group_sizes, dim=0), zeros)
print("cumsum_m", cumsum_m)
g = torch.sum((cumsum_tile <= tile) & g_mask)
print("g", g.item())
prev_mask = g_range < g
print("prev_mask", prev_mask)
prev_cumsum_m = torch.max(torch.where(prev_mask, cumsum_m, zeros))
print("prev_cumsum_m", prev_cumsum_m.item())
prev_cumsum_tile = torch.max(torch.where(prev_mask, cumsum_tile, zeros))
print("prev_cumsum_tile", prev_cumsum_tile.item())
g_cumsum_m = torch.max(torch.where(g_range == g, cumsum_m, zeros))
print("g_cumsum_m", g_cumsum_m.item())

m = g_cumsum_m - prev_cumsum_m
num_m_tiles_out = cdiv(m, BLOCK_SIZE_M)
tile_in_mm = tile - prev_cumsum_tile

print("out: g>", g.item())
print("out: m>", m.item())
print("out: num_m_tiles>", num_m_tiles_out.item())
print("out: last_m>", prev_cumsum_m.item())
print("out: tile_in_mm>", tile_in_mm.item())
