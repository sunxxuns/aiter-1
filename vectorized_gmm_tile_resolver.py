import torch

device = "cpu"
gs_dtype = torch.int64
M = 267424
K = 1280
N = 2560
G = 32

gs_list = [M // G] * G
gs_list[-1] += M % G
group_sizes = torch.tensor(gs_list, dtype=gs_dtype, device=device)

BLOCK_SIZE_M = 256
BLOCK_SIZE_K = 64
BLOCK_SIZE_N = 256


def cdiv(x, y):
    return (x + y - 1) // y


tile = 1983

cumsum_m = torch.cumsum(group_sizes, dim=0)
num_m_tiles = cdiv(group_sizes, BLOCK_SIZE_M)
num_n_tiles = cdiv(N, BLOCK_SIZE_N)
num_tiles = num_m_tiles * num_n_tiles
cumsum_tile = torch.cumsum(num_tiles, dim=0)
g = torch.sum(cumsum_tile <= tile)
g_range = torch.arange(G)
prev_mask = g_range < g
zeros = torch.zeros((G,), dtype=gs_dtype)
prev_cumsum_m = torch.where(prev_mask, cumsum_m, zeros).max()
prev_cumsum_tile = torch.where(prev_mask, cumsum_tile, zeros).max()
g_cumsum_m = torch.where(g_range == g, cumsum_m, zeros).max()
m = g_cumsum_m - prev_cumsum_m
num_m_tiles_out = cdiv(m, BLOCK_SIZE_M)
tile_in_mm = tile - prev_cumsum_tile

print("group_sizes", group_sizes)
print("cumsum_m", cumsum_m)
print("num_m_tiles", num_m_tiles)
print("num_n_tiles", num_n_tiles)
print("num_tiles", num_tiles)
print("cumsum_tile", cumsum_tile)
print("g", g.item())
print("prev_mask", prev_mask)
print("prev_cumsum_m", prev_cumsum_m.item())
print("prev_cumsum_tile", prev_cumsum_tile.item())
print("g_cumsum_m", g_cumsum_m.item())

print("out: g>", g.item())
print("out: m>", m.item())
print("out: num_m_tiles>", num_m_tiles_out.item())
print("out: last_m>", prev_cumsum_m.item())
print("out: tile_in_mm>", tile_in_mm.item())
