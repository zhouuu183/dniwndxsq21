import torch

from models.ear_modules_v5 import MorphologyInteriorAnalyzer


def circle_masks(size=64):
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    radius = ((x - 32).float().square() + (y - 32).float().square()).sqrt()
    disk = (radius <= 12).float().view(1, 1, size, size)
    ring = ((radius >= 9) & (radius <= 12)).float().view(1, 1, size, size)
    return disk, ring


def test_solid_fills_material_and_hollow_preserves_hole():
    disk, ring = circle_masks()
    analyzer = MorphologyInteriorAnalyzer()
    background = torch.full((1, 3, 64, 64), 0.4)
    solid_source = background * (1 - disk) + torch.tensor([0.9, 0.25, 0.1]).view(1, 3, 1, 1) * disk
    solid = analyzer(solid_source, disk, disk, disk)
    assert solid["morphology"][0] == "solid"
    assert float(solid["solid_interior"].sum()) > 0
    hollow_source = background * (1 - ring) + torch.tensor([0.9, 0.8, 0.1]).view(1, 3, 1, 1) * ring
    hollow = analyzer(hollow_source, ring, ring, ring)
    assert hollow["morphology"][0] == "hollow"
    assert float(hollow["hollow_interior"].sum()) > 0
    assert float((hollow["hollow_interior"] * ring).sum()) == 0.0

