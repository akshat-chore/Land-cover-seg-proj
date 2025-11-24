import numpy as np
import matplotlib

# Use non-interactive backend for server-side rendering
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation


class FloodSimulator:
    """
    Simple CA-style flood spread model on top of a land-cover mask.

    Expected class IDs (aligned with your segmentation):
        0: background
        1: building
        2: woodland
        3: water
    """

    BACK_ID = 0
    BUILD_ID = 1
    WOODS_ID = 2
    WATER_ID = 3

    def __init__(self, mask_array: np.ndarray):
        if mask_array.ndim != 2:
            raise ValueError("FloodSimulator expects a 2D mask")

        self.height, self.width = mask_array.shape
        self.mask = mask_array.astype(np.uint8)

        # Basic class masks
        self.water_sources = (self.mask == self.WATER_ID)
        self.building_mask = (self.mask == self.BUILD_ID)
        self.woodland_mask = (self.mask == self.WOODS_ID)
        self.background_mask = (self.mask == self.BACK_ID)

        # Land where water has spread (non-building)
        self.flooded_land = np.zeros((self.height, self.width), dtype=bool)
        # Buildings that have become flooded
        self.flooded_buildings = np.zeros((self.height, self.width), dtype=bool)

        # Simple "age" of water presence (for visualization)
        self.water_age = np.zeros((self.height, self.width), dtype=np.float32)

        # Track first time when any pixel becomes wet (including initial water)
        self.arrival_time = np.full((self.height, self.width), -1, dtype=np.int16)
        self.arrival_time[self.water_sources] = 0

    @property
    def wet_any(self) -> np.ndarray:
        """
        Boolean mask of any cell that currently has water:
        - original water body
        - flooded land
        - flooded buildings
        """
        return self.water_sources | self.flooded_land | self.flooded_buildings

    def _neighbors_8(self, grid: np.ndarray) -> np.ndarray:
        """
        8-neighbourhood of a boolean grid (dilation).
        """
        up = np.roll(grid, -1, axis=0)
        down = np.roll(grid, 1, axis=0)
        left = np.roll(grid, -1, axis=1)
        right = np.roll(grid, 1, axis=1)
        up_left = np.roll(up, -1, axis=1)
        up_right = np.roll(up, 1, axis=1)
        down_left = np.roll(down, -1, axis=1)
        down_right = np.roll(down, 1, axis=1)

        return up | down | left | right | up_left | up_right | down_left | down_right

    def step(self, frame_idx: int):
        """
        One time step of flood spread.
        Water expands 1 pixel per step from existing wet cells:
        - through background + woodland
        - into buildings (they can flood) but water does NOT propagate *through* them.
        """
        current_wet = self.wet_any

        # 8-neighbourhood of all wet cells
        neigh = self._neighbors_8(current_wet)

        # Land that can be newly flooded this step (background + woodland)
        land_traversable = self.background_mask | self.woodland_mask
        new_flood_land = neigh & land_traversable & (~self.flooded_land) & (~self.water_sources)

        # Buildings that get flooded (have a wet neighbour) but do not pass water further
        new_flood_buildings = neigh & self.building_mask & (~self.flooded_buildings)

        # Update state
        if np.any(new_flood_land):
            self.flooded_land[new_flood_land] = True
            self.arrival_time[new_flood_land] = frame_idx

        if np.any(new_flood_buildings):
            self.flooded_buildings[new_flood_buildings] = True
            self.arrival_time[new_flood_buildings] = frame_idx

        # Update age for any wet cell
        wet = self.wet_any
        self.water_age[wet] += 1.0


def run_simulation_pipeline(mask: np.ndarray, output_gif_path: str):
    """
    Run the flood simulation on a land-cover mask and save an animated GIF.

    Parameters
    ----------
    mask : np.ndarray
        2D uint8 array with class IDs (0=bg, 1=building, 2=woodland, 3=water).
    output_gif_path : str
        Path where the GIF will be written.

    Returns
    -------
    metrics : dict
        {
            impact_ratio: float (% of building pixels flooded),
            total_buildings: int,
            flooded_buildings: int,
            safe_buildings: int,
            safe_ratio: float (% of building pixels that remain safe),
            status: str (NO_FLOODING / MINOR / MODERATE / CRITICAL_HIGH_IMPACT),
            gif_path: str,
            arrival_time_map: List[List[int]],  # downsampled
            building_flood_mask: np.ndarray (bool mask, same shape as input mask)
        }
    """
    # Number of frames (we'll step the sim only every 2 frames)
    MAX_FRAMES = 60
    MINUTES_PER_FRAME = 1  # purely for the title "T = X min"

    sim = FloodSimulator(mask)

    # ---- Matplotlib Figure ----
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    ax.axis("off")

    # Initial image
    initial_img = np.zeros((sim.height, sim.width, 3), dtype=np.uint8)
    im = ax.imshow(initial_img)
    title_obj = ax.set_title("Flood Simulation: T = 0 min", fontsize=12)

    def build_frame_image() -> np.ndarray:
        """
        Compose RGB image from current simulator state.
        - background + woodland + buildings in grey tones
        - water as deep blue
        - flooded buildings as brighter blue
        - white outline around current flood front
        """
        img = np.zeros((sim.height, sim.width, 3), dtype=np.uint8)

        # Base land colors (all greys, like your notebook)
        img[sim.background_mask] = [110, 110, 110]
        img[sim.woodland_mask] = [130, 130, 130]
        img[sim.building_mask] = [160, 160, 160]

        # Initial water body in darker blue
        img[sim.water_sources] = [25, 75, 155]

        # Flooded land (non-building) overlay
        flooded_land = sim.flooded_land
        img[flooded_land] = [40, 90, 180]

        # Flooded buildings overlay (slightly brighter blue)
        flooded_buildings = sim.flooded_buildings
        img[flooded_buildings] = [70, 130, 210]

        # White outline around current wet region (like a flood edge contour)
        wet = sim.wet_any
        if np.any(wet):
            # Compute a simple "edge": wet cells that have at least one dry neighbour
            up = np.roll(wet, -1, axis=0)
            down = np.roll(wet, 1, axis=0)
            left = np.roll(wet, -1, axis=1)
            right = np.roll(wet, 1, axis=1)
            inner = wet & up & down & left & right
            edge = wet & (~inner)
            img[edge] = [255, 255, 255]

        return img

    def update(frame_idx: int):
        # Make spread slower: only advance the flood every 2nd frame
        if frame_idx > 0 and frame_idx % 2 == 0:
            sim.step(frame_idx)

        # Update visualization
        img = build_frame_image()
        im.set_data(img)

        # Update title with time in minutes
        t_min = frame_idx * MINUTES_PER_FRAME
        title_obj.set_text(f"Flood Simulation: T = {t_min} min")

        return [im, title_obj]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=MAX_FRAMES,
        blit=False,
        repeat=False,
    )
    # Slower playback: 60 frames / 4 fps = 15 seconds
    anim.save(output_gif_path, writer="pillow", fps=4)
    plt.close(fig)

    # ---- Metrics after simulation completes ----
    print("📊 Computing flood impact metrics.")

    building_mask = (mask == FloodSimulator.BUILD_ID)
    total_buildings = int(np.sum(building_mask))

    # Buildings that ended up flooded (bool mask)
    flooded_buildings_mask = sim.flooded_buildings
    flooded_buildings = int(np.sum(flooded_buildings_mask))

    safe_buildings_mask = building_mask & (~flooded_buildings_mask)
    safe_buildings = int(np.sum(safe_buildings_mask))

    if total_buildings > 0:
        impact_ratio = (flooded_buildings / total_buildings) * 100.0
        safe_ratio = (safe_buildings / total_buildings) * 100.0
    else:
        impact_ratio = 0.0
        safe_ratio = 0.0

    # Status label based on impact ratio
    if total_buildings == 0 or flooded_buildings == 0:
        status = "NO_FLOODING"
    elif impact_ratio < 5.0:
        status = "MINOR"
    elif impact_ratio < 20.0:
        status = "MODERATE"
    else:
        status = "CRITICAL_HIGH_IMPACT"

    # Downsample arrival_time for lighter JSON payload (1/4 resolution)
    ds_factor = 4
    arrival_time_ds = sim.arrival_time[::ds_factor, ::ds_factor].tolist()

    metrics = {
        "impact_ratio": float(round(impact_ratio, 2)),      # %
        "total_buildings": int(total_buildings),
        "flooded_buildings": int(flooded_buildings),
        "safe_buildings": int(safe_buildings),
        "safe_ratio": float(round(safe_ratio, 2)),          # %
        "status": status,
        "gif_path": output_gif_path,
        "arrival_time_map": arrival_time_ds,
        # NEW: full-resolution flooded-building mask (bool ndarray)
        "building_flood_mask": flooded_buildings_mask,
    }

    return metrics
