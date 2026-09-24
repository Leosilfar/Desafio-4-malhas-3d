from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _sample(points: np.ndarray, maximum: int = 12000) -> np.ndarray:
    if len(points) <= maximum:
        return np.arange(len(points))
    return np.random.default_rng(9).choice(len(points), maximum, replace=False)


def overlay(reference: np.ndarray, moving: np.ndarray, path: str, title: str) -> None:
    figure = plt.figure(figsize=(10, 8), dpi=160)
    axis = figure.add_subplot(projection="3d")
    ri, mi = _sample(reference), _sample(moving)
    axis.scatter(*reference[ri].T, s=0.5, c="#1f77b4", alpha=0.48, label="Referencia: malha 1")
    axis.scatter(*moving[mi].T, s=0.5, c="#d62728", alpha=0.48, label="Movel: malha 2")
    axis.set(title=title, xlabel="X (unidade nativa)", ylabel="Y (unidade nativa)", zlabel="Z (unidade nativa)")
    axis.legend(markerscale=8, loc="upper left")
    axis.set_box_aspect(np.ptp(np.vstack((reference, moving)), axis=0))
    figure.tight_layout(); figure.savefig(path); plt.close(figure)


def distance_map(points: np.ndarray, distances: np.ndarray, path: str, clip: float) -> np.ndarray:
    figure = plt.figure(figsize=(10, 8), dpi=160)
    axis = figure.add_subplot(projection="3d")
    idx = _sample(points)
    plot = axis.scatter(*points[idx].T, s=0.8, c=distances[idx], cmap="turbo", vmin=0, vmax=clip)
    colorbar = figure.colorbar(plot, ax=axis, shrink=0.72, pad=0.08)
    colorbar.set_label("Distancia movel -> referencia (unidade nativa)")
    axis.set(title="Mapa point-wise por vertice (PLY sem faces)", xlabel="X (unidade nativa)", ylabel="Y (unidade nativa)", zlabel="Z (unidade nativa)")
    axis.set_box_aspect(np.ptp(points, axis=0)); figure.tight_layout(); figure.savefig(path); plt.close(figure)
    normalized = np.clip(distances / max(clip, 1e-8), 0, 1)
    return (plt.get_cmap("turbo")(normalized)[:, :3] * 255).astype(np.uint8)


def candidate_region_map(points: np.ndarray, candidate_mask: np.ndarray, path: str) -> None:
    figure = plt.figure(figsize=(10, 8), dpi=160)
    axis = figure.add_subplot(projection="3d")
    idx = _sample(points)
    colors = np.where(candidate_mask[idx], "#d62728", "#1f77b4")
    axis.scatter(*points[idx].T, s=0.8, c=colors, alpha=0.65)
    axis.set(
        title="Regiao candidata a alteracao (P90 dos residuos NN)",
        xlabel="X (unidade nativa)",
        ylabel="Y (unidade nativa)",
        zlabel="Z (unidade nativa)",
    )
    axis.set_box_aspect(np.ptp(points, axis=0))
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
