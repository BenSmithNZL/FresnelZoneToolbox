import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


def plot_fresnel_zone(points: np.ndarray) -> None:
    projections = [
        ("EN", 0, 1),
        ("EZ", 0, 2),
        ("NZ", 1, 2),
    ]
    labels = ["Easting", "Northing", "Z"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))


    for ax, (name, i, j) in zip(axes, projections, strict=True):
        xy = points[:, [i, j]]
        hull = ConvexHull(xy)

        # Points
        ax.scatter(xy[:, 0], xy[:, 1], s=5, c="tab:blue", alpha=0.7, label="Fresnel points")

        # Hull outline (repeat first vertex to close the loop)
        v = np.append(hull.vertices, hull.vertices[0])
        ax.plot(xy[v, 0], xy[v, 1], color="tab:red", linewidth=1.5, label="Convex hull")

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.set_title(f"{name} projection")
        ax.set_aspect("auto")

    axes[0].legend()
    fig.tight_layout()
    plt.show()