"""Visualise the Frensel zones."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


def plot_fresnel_zone(points: np.ndarray) -> None:
    """Plot the Fresnel zones in three projections."""
    projections = [
        ("EN", 0, 1),
        ("EZ", 0, 2),
        ("NZ", 1, 2),
    ]
    labels = ["Easting", "Northing", "Z"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))


    for ax, (name, i, j) in zip(axes, projections, strict=True):
        xy = points[:, [i, j]]
        

        # Surface points
        ax.scatter(
            xy[:, 0], 
            xy[:, 1], 
            s=3, 
            c="tab:blue", 
            alpha=1, 
            label="Surface points"
        )
        
        # Label point A
        ax.annotate(
            "A",                 
            xy=xy[0],             
            xytext=xy[0],        
            c="tab:blue"
        )
        
        # Label point B
        ax.annotate(
            "B",                 
            xy=xy[1],             
            xytext=xy[1],        
            c="tab:blue"
        )

        # Hull outline (repeat first vertex to close the loop)
        hull = ConvexHull(xy)
        v = np.append(hull.vertices, hull.vertices[0])
        ax.plot(
            xy[v, 0], 
            xy[v, 1], 
            color="tab:red", 
            linewidth=0.75, 
            label="Convex hull"
        )
        
        # Filled area
        ax.fill(
            xy[v, 0],
            xy[v, 1],
            facecolor="tab:red",
            alpha=0.2,          # transparency so points/gridlines show through
            edgecolor="none",
            zorder=1,
            label="Fresenl zone"
        )

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.set_title(f"{name} projection")
        ax.set_aspect("auto")
        

    axes[0].legend()
    fig.tight_layout()
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.show()