"""Helios specific wandb callback."""

import logging

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from olmo_core.train.callbacks.wandb import WandBCallback

from helios.data.dataloader import HeliosDataLoader

logger = logging.getLogger(__name__)


class HeliosWandBCallback(WandBCallback):
    """Helios specific wandb callback."""

    upload_dataset_distribution_pre_train: bool = True

    def pre_train(self):
        """Pre-train callback for the wandb callback."""
        super().pre_train()
        if self.upload_dataset_distribution_pre_train:
            assert isinstance(self.trainer.data_loader, HeliosDataLoader)
            dataset = self.trainer.data_loader.dataset
            logger.info("Gathering locations of entire dataset")
            latlons = dataset.get_geographic_distribution()
            logger.info(f"Uploading dataset distribution to wandb: {latlons.shape}")
            # Create map using EPSG:4326 (WGS 84) projection
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection=ccrs.epsg("4326"))

            # Add map features
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.LAND, alpha=0.1)
            ax.add_feature(cfeature.OCEAN, alpha=0.1)

            # Plot the data points
            ax.scatter(
                latlons[:, 1],
                latlons[:, 0],
                transform=ccrs.PlateCarree(),
                alpha=0.5,
                s=1,
            )

            ax.set_global()  # Show the entire globe
            ax.gridlines()
            ax.set_title("Geographic Distribution of Dataset")

            # Log to wandb
            self.wandb.log({"dataset_distribution": self.wandb.Image(fig)})
            plt.close(fig)
