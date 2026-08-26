import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio
from datetime import timedelta


class ZOBVisualizer:
    def __init__(self, data_path, output_dir):
        """
        Initialize the visualizer with the path to the adjusted data and output directory.

        Args:
            data_path (str): Path to the adjusted trajectory data.
            output_dir (str): Directory to save visualizations.
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.data = None

    def load_data(self):
        """
        Load the adjusted trajectory data.
        """
        self.data = pd.read_csv(self.data_path)
        self.data['new_recTime'] = pd.to_datetime(self.data['new_recTime'])
        print(f"Data loaded from {self.data_path}")

    def visualize(self, gif_name="trajectory.gif", skip_frames=5, fps=3):
        """
        Generate a GIF visualization of flight trajectories.

        Args:
            gif_name (str): Name of the output GIF file.
            skip_frames (int): Number of frames to skip between visualizations.
            fps (int): Frames per second for the GIF.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        images = []

        earliest_time = self.data['new_recTime'].min()
        latest_time = self.data['new_recTime'].max()

        time_range = pd.date_range(earliest_time, latest_time, periods=100)

        for current_time in time_range[::skip_frames]:
            fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={"projection": ccrs.PlateCarree()})
            ax.set_extent([-85, -75, 38, 45])
            ax.add_feature(cfeature.BORDERS)
            ax.add_feature(cfeature.COASTLINE)

            for ac_id in self.data['acId'].unique():
                ac_data = self.data[self.data['acId'] == ac_id]
                subset = ac_data[ac_data['new_recTime'] <= current_time]
                ax.plot(subset['coord2'], subset['coord1'], label=ac_id)

            ax.set_title(f"Time: {current_time}")
            ax.legend()
            image_path = os.path.join(self.output_dir, f"frame_{current_time}.png")
            plt.savefig(image_path)
            images.append(image_path)
            plt.close()

        with imageio.get_writer(os.path.join(self.output_dir, gif_name), mode='I', fps=fps) as writer:
            for image in images:
                writer.append_data(imageio.imread(image))

        print(f"GIF saved to {os.path.join(self.output_dir, gif_name)}")


if __name__ == "__main__":
    visualizer = ZOBVisualizer("/path/to/new_trajectory_data.csv", "/path/to/output/")
    visualizer.load_data()
    visualizer.visualize()
