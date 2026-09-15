import pandas as pd


class ZOBSpeedAdjuster:
    def __init__(self, data):
        """
        Initialize the speed adjuster with flight data.

        Args:
            data (pd.DataFrame): DataFrame containing flight data.
        """
        self.data = data

    def adjust_speed(self, adjustments):
        """
        Apply speed adjustments based on the rules provided.

        Args:
            adjustments (dict): A dictionary where keys are aircraft IDs and
                                values are dictionaries with 'time_shift' and 'speed_factor'.
        """
        self.data['time_diff'] = self.data.groupby('acId')['recTime'].diff().fillna(pd.Timedelta(seconds=0))

        for ac_id, rules in adjustments.items():
            ac_id_data = self.data[self.data['acId'] == ac_id].copy()
            start_time = ac_id_data['recTime'].iloc[0] + pd.Timedelta(minutes=rules.get('time_shift', 0))
            speed_factor = rules.get('speed_factor', 1.0)

            ac_id_data['new_recTime'] = start_time + (ac_id_data['time_diff'].cumsum() * speed_factor)
            self.data.loc[self.data['acId'] == ac_id, 'new_recTime'] = ac_id_data['new_recTime']

        print("Speed adjustments applied.")

    def save_adjusted_data(self, output_path):
        """
        Save the adjusted data to a new CSV file.

        Args:
            output_path (str): Path to save the adjusted data.
        """
        self.data.to_csv(output_path, index=False)
        print(f"Adjusted data saved to {output_path}")


if __name__ == "__main__":
    input_data = pd.read_csv("/path/to/output_filtered_rows.csv")
    input_data['recTime'] = pd.to_datetime(input_data['recTime'])

    adjustments = {
        'AJT188_KBOStoKCVG': {'time_shift': -6, 'speed_factor': 0.5},
        'CHR10_KTEBtoKDAL': {'time_shift': 10, 'speed_factor': 1},
        'CJT246_KJFKtoCYHM': {'time_shift': -7, 'speed_factor': 1},
        'CJT952_CYHMtoKCVG': {'time_shift': 1, 'speed_factor': 1},
    }

    adjuster = ZOBSpeedAdjuster(input_data)
    adjuster.adjust_speed(adjustments)
    adjuster.save_adjusted_data("/path/to/new_trajectory_data.csv")
