import pickle
import tensorflow as tf
from tensorflow.keras import models
import silence_tensorflow.auto
import numpy as np
from .transformer import Transformer


class Avocato():
    def __init__(self, modelname):
        self.metadata = pickle.load(
            open("/mnt/hackathon2021/modelcache/meta/" + modelname + ".pkl",
                 "rb"))
        self.model = Transformer(
            num_layers=self.metadata["num_layers"],
            d_model=self.metadata["d_model"],
            num_heads=self.metadata["num_heads"],
            dff=self.metadata["dff"],
            input_dimensions=self.metadata["input_dim"],
            target_dimensions=self.metadata["output_dim"],
            rate=self.metadata["rate"],
        )
        self.model.compile()
        self.model.load_weights("/mnt/hackathon2021/modelcache/models/" +
                                modelname)

    def __call__(self, net_in):
        """
        Args:
        -----
            - net_in: pd dataframe consisting of n_timesteps of data for HARPS (timestamp, harp, ...parameters)
        """

        in_data = np.zeros(
            (1, self.metadata["n_timesteps"], self.metadata["max_n_harps"],
             self.metadata["n_features"]))
        unique_timesteps = net_in["timestamp"].unique()
        for t_id, timestep in enumerate(unique_timesteps):
            if t_id >= self.metadata["n_timesteps"]:
                continue
            unique_harps = net_in[net_in["timestamp"] ==
                                  timestep]["harp"].unique()
            for h_id, harp in enumerate(unique_harps):
                if h_id >= self.metadata["max_n_harps"]:
                    continue
                in_data[0, t_id, h_id] = net_in[
                    (net_in["timestamp"] == timestep) &
                    (net_in["harp"] == harp
                     )].loc[:,
                            net_in.columns != "timestamp"].to_numpy().reshape(
                                self.metadata["n_features"])
                in_data[0, t_id, h_id] -= self.metadata["sharp_mean"]
                in_data[0, t_id, h_id] /= self.metadata["sharp_std"]

        output = np.zeros((
            1,
            self.metadata["n_timesteps"],
            self.metadata["n_out"],
        ))

        out_list = []
        for i in tf.range(self.metadata["n_timesteps"]):
            output, _ = self.model([in_data, output], training=False)
            out_list.append(output * self.metadata["xray_std"] +
                            self.metadata["xray_mean"])

        return out_list
