"""
Here we keep vairous types of callabacks
"""

import logging
import os

from dotenv import load_dotenv
from keras.callbacks import TensorBoard

load_dotenv()


def tensorboard_callback():
    logging.info("Instantiating tensorboard callback")
    log_dir = os.environ["TENSORBOARD_SINK"]

    if not os.path.exists(log_dir):
        logging.warn(f"path {log_dir} did not exist, creating")
        os.makedirs(log_dir)
    return TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        write_steps_per_second=True,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=1,
        embeddings_metadata=None,
    )
