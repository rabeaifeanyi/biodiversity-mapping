import pickle
import logging

def load_cache(path):
    """
    Loads a coordinate cache from a pickle file.

    Args:
        path (str or Path): Path to the cache file.

    Returns:
        dict: The cache dictionary containing processed image coordinates.
              Returns an empty dict if the file does not exist.
    """
    try:
        with open(path, "rb") as f:
            cache = pickle.load(f)
        logging.info(f"Cache loaded: {path}")
        return cache
    except FileNotFoundError:
        logging.info("No cache found, new cache is created.")
        return {}

def save_cache(path, cache):
    """
    Saves the coordinate cache to a pickle file.

    Args:
        path (str or Path): Path to save the cache file.
        cache (dict): The cache dictionary to serialize.
    """
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    logging.info(f"Cache saved: {path}")
