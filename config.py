import getpass
import os

URLS = [
    # "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    # "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    # "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]


def set_env(key: str):
    """Prompt for an environment variable if it is not already set."""
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")
