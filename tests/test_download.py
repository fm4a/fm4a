"""
Tests for the fm4a.downloads module.
"""
import numpy as np
import requests

from fm4a.download import get_merra_urls

def url_exists(url: str) -> bool:
    """
    Test whether given url exists.

    Args:
        url: A string containing the URL.

    Return:
        'True' if the URL returns a valid status code, 'False' otherwise.
    """
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def test_get_merra_urls():
    """
    Ensure that returned URLs are valid.
    """
    time = np.datetime64("2020-01-01")
    urls = get_merra_urls(time)
    for url in urls:
        assert url_exists(url)
