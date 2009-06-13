#!/usr/bin/env python
# Recipe from http://code.activestate.com/recipes/496685/

def download(url):
    """Copy the contents of a file from a given URL to a local file."""
    import urllib
    webFile = urllib.urlopen(url)
    localFile = open(url.split('/')[-1], 'w')
    localFile.write(webFile.read())
    webFile.close()
    localFile.close()
    return
