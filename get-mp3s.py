#!/usr/bin/python3
import sys
import re
import urllib.request

with open(sys.argv[1], 'r') as input:
    for url in (x.rstrip("\r\n") for x in input):
        path = 'audio/' + re.sub('/', '-', url)
        print(path)

        contents = urllib.request.urlopen(url).read()
        with open(path, 'wb') as out:
            out.write(contents)
