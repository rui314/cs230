#!/usr/bin/python3
import urllib.request
import re

feed_list = urllib.request.urlopen('https://www.podcastinsights.com/top-us-podcasts/').read().decode('utf-8')
feeds = set()
audios = set()

for x in re.findall(r'https://podcasts.apple.com/us/podcast/[^/"]+/id\d+', feed_list):
    feeds.add(re.sub('.*id', 'https://pcr.apple.com/id', x))

for feed_url in feeds:
    feed = urllib.request.urlopen(feed_url).read().decode('utf-8')
    for x in re.findall(r'"http[^"]+mp3["?]', feed):
        audios.add(x[1:-1])

for x in audios:
    print(x)
