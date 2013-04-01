# timelapse

This is a simple python script that uses numpy and PIL to register
a stack of photographs against one another using Cross-correlation.

Cross-correlation is a well documented image processing technique,
which can be read up on by googling 
[Auto correlation](http://www.google.com/search?q=autocorrelation) or 
[Cross correlation](http://www.google.com/search?q=crosscorrelation)

[![Build Status](https://www.travis-ci.org/dzwarg/timelapse.png)](https://www.travis-ci.org/dzwarg/timelapse)

## Usage

    ./register.py -d <directory> -s <standard set>
    
Where the `directory` refers to the directory that contains all of the
unaligned images from a timelapse sequence.

The `standard set` is a directory that contains a slice of the image sequence
from `directory` that should be used as a reference set.  This reference set
will have all images in the `directory` aligned to them.

Register will shift the registered images, and save them into a folder named
'shifted', inside of the `directory`.  In addition, register will log the
amounts of the offsets in a .csv file, 'log.csv'.

## Known Limitations

Currently, the registration script only registers the images for each hour.
The script reads the filename to determine the time, where the filename is
constructed as:

    YYMMDDHHMMSS.jpg

## Issues

Use the github issue tracker if you encounter any problems:

https://github.com/dzwarg/timelapse/issues
