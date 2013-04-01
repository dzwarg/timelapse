#!/usr/bin/env python

import glob;
import os;
from numpy import *;
from numpy.fft import *;
from PIL import Image;
from PIL import ImageChops;
from optparse import OptionParser;

def dumpImage( arr, name, doLog=True ):
    """
    Dump an image to a file, optionally map pixels to the log of the pixel value.

    :param arr: An array of pixel values.
    :type arr: numpy.array
    :param name: The name of the output file.
    :type name: string
    :param doLog: A flag indicating if the output pixel values are log(pixel) values.
    :type doLog: boolean

    >>> def imgstat(path):
    ...   x = open(path, 'r')
    ...   data = x.read()
    ...   x.close()
    ...   os.unlink(path)
    ...   return (data[:10], data[-10:], len(data),)
    ...
    >>> arr = array([ [10,10], [20,20] ])
    >>> arr
    array([[10, 10],
           [20, 20]])
    >>> dumpImage(arr, '/tmp/dump1.jpg')
    >>> imgstat('/tmp/dump1.jpg')
    ('\\xff\\xd8\\xff\\xe0\\x00\\x10JFIF', '\\x07\\xdb\\x7f\\xd7%\\xfeB\\xbf\\xff\\xd9', 342)
    >>> dumpImage(arr, '/tmp/dump2.png')
    >>> imgstat('/tmp/dump2.png')
    ('\\x89PNG\\r\\n\\x1a\\n\\x00\\x00', '\\x00\\x00IEND\\xaeB`\\x82', 71)
    >>> dumpImage(arr, '/tmp/dump3.jpg', doLog=False)
    >>> imgstat('/tmp/dump3.jpg')
    ('\\xff\\xd8\\xff\\xe0\\x00\\x10JFIF', '\\xd7\\xfd\\x82\\xad\\x7f\\xf4R\\xd7\\xff\\xd9', 346)
    >>> dumpImage(arr, '/tmp/dump4.png', doLog=False)
    >>> imgstat('/tmp/dump4.png')
    ('\\x89PNG\\r\\n\\x1a\\n\\x00\\x00', '\\x00\\x00IEND\\xaeB`\\x82', 71)
    """
    if ( doLog ) :
        maxpx = log10( arr.max() );
    else :
        maxpx = arr.max();

    imOut = Image.new( "L", [ len(arr), len(arr[0]) ] );
    for x in range( len(arr) ):
        for y in range( len(arr[x]) ):
            if ( arr[x][y].real == 0 ) :
                pval = 0;
            else :
                if ( doLog ) :
                    pval = log10( abs(arr[x][y].real) ) / maxpx * 255;
                else :
                    pval = arr[x][y].real / maxpx * 255;
            imOut.putpixel( (x,y), pval );

    imOut.save( name );

    
def compareImages( fixedImage, move, log ):
    """
    Compare a reference image to a second image, and log the difference. 
    
    The comparison process is:
    
    1. move file is opened
    2. move file is cross-correlated with the fixedImage
    3. shift distance is calculated from cross-correlation
    4. shift distance is recorded in the log in the format: ``<filename>,<cross-
       correlation max real component>,<cross-correlation max imaginary 
       component>,<x shift>,<y shift>``
    5. shifted image is returned
    
    :param fixedImage: A reference image.
    :type fixedImage: Image
    :param move: An image file to shift. This image will not be modified.
    :type move: string
    :param log: A file to log how many pixels the ``move`` image was shifted.
    :type log: file
    :returns: A shifted image.
    :rtype: Image

    >>> fixedArr = array([ [0, 0, 0], [0, 255, 0], [0, 0, 0] ])
    >>> fixedFile = '/tmp/fixedImage.png'
    >>> dumpImage(fixedArr, fixedFile)
    >>> fixedImage = Image.open(fixedFile)
    >>> moveArr = array([ [0, 255, 0], [0, 0, 0], [0, 0, 0] ])
    >>> moveFile = '/tmp/moveImage.png'
    >>> dumpImage(moveArr, moveFile)
    >>> log = open('/tmp/shift.log', 'w')
    >>> shifted = compareImages(fixedImage, moveFile, log)
    >>> log.close()
    >>> list(shifted.getdata())
    [0, 0, 0, 0, 255, 0, 0, 0, 0]
    >>> log = open('/tmp/shift.log', 'r')
    >>> log.read()
    '/tmp/moveImage.png,65025.00,0.00,2,0\\n'
    >>> log.close()
    >>> os.unlink(fixedFile)
    >>> os.unlink(moveFile)
    >>> os.unlink(log.name)
    """

    #
    # Create a fixed and moving image, and read in the image data
    #
    movingImage = Image.open( move );
    bwmovingImage = movingImage.convert( "L" );

    f_size = fixedImage.size;
    #print "Fixed image size: %dx%d" % ( f_size[0], f_size[1] );

    f_list = list(fixedImage.getdata());
    #print "Fixed data list: %d" % len(f_list);

    f_arr = array( f_list );
    f_arr = f_arr.reshape( f_size[1], f_size[0] );
    #print "Fixed data numpy array: %d, d=%d" % (len(f_arr), f_arr.ndim);

    f_fourier = fft( rot90( fft( f_arr ) ) );
    #print "Fixed fourier transform: %dx%d, d=%d" % (len(f_fourier), len(f_fourier[0]), f_fourier.ndim);

    #dumpImage( f_fourier, "fixfft.jpg" );

    m_size = bwmovingImage.size;
    #print "Moving image size: %dx%d" % ( m_size[0], m_size[1] );

    m_list = list( bwmovingImage.getdata() );
    #print "Moving data list: %d" % len(m_list);

    m_arr = array( m_list );
    m_arr = m_arr.reshape( m_size[1], m_size[0] );
    #print "Moving data numpy array: %d, d=%d" % (len(m_arr), m_arr.ndim);

    m_fourier = fft( rot90( fft( m_arr ) ) );
    #print "Moving fourier transform: %dx%d, d=%d" % (len(m_fourier), len(m_fourier[0]), m_fourier.ndim);

    #dumpImage( m_fourier, "movefft.jpg" );

    correlation = multiply( f_fourier.conj(), m_fourier);
    #print "Correlation matrix dims: %d" % correlation.ndim;
    #print "Correlation matrix size: %dx%d" % (len(correlation), len(correlation[0]));

    #dumpImage( correlation, "corr.jpg" );

    inverse = ifft( rot90( rot90( rot90( ifft( correlation ) ) ) ) );

    #dumpImage( inverse, "inv.jpg" );

    max_val = inverse.max();
    #print "Maximum correlation value: %3.2f, %3.2fj" % (max_val.real, max_val.imag);

    pos = where( inverse == max_val );
    #print "Position: " + str(pos);

    log.write( "%s,%3.2f,%3.2f,%d,%d\n" % (move, max_val.real, max_val.imag, pos[1][0], pos[0][0]) );

    return ImageChops.offset( movingImage, -pos[1][0], -pos[0][0] );

def getStandardSet( standard ):
    """
    Get the all the reference images from the specified folder.

    The path may be absolute or relative, contain a trailing slash or not.

    :param standard: The path to the reference images.
    :type standard: string
    :returns: An array of the opened PIL Image objects.
    :rtype: list(Image)

    >>> referenceDir = '/tmp/reference'
    >>> stdset = getStandardSet(referenceDir)
    >>> len(stdset)
    0
    >>> os.mkdir(referenceDir)
    >>> for i in range(0, 10):
    ...   arr = array([ [0, 0, 0], [0, 255, 0], [0, 0, 0] ])
    ...   dumpImage(arr, '%s/ref%d.jpg' % (referenceDir, i), doLog=False)
    ...
    >>> stdset = getStandardSet(referenceDir)
    >>> len(stdset)
    10
    >>> for i in stdset:
    ...   print type(i)
    ...
    <type 'instance'>
    <type 'instance'>
    <type 'instance'>
    <type 'instance'>
    <type 'instance'>
    <type 'instance'>
    <type 'instance'>
    <type 'instance'>
    <type 'instance'>
    <type 'instance'>
    >>> for i in range(0, 10):
    ...   os.unlink('%s/ref%d.jpg' % (referenceDir, i))
    ...
    >>> os.rmdir(referenceDir)
    >>> stdset = getStandardSet(referenceDir)
    >>> len(stdset)
    0
    """
    if not os.path.exists(os.path.abspath(standard)):
        return []

    files = glob.glob( os.path.abspath(standard) + '/*jpg' );

    def mapFileToImage(x):
        im = Image.open( x );
        return im.convert( "L" );

    return map(mapFileToImage, files)

def main():
    #
    # Get the command line options
    #
    oParser = OptionParser();
    oParser.add_option("-d","--directory",dest="directory", help="specify a directory");
    oParser.add_option("-s","--standard",dest="standard", help="specify a standard set");

    (options, args) = oParser.parse_args();

    #
    # Check that the directory was passed as a parameter
    #
    if( options.directory is None ):
        print "Please provide a directory.";
        return 1;

    if( options.standard is None ):
        print "Please provide a standard image set.";
        return 1;

    log = open( "log.csv", "w" );
    log.write( "file,real,imaginary,x,y\n" );

    stds = getStandardSet( options.standard );

    for h in range(0,23):
        #
        # Get all jpeg files in the target dir.
        #
        globTarget = '*%02d0000.jpg' % h;

        # Get all the jpeg images
        files = glob.glob( options.directory + globTarget );
        # Sort all the files
        files.sort();

        if ( len(files) == 0 ):
            continue;

        print "Processing %d files at %02d00 hrs." % ( len( files ), h );

        dirname = '%sshifted%02d' % (options.directory, h);
        if ( not os.path.exists( dirname ) ):
            # Make the folder for the output images
            os.mkdir(dirname);

        log.write( "Hour %d\n" % h );

        #
        # Iterate through all files globbed
        #
        for curFile in files:
            #
            # Compare the images
            #
            shift = compareImages( stds[h], curFile, log );
            shift.save( '%sshifted%02d/%s' % (options.directory, h, os.path.basename(curFile) ), \
                "JPEG" );

    log.close();

if(__name__ == '__main__'):
    main();
