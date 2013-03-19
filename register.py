#!/usr/bin/env python

import glob;
import os;
from numpy import *;
from numpy.fft import *;
from PIL import Image;
from PIL import ImageChops;
from optparse import OptionParser;

def dumpImage( arr, name, doLog=True ):
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

	
#
# A method for comparing two images.
#
def compareImages( fixedImage, move, log ):
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
	files = glob.glob( standard + "*jpg" );

	stds = [];
	
	for f in sort(files):
		im = Image.open( f );
		im = im.convert( "L" );
		stds.append( im );

		#blank = Image.new( "L", im.size );

		#box1 = [ 64, 128, 114, 178 ];
		#region1 = im.crop( box1 );

		#blank.paste( region1, box1 );

		#stds.append( blank );
		
	return stds;

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
			os.spawnlp(os.P_WAIT, 'mkdir', 'mkdir', dirname );

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
