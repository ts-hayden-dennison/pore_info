import mdtraj
import numpy
import scipy
from scipy.spatial.transform import Rotation
import math
import sys
import csv
from shapely.geometry import Polygon, LinearRing, Point, LineString
from shapely.ops import polylabel
from shapely.validation import make_valid
import shapely
from PIL import Image, ImageDraw
import argparse
import os


# given a plane expressed by a point and a normal vector, returns array of
# vertices projected onto the plane
# from https://stackoverflow.com/questions/35656166/projecting-points-onto-a-plane-given-by-a-normal-and-a-point
def planeprojection(normalvector, centroid, vertices):
    shape = vertices.shape #shape of vertex array, can be one vertex or multiple vertices to project
    if len(shape)==1:#meaning there is only one vertex
        vertex = vertices
        #dot product of position vector to the vertex from plane and normal vector
        dotscalar = numpy.dot(numpy.subtract(vertex, centroid), normalvector)
        #now returning the position vector of the projection onto the plane
        return numpy.subtract(vertex,dotscalar*normalvector)
    else:
        #array to store projectedvectors
        projectedvectors = numpy.zeros((shape[0],shape[1]))
        #now projecting onto plane, one by one
        for counter in range(shape[0]):
            vertex = vertices[counter,:]
            dotscalar = numpy.dot(numpy.subtract(vertex, centroid), normalvector)
            #now returning the position vector of the projection onto the plane
            projectedvectors[counter,:] = numpy.subtract(vertex, dotscalar*normalvector)
        #now returning the vectors projected
        return projectedvectors

def draw_scaled(geom, output, fnumber, i=0):
	size = 500
	margin = 50
	black = (0, 0, 0)
	white = (255, 255, 255)
	img = Image.new('RGB', (size, size), white)
	im_px_access = img.load()
	draw = ImageDraw.Draw(img)
	
	# need to scale pore to the proper size for the image
	# to do this we need to multiply the points by a transformation matrix that scales
	# and then translates them to match the coordinate system of PIL
	#bounds is minx, miny, maxx, maxy
	#for shape in lr:
	xscale = (size-margin*2)/(geom.bounds[2]-geom.bounds[0])
	yscale = (size-margin*2)/(geom.bounds[3]-geom.bounds[1])
	lr_scaled = shapely.affinity.scale(geom, xfact=xscale, yfact=yscale, origin='center')
	xshift = margin-(lr_scaled.bounds[0])
	yshift = margin-(lr_scaled.bounds[1])
	lr_translated = shapely.affinity.translate(lr_scaled, xoff=xshift, yoff=yshift)
	
	#print(CZr)
	#print(CZr_projected)
	#print(CZr_projected_aligned)
	#print(zprime)
	#print(math.degrees(angle))
	newpoints = []
	if lr_translated.geom_type in ['LinearRing', 'MultiPolygon', 'LineString']:
		for point in list(lr_translated.coords):
			newpoints.append((int(point[0]), int(point[1])))
	elif lr_translated.geom_type == 'Polygon':
		for point in list(lr_translated.exterior.coords):
			newpoints.append((int(point[0]), int(point[1])))
	else:
		print(lr_translated.geom_type)
		raise ValueError('{} is not a valid shapely geometry to draw'.format(lr_translated.geom_type))
	#print(newpoints)
	draw.line(newpoints, width=5, fill=black)
	name = '{}_poreseg_f{}_{}.png'.format(output, fnumber, i)
	img.save(name)
	data_dir = '{}_poreseg_out'.format(args.output)
	os.rename(name, os.path.join(data_dir, name))
	return

def pore_numbers_3mt6(trajfile, topfile, chainlist, fskip, output):
	# for each frame calculate the pore width and other variables
	# assumes that indices denotes the ARG residues that are members of the pores
	# angledata shape is (chainID, frames, 2) (theta angle, azimuthal angle)
	# poredata shape is (frame, 3) (pore area, pore width, pore angle)
	top = mdtraj.load(topfile).topology
	traj = mdtraj.load_xtc(trajfile, top)
	angledata = []
	for i in range(0, len(chainlist)):
		angledata.append([])
	poredata = []
	chains = ' '.join([str(x) for x in chainlist])
	
	CAs = top.select('name CA and residue 15 and resname ARG and chainid ' + chains)
	
	CZs = top.select('name CZ and residue 15 and resname ARG and chainid ' + chains)
	imgs = []
	for fnumber, frame in enumerate(traj):
		poredata.append([0, 0, 0])
		CAr = frame.xyz[0][CAs]
		CZr = frame.xyz[0][CZs]
		#print(CAr)
		
		# find best fitting center between all the alpha carbons of ARG15s, which denotes the pore center
		# (is placed inside array, which is why [0])
		porecenter = numpy.mean(CZr, axis=0, keepdims=True)[0]
		
		# find the singular value decomposition of the CZr with the centroid removed
		u, s, vh = numpy.linalg.svd(CZr - porecenter, full_matrices=True)
		
		
		# extract the best fitting right singular vector which is the normal of the plane
		# this is the last column of v and is guaranteed to be orthogonal to all other dimensional vectors of CZr and v
		# need to transpose vh first to get v
		# then get the rightmost column ([:,-1] means every row, last column in matrix read from left to right)
		zprime = numpy.transpose(vh)[:, -1]
		
		# find the centroid of the protein
		# (is placed inside array, which is why [0])
		center = numpy.mean(frame.xyz[0], axis=0, keepdims=True)[0]
		
		# if the porecenter+zprime is closer to the protein center than porecenter, zprime needs to be flipped to point
		# away from the decomposition chamber (out of the protein)
		# ord=2 denotes Euclidian norm, or length of the vector
		if numpy.linalg.norm((porecenter+zprime)-center, ord=2) < numpy.linalg.norm(porecenter-center, ord=2):
			zprime = zprime*-1
		
		#print(zprime)
		# find the projections of all CAr and CZr onto the plane of the pore along the zprime axis
		CAr_projected = planeprojection(zprime, porecenter, CAr)
		#print(CZr)
		CZr_projected = planeprojection(zprime, porecenter, CZr)
		#print(CZr_projected)
		
		# these points are not necessarily aligned with the xyz axes
		# (although many times they are approximately aligned with the x axis meaning the x axis can pass through the center of the pore,
		# due to alignment of the entire protein during simulation setup)
		# so rotate them in such a way that the zprime axes is pointing directly out of the screen
		# which corresponds to looking at the pore from the outside of the protein
		
		
		zprimemag = numpy.linalg.norm(zprime, ord=2)
		
		# 0, 0, 1 is pointing out of the screen in VMD
		angle = math.acos(numpy.dot(zprime, [0, 0, 1])/(zprimemag))
		#print(math.degrees(angle))
		
		# even though we have the angle we need an axis to rotate about
		# the cross product of the zprime and z axes can give us this axis because it is perpendicular to both
		rotaxis = numpy.cross(zprime, [0, 0, 1])
		
		# CZr are row vectors so initialize the rotation matrix with a row vector
		r = Rotation.from_rotvec(angle*rotaxis)
		
		# CZr are row vectors
		# apply() rotates each vector by the rotation matrix in turn
		# and returns a matrix of row vectors
		CZr_projected_aligned = r.apply(CZr_projected)
		#p = Polygon(CZr_projected_aligned)
		
		pore = make_valid(Polygon(CZr_projected_aligned))
		
		# p may be a multipolygon or some other object
		# so find the one with the largest area
		# and print the other areas
		
		if pore.geom_type == 'MultiPolygon':
			draw_scaled(Polygon(CZr_projected_aligned), output, fnumber)
			largest = 0
			p = None
			print()
			print('frame {} complex pore shape, taking largest area'.format(fnumber))
			for i, poreseg in enumerate(pore.geoms):
				print(poreseg.area)
				if poreseg.area > largest:
					p = poreseg
					largest = poreseg.area
				poreseg = LineString(poreseg.exterior.coords)
				draw_scaled(poreseg, output, fnumber, i+1)
			print('largest area: >>>{}<<<'.format(largest))
		else:
			print()
			print('frame {} recording full pore'.format(fnumber))
			p = pore
			print(p.area)
			print(p.geom_type)
		
		lr = LinearRing(p.exterior.coords)
		# find the largest inscribed circle of this coplanar set of points
		# the diameter of this circle is the pore width
		#Point of inaccessibility is the center of the largest inscribed circle
		poa = polylabel(p, tolerance=0.000001)
		
		#find closest distance to any point on the heptagon which is the radius of the largest inscribed circle
		pore_width = poa.distance(lr)*2
		
		#print("LIC: {}\nArea: {}\nDiameter: {}".format(poa, p.area, pore_width))
		poredata[-1][0] = p.area
		poredata[-1][1] = pore_width
		poredata[-1][2] = math.degrees(angle)
		
		# find all the theta angles
		for chainindex, alphaC in enumerate(CAr):
			r = alphaC - CZr[chainindex]
			# theta is dot product between vector from CA to CZ and zprime
			zprimemag = numpy.linalg.norm(zprime, ord=2)
			rmag = numpy.linalg.norm(r, ord=2)
			theta = math.degrees(math.acos(numpy.dot(zprime, r)/(zprimemag*rmag)))
			angledata[chainindex].append([theta, 0])
		
		# find azimuthal angles
		# use CAr_projected because we want yprime to be orthogonal to zprime
		for chainindex, alphaC in enumerate(CAr_projected):
			yprime = alphaC-porecenter
			# original paper had the xprime axes being yprime cross zprime
			xprime = numpy.cross(yprime, zprime)
			
			r = CZr_projected[chainindex]-alphaC
			
			xprimemag = numpy.linalg.norm(xprime, ord=2)
			rmag = numpy.linalg.norm(r, ord=2)
			azimuth = math.degrees(math.acos(numpy.dot(xprime, r)/(xprimemag*rmag)))
			#add azimuth to newest frame
			angledata[chainindex][-1][-1] = azimuth
		
		
		size = 500
		margin = 50
		if fnumber%fskip == 0:
			black = (0, 0, 0)
			white = (255, 255, 255)
			img = Image.new('RGB', (size, size), white)
			im_px_access = img.load()
			draw = ImageDraw.Draw(img)
			
			# need to scale pore to the proper size for the image
			# to do this we need to multiply the points by a transformation matrix that scales
			# and then translates them to match the coordinate system of PIL
			#bounds is minx, miny, maxx, maxy
			xscale = (size-margin*2)/(lr.bounds[2]-lr.bounds[0])
			yscale = (size-margin*2)/(lr.bounds[3]-lr.bounds[1])
			lr_scaled = shapely.affinity.scale(lr, xfact=xscale, yfact=yscale, origin='center')
			xshift = margin-(lr_scaled.bounds[0])
			yshift = margin-(lr_scaled.bounds[1])
			lr_translated = shapely.affinity.translate(lr_scaled, xoff=xshift, yoff=yshift)
			
			#print(CZr)
			#print(CZr_projected)
			#print(CZr_projected_aligned)
			#print(zprime)
			#print(math.degrees(angle))
			newpoints = []
			for point in list(lr_translated.coords):
				newpoints.append((int(point[0]), int(point[1])))
			#print(newpoints)
			draw.line(newpoints, width=5, fill=black)
			#origin = (lr_translated.bounds[0], lr_translated.bounds[1])
			#poa_scaled = shapely.affinity.scale(poa, xfact=xscale, yfact=yscale, origin='center')
			#poa_translated = shapely.affinity.translate(poa_scaled, xoff=xshift, yoff=yshift)
			poa_translated = polylabel(Polygon(lr_translated), tolerance=1)
			x, y = poa_translated.coords[0]
			draw.point((x, y), black)
			d = poa_translated.distance(lr_translated)
			rect = (x-d, y-d, x+d, y+d)
			draw.arc(rect, 0, 359, fill=(255, 0, 0), width=2)
			imgs.append(img)
		
	return angledata, poredata, imgs

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Calculate key data of 3mt6/1yg6 pores")
	parser.add_argument('-f', '--trajectory', type=str, help='The trajectory file to analyze', required=True)
	parser.add_argument('-s', '--topology', type=str, help='The topology file to use', required=True)
	parser.add_argument('-c', '--chains', type=int, help='The chain indexes to pick from', nargs='+', required=True)
	parser.add_argument('-o', '--output', type=str, help='The prefix of the output files', default='3mt6')
	parser.add_argument('-fs', '--frameskip', type=int, help='Every frameskip frames, output a picture of the pore', \
						default=10)
	args = parser.parse_args()
	
	data_dir = '{}_out'.format(args.output)
	try:
		os.mkdir(data_dir)
	except FileExistsError:
		pass
	try:
		os.mkdir('{}_poreseg_out'.format(args.output))
	except FileExistsError:
		pass
	angledata, poredata, imgs = pore_numbers_3mt6(args.trajectory, args.topology, args.chains, args.frameskip, args.output)
	
	for i, img in enumerate(imgs):
		name = '{}_frame_{}.png'.format(args.output, i*args.frameskip)
		img.save(name)
		os.rename(name, os.path.join(data_dir, name))
		
	
	# Write angle data for each chain
	with open('{}_angles.csv'.format(args.output), 'w') as f:
		write = csv.writer(f)
		for index, chain in enumerate(angledata):
			write.writerow(['Chain ID {}'.format(args.chains[index])])
			write.writerow(['Frame', 'Theta', 'Azimuth'])
			for frameindex, frame in enumerate(chain):
				write.writerow([frameindex, *tuple(frame)])
	os.rename('{}_angles.csv'.format(args.output), os.path.join(data_dir, '{}_angles.csv'.format(args.output)))
	
	
	# Write pore width and diameter
	with open('{}_pore.csv'.format(args.output), 'w') as f:
		write = csv.writer(f)
		write.writerow(['Frame', 'Pore Area', 'Pore Width', 'Pore Angle'])
		for frameindex, frame in enumerate(poredata):
			write.writerow([frameindex, *tuple(frame)])
	os.rename('{}_pore.csv'.format(args.output), os.path.join(data_dir, '{}_pore.csv'.format(args.output)))
	
	
	sys.exit()
#
