"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple

import matplotlib.pyplot as plt

from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        # return homography
        """INSERT YOUR CODE HERE"""

        def RowsFromPoint(x,y, xPrime, yPrime):
            """
            this function recieves the coordinates of a 
            matching point in the source image, (x,y),
            and in the destination image, (xPrime,yPrime),
            and outputs the coefficents of the two equations
            derived from the point as two numpy arrays of length 9.

            after computing this function for all matching points,
            the resulting arrays can be stacked vertically to get a 
            2Nx9 matrix that can be used to find the homography
            using a SVD decomposition.
            """
            homographyRow = np.ones(9)
            homographyRow[0] = x
            homographyRow[1] = y
            homographyRow[3] = 0
            homographyRow[4] = 0
            homographyRow[5] = 0
            homographyRow[6] = -(x*xPrime)
            homographyRow[7] = -(y*xPrime)
            homographyRow[8] = -xPrime 

            homographyRow1 = np.ones(9)
            homographyRow1[0] = 0
            homographyRow1[1] = 0
            homographyRow1[2] = 0
            homographyRow1[3] = x
            homographyRow1[4] = y
            homographyRow1[6] = -(x*yPrime)
            homographyRow1[7] = -(y*yPrime)
            homographyRow1[8] = -yPrime

            return   [homographyRow,homographyRow1]
            
        rows = []    
        for pointX,pointY,xPrimeDst,yPrimeDst in zip(match_p_src[0],match_p_src[1],match_p_dst[0],match_p_dst[1]):
            rows += RowsFromPoint(pointX,pointY,xPrime=xPrimeDst,yPrime=yPrimeDst)
            
        
        eqMat = np.vstack(rows)
        u,sigma,v = np.linalg.svd(eqMat)
        
        return v[8].reshape(3,3)
        

        

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        """INSERT YOUR CODE HERE"""

        '''
        print(src_image.reshape(-1))
        print(src_image[0][0])
        '''
        #print(dst_image_shape)
        dst_image = np.zeros(dst_image_shape)

        
        for i in range(len(src_image[:])):
            for j in range(len(src_image[i][:])):
                new_pixel = np.matmul(homography,np.array([j,i,1]))

                dstY = round(new_pixel[0]/new_pixel[2])
                dstX = round(new_pixel[1]/new_pixel[2])
                
                if dstX > 0 and dstX < dst_image_shape[0] and dstY > 0 and dstY < dst_image_shape[1]:
                    dst_image[dstX][dstY]=src_image[i][j]/255
                    #print(src_image[i][j])
                #print(str(i)," ",str(j)," pixel color: ", src_image[i][j])

        
        '''
        for i in range(600,800):
            for j in range(800,1000):
                    print("adding")
                    dst_image[i][j]=dst_image[51][163]
        '''


        return dst_image
        
        pass

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        """INSERT YOUR CODE HERE"""

        dst_image = np.zeros(dst_image_shape) # destination image matrix

        src_image_shape = src_image.shape
        #print(src_image_shape)
        x_axis_coordinates = np.arange(src_image_shape[1])
        y_axis_coordinates = np.arange(src_image_shape[0])
        x_coords,y_coords = np.meshgrid(x_axis_coordinates,y_axis_coordinates)
        # both x_coords and y_coords are of shape(height,width), we reshape them
        x_coords,y_coords = x_coords.flatten(),y_coords.flatten()
        
        # now generate 3x(height*width) source coordinates matrix
        source_coordinates = np.vstack([x_coords,y_coords,np.ones_like(x_coords)])
        
        # multiply by homography to get destination coordinates, and then normalize by dividing
        # the first two rows by the third
        dest_coordinates = np.matmul(homography,source_coordinates)
        dest_non_homogeneous_coordinates = (dest_coordinates[:-1]/dest_coordinates[2]).round().astype(int)

        # boolean 1D array for column selection
        clipped_coordinates_mask = \
            np.all(dest_non_homogeneous_coordinates >= 0,axis=0) & \
            np.all(np.less(dest_non_homogeneous_coordinates,np.array([[dst_image_shape[1]],[dst_image_shape[0]]])),axis=0)
        # keep only pixels that are transformed inside the bounds of the dest image
        clipped_dst_coordinates = dest_non_homogeneous_coordinates[:,clipped_coordinates_mask]
        clipped_src_coordinates = source_coordinates[:-1][:,clipped_coordinates_mask]

        #print(clipped_src_coordinates)
        #transform pixels into dst_image from src_image
        data = src_image[clipped_src_coordinates[1],clipped_src_coordinates[0],:]
        #print(data.shape)
        dst_image[clipped_dst_coordinates[1],clipped_dst_coordinates[0],:] = data/255
        #print(dst_image.shape,dst_image.max(),dst_image.min(),dst_image.mean())
        return dst_image

        pass

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        # return fit_percent, dist_mse
        """INSERT YOUR CODE HERE"""

        def HammiltonDistance(x,y,xPrime,yPrime):
            return abs(x-xPrime) + abs(y-yPrime)

        perfect_match_x = []
        perfect_match_y = []
        perfect_match_dstX = []
        perfect_match_dstY = []
        #print(max_err)

        for x,y,dstX,dstY in zip(match_p_src[0],match_p_src[1],match_p_dst[0],match_p_dst[1]):
            new_pixel = np.matmul(homography,np.array([x,y,1]))

            true_new_pixel = [round(new_pixel[0]/new_pixel[2]),round(new_pixel[1]/new_pixel[2])]
            #print(HammiltonDistance(dstX,dstY,true_new_pixel[0],true_new_pixel[1]))

            if HammiltonDistance(dstX,dstY,true_new_pixel[0],true_new_pixel[1]) <= max_err:
                perfect_match_x += [true_new_pixel[0]]
                perfect_match_y += [true_new_pixel[1]]
                perfect_match_dstX += [dstX]
                perfect_match_dstY += [dstY]
                #print("point" + str(x) +" " + str(y))

        if len(perfect_match_dstX) == 0:
            return (0,10**9)

        distance = []
        for x,y,dstX,dstY in zip(perfect_match_x,perfect_match_y,perfect_match_dstX,perfect_match_dstY):
            distance += [HammiltonDistance(x,y,dstX,dstY)]
        #print(distance)

        mse = np.square(distance).mean()
        return (len(perfect_match_dstX)/len(match_p_src[0]),mse)

        pass

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        # return mp_src_meets_model, mp_dst_meets_model
        """INSERT YOUR CODE HERE"""


        # transform src points to dst points:
        homogeneous_src_coords = np.vstack([match_p_src,np.ones_like(match_p_src[0,:])])
        transformed_homogeneous_coords = np.matmul(homography,homogeneous_src_coords)
        final_dst_coords = (transformed_homogeneous_coords[:-1]/transformed_homogeneous_coords[2]).round().astype(int)

        # Note: we use Hamming distance as our distance function
        distances = np.abs(match_p_dst - final_dst_coords).sum(axis=0)
        meets = distances <= max_err

        mp_src_meets_model = match_p_src[:,meets]
        mp_dst_meets_model = match_p_dst[:,meets]


        return mp_src_meets_model, mp_dst_meets_model
        pass

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # # use class notations:
        w = inliers_percent
        t = max_err
        # # p = parameter determining the probability of the algorithm to
        # # succeed
        p = 0.99
        # # the minimal probability of points which meets with the model
        d = 0.5
        # # number of points sufficient to compute the model
        n = 4
        # # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        # return homography
        """INSERT YOUR CODE HERE"""


        iterations = 0
        bestFit = np.zeros((3,3))
        bestErr = 10**9

        for i in range(k):
            indecies= np.random.choice(match_p_src.shape[1], 4,  replace=False)
            
            calc_matrix = Solution.compute_homography_naive(match_p_src[:,indecies],match_p_dst[:,indecies])
            mat_percentage ,matrix_err = Solution.test_homography(calc_matrix,match_p_src,match_p_dst,max_err)
            if mat_percentage > d:
                srcInliers, dstInliers = Solution.meet_the_model_points(calc_matrix,match_p_src,match_p_dst,t)
                calc_matrix = Solution.compute_homography_naive(srcInliers,dstInliers)
                mat_percentage ,matrix_err = Solution.test_homography(calc_matrix,match_p_src,match_p_dst,max_err)
                if matrix_err < bestErr:
                    bestErr = matrix_err
                    bestFit = calc_matrix

        return bestFit

        pass

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        # return backward_warp
        """INSERT YOUR CODE HERE"""
        
        # generate pixel coordinates of destination image
        dst_meshgrid_x,dst_meshgrid_y = np.meshgrid(np.arange(dst_image_shape[1]),np.arange(dst_image_shape[0]))

        dst_pixel_coordinates_homogeneous = np.vstack([dst_meshgrid_x.flatten(),dst_meshgrid_y.flatten(),
            np.ones(dst_image_shape[1]*dst_image_shape[0])])
        print("dst_pixel_coordinates_homogeneous",dst_pixel_coordinates_homogeneous)
        src_points_coordinates_homogeneous = np.matmul(backward_projective_homography,dst_pixel_coordinates_homogeneous)

        # points on the source image plane that need to be sampled using linear interpolation
        src_points_coordinates = (src_points_coordinates_homogeneous[:-1]/src_points_coordinates_homogeneous[2])
        print("src_points_coordinates",src_points_coordinates)

        src_image_shape = src_image.shape
        src_meshgrid_x,src_meshgrid_y = np.meshgrid(np.arange(src_image_shape[1]),np.arange(src_image_shape[0]))
        src_pixel_coordinates = np.vstack([src_meshgrid_x.flatten(),src_meshgrid_y.flatten()])
        src_pixel_coordinates_transposed = src_pixel_coordinates.transpose()
        # need to interpolate every color channel separatly
        print("src_points_coordinates",src_pixel_coordinates)

        dst_shape_2d = (dst_image_shape[0],dst_image_shape[1])

        print("interpolating red")
        red_values = griddata(src_pixel_coordinates_transposed,
            src_image[src_pixel_coordinates[1],src_pixel_coordinates[0],0],
            (src_points_coordinates[0].reshape(dst_shape_2d),src_points_coordinates[1].reshape(dst_shape_2d)),
            method='linear',fill_value = 15)
        print("interpolating green")
        green_values = griddata(src_pixel_coordinates_transposed,
            src_image[src_pixel_coordinates[1],src_pixel_coordinates[0],1],
            (src_points_coordinates[0].reshape(dst_shape_2d),src_points_coordinates[1].reshape(dst_shape_2d)),
            method='linear',fill_value = 15)
        print("interpolating blue")
        blue_values = griddata(src_pixel_coordinates_transposed,
            src_image[src_pixel_coordinates[1],src_pixel_coordinates[0],2],
            (src_points_coordinates[0].reshape(dst_shape_2d),src_points_coordinates[1].reshape(dst_shape_2d)),
            method='linear',fill_value = 15)
        print(blue_values.max(),red_values.max(),green_values.max())
        interpolated_image = np.dstack([red_values,green_values,blue_values])

        print(interpolated_image.mean())
        print(interpolated_image.max())

        #backward_warp = np.zeros(dst_image_shape)
        
        #backward_warp[src_pixel_coordinates[1],src_pixel_coordinates[0],:] = interpolated_image[src_pixel_coordinates[1],src_pixel_coordinates[0],:]

        return interpolated_image

        pass

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # return final_homography
        """INSERT YOUR CODE HERE"""

        translation = np.array([
            [1,0,-pad_left],
            [0,1,-pad_up],
            [0,0,1]
        ])

        final_homography = np.matmul(backward_homography,translation)
        return final_homography

        pass

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # return np.clip(img_panorama, 0, 255).astype(np.uint8)
        """INSERT YOUR CODE HERE"""

        forward_homography = self.compute_homography(
                                    match_p_src,
                                    match_p_dst,
                                    inliers_percent,
                                    max_err)
        height,width,pads = Solution.find_panorama_shape(src_image,dst_image,forward_homography)    

        final_panorama = np.zeros((height,width,3))
    
        backward_homography = self.compute_homography(
                                    match_p_dst,
                                    match_p_src,
                                    inliers_percent,
                                    max_err)

        translated_backwards_homography = Solution.add_translation_to_backward_homography(
            backward_homography,pads.pad_left,pads.pad_up)
        
        # creates an image of the src matrix warped into the dst image
        src_warp = Solution.compute_backward_mapping(translated_backwards_homography,src_image,final_panorama.shape)

        print(src_warp)
        print(src_warp.shape)
        print(src_warp.mean())
        print(src_warp.max())

        plt.figure()
        pplot = plt.imshow(src_warp/255)
        plt.title('WARPED')
        # plt.show()
        plt.show()


        # copy destination image into final panorama
        dst_image_shape = dst_image.shape
        final_panorama[pads.pad_up:pads.pad_up + dst_image_shape[0],pads.pad_left:pads.pad_left + dst_image_shape[1],:] = dst_image

        # copy warped src image into final panorama 
        src_warp[pads.pad_up:pads.pad_up + dst_image_shape[0],pads.pad_left:pads.pad_left + dst_image_shape[1],:] = 0
        final_panorama += src_warp
        
        return final_panorama/255


        pass
