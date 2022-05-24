import numpy as np

# we may assume SIDE > blur_side 
SIDE = 100

blur_shift = 1
blur_side = 2*blur_shift+1
number_of_neighbors = blur_side**2

dog_translucency = 0.9 

... # code to load of `cow` and `dog` arrays from image files:
    #   `cow` is a grayscale SIDE x SIDE integer array.
    #   `dog` is a color SIDE x SIDE x 3 integer arrays.
    # The arrays' axes represent SIDE many pixel rows times SIDE pixel columns
    # times grayscale intensity or {Red,Green,Blue} intensities.  The
    # intensities range 0 through 255.

# the code below is supposed to make `blurdog`'s pixel at row r and column c
# the average of the `dog`'s pixels that neighbor `dog`'s pixels at row r and
# column c.
blurdog = 0 
for row_shift in range(-blur_shift, blur_shift+1):
  for col_shift in range(-blur_shift, blur_shift+1):
    blurdog += dog[0+row_shift:SIDE+row_shift,
                   0+col_shift:SIDE+col_shift ] / number_of_neighbors 

# the code below is supposed to make: the bigger `dog_translucency` is, the
# more similar `cowdog` should be to `cow`.
cowdog = cow + dog_translucency * (blurdog-cow) 

... # code to save cowdog as an image 


blurdog = 0
blurdog = np.convolve(dog, np.ones((blur_side, blur_side)), 'valid')
blurdog /= float(number_of_neighbors) # cast is unneeded but clarifies intent 
cowdog = cow[blur_shift:-blur_shift,
             blur_shift:-blur_shift,np.newaxis].astype(np.float32)
cowdog += (1-dog_translucency) * (blurdog-cowdog) 

blurdog = 0
for row_shift in range(blur_side):
  for col_shift in range(blur_side):
    blurdog += dog[row_shift:SIDE,
                   col_shift:SIDE ]
blurdog /= float(number_of_neighbors) # cast is unneeded but clarifies intent 
cowdog = cow[:,:,np.newaxis].astype(np.float32)
cowdog += dog_translucency * (cowdog-blurdog) 

blurdog = np.zeros((SIDE-blur_shift,SIDE-blur_shift,3), dtype=np.float32) 
for row_shift in range(-blur_shift, blur_shift+1):
  for col_shift in range(-blur_shift, blur_shift+1):
    blurdog += dog[blur_shift:-blur_shift,
                   blur_shift:-blur_shift ]
blurdog /= float(number_of_neighbors) # cast is unneeded but clarifies intent 
cowdog = cow[:,:,np.newaxis].astype(np.float32)
cowdog += dog_translucency * (cowdog-blurdog) 

blurdog = np.zeros((SIDE-blur_shift,SIDE-blur_shift,3), dtype=np.float32) 
for row_shift in range(-blur_shift, blur_shift+1):
  for col_shift in range(-blur_shift, blur_shift+1):
    blurdog += dog[blur_shift+row_shift:SIDE-blur_shift+row_shift,
                   blur_shift+col_shift:SIDE-blur_shift+col_shift ]
blurdog /= float(number_of_neighbors) # cast is unneeded but clarifies intent 
cowdog = cow[blur_shift:-blur_shift,
             blur_shift:-blur_shift,np.newaxis].astype(np.float32)
cowdog += (1-dog_translucency) * (blurdog-cowdog) 

#.  To fix this, we'd want to initialize `blurdog` to a (SIDE-2) x (SIDE-2) x 3
# floating point array, then in the loop body replace "0" by "1" and "SIDE" by
# "1-SIDE".  Then to   We'd also want to replace dog_translucency in the last
# line by (1-dog_translucency), for semantic correctness.  
