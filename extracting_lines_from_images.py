import cv2
import io
import numpy

from PIL import Image

INPUT_IMAGES='/tmp/images/*.png'

'''
This is the scalable framework for the Line coordinates from images.
Reference : https://github.com/kumardeepak/image_shape_extraction

'''
def extract_line_coords(image):
  name, img = image
  pil_image = Image.open(io.BytesIO(img)).convert('RGB') 
  open_cv_image = numpy.array(pil_image) 
  open_cv_image = open_cv_image[:, :, ::-1].copy() 
  gray     = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
  MAX_THRESHOLD_VALUE     = 255
  BLOCK_SIZE              = 15
  THRESHOLD_CONSTANT      = 0
  SCALE                   = 15
  # Filter image
  filtered                = cv2.adaptiveThreshold(~gray, MAX_THRESHOLD_VALUE, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, THRESHOLD_CONSTANT)
  horizontal              = filtered.copy()
  horizontal_size         = int(horizontal.shape[1] / SCALE)
  horizontal_structure    = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
  # isolate_lines
  cv2.erode(horizontal, horizontal_structure, horizontal, (-1, -1)) # makes white spots smaller
  cv2.dilate(horizontal, horizontal_structure, horizontal, (-1, -1))

  contours                = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours                = contours[0] if len(contours) == 2 else contours[1]
  lines                   = []
  for index, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    if w > 50:
      lines.append((x,y,w,h))
  return os.path.basename(name), lines



images_rdd = spark.sparkContext.binaryFiles(INPUT_IMAGES)
line_coord_df = images_rdd.map(lambda img: extract_line_coords(img)).toDF()
line_coord_df.show(20, False)


# +----------+---------------------+
# |_1        |_2                   |
# +----------+---------------------+
# |out023.png|[]                   |
# |out037.png|[]                   |
# |out036.png|[]                   |
# |out022.png|[[108, 1130, 216, 1]]|
# |out008.png|[]                   |
# |out034.png|[[108, 1130, 216, 1]]|
# |out020.png|[[108, 1037, 216, 1]]|
# |out021.png|[[108, 1130, 216, 1]]|
# |out035.png|[[108, 1130, 216, 1]]|
# |out009.png|[[108, 1130, 216, 1]]|
# +----------+---------------------+