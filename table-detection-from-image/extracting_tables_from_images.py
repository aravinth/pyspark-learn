'''
This is the scalable framework for identifying the table details from images.
Reference : https://github.com/kumardeepak/hw-recog-be

'''
import cv2
import io
import numpy
import os
from PIL import Image
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# Input dir containing the images (jpeg/png)
INPUT_IMAGES = '/Users/TIMAC044/Documents/Anuvaad/table-detection/*.png'
# Util class for identifying a rectangle
RECT_UTIL    = '/Users/TIMAC044/Documents/Anuvaad/table-detection/rect.py'
# Util class for identifying a rectangle
TABLE_UTIL   = '/Users/TIMAC044/Documents/Anuvaad/table-detection/table.py'

spark = SparkSession \
    .builder \
    .appName("Extract Table details from Images") \
    .getOrCreate()

spark.sparkContext.addPyFile(RECT_UTIL)
spark.sparkContext.addPyFile(TABLE_UTIL)


def extract_table_coords(image):
  from rect import RectRepositories
  from table import TableRepositories
  name, img       = image
  pil_image       = Image.open(io.BytesIO(img)).convert('RGB') 
  open_cv_image   = numpy.array(pil_image) 
  open_cv_image  = open_cv_image[:, :, ::-1].copy() 
  Rects           = RectRepositories(open_cv_image)
  lines, _        = Rects.get_tables_and_lines ()
  table           = None
  TableRepo       = TableRepositories(open_cv_image, table)
  tables          = TableRepo.response ['response'] ['tables']
  lines           = []
  for table in tables:
    base_x = int(table.get('x'))
    base_y = int(table.get('y'))
    for t in table.get('rect'):
      x = base_x + int(t['x'])
      y = base_y + int(t['y'])
      w = int(t['w'])
      h = int(t['h'])
      row = int(t['row'])
      col = int(t['col'])
      lines.append((row, col, x, y, w, h))
  return os.path.basename(name), lines


# Read the images
IMAGES_RDD           = spark.sparkContext.binaryFiles(INPUT_IMAGES)

# For each of the images, extract the coordinates along with row & column
TABLE_COORD_DF       = IMAGES_RDD.map(lambda img: extract_table_coords(img)).toDF()

# Explode the cells (one row per cell)
EXPLODED_DF          = TABLE_COORD_DF.select(F.col("_1").alias("filename"), \
                                         F.explode(F.col("_2")).alias("coord"))
# Final view
FINAL_TABLE_COORD_DF = EXPLODED_DF.select("filename", \
                                    F.col("coord._1").alias("row"), \
                                    F.col("coord._2").alias("col"), \
                                    F.col("coord._3").alias("x"), \
                                    F.col("coord._4").alias("y"), \
                                    F.col("coord._5").alias("w"), \
                                    F.col("coord._6").alias("h"))

FINAL_TABLE_COORD_DF.show(500, False)

#                 SAMPLE OUTPUT
#                 ~~~~~~~~~~~~~
#  +-----------------+---+---+----+---+---+---+
#  |filename         |row|col|x   |y  |w  |h  |
#  +-----------------+---+---+----+---+---+---+
#  |sample_table2.png|0  |0  |3   |2  |64 |79 |
#  |sample_table2.png|0  |1  |67  |2  |106|79 |
#  |sample_table2.png|0  |2  |172 |2  |106|80 |
#  |sample_table1.png|0  |0  |3   |3  |57 |46 |
#  |sample_table1.png|0  |1  |60  |3  |407|46 |
#  |sample_table1.png|0  |2  |467 |3  |383|46 |
#  |sample_table1.png|0  |3  |850 |3  |107|46 |
#  +-----------------+---+---+----+---+---+---+
