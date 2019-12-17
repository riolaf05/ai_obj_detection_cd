
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ElementTree

# 1) load the annotation file
# function to extract bounding boxes from an annotation file
def extract_boxes(filename):
	# load and parse the file
	tree = ElementTree.parse(filename)
	# get the root of the document
	root = tree.getroot()
	# extract each bounding box
	boxes = list()
	for box in root.findall('.//bndbox'):
		xmin = int(box.find('xmin').text)
		ymin = int(box.find('ymin').text)
		xmax = int(box.find('xmax').text)
		ymax = int(box.find('ymax').text)
		coors = [xmin, ymin, xmax, ymax]
		boxes.append(coors)
	# extract image dimensions
	width = int(root.find('.//size/width').text)
	height = int(root.find('.//size/height').text)
	return boxes, width, height

filelist = [f for f in listdir('/home/scripts/kangaroo/annots') if isfile(join('/home/scripts/kangaroo/annots', f))]

boxes={}

for filename in filelist:
    boxes[filename] = []
    boxes[filename].append(extract_boxes(filename))

# 2) 
#The mask-rcnn library requires that train, validation, and test datasets be managed by a mrcnn.utils.Dataset object.
#This means that a new class must be defined that extends the mrcnn.utils.Dataset class and defines a function to load the dataset

# class that defines and loads the kangaroo dataset
class KangarooDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "kangaroo")
		# define data locations
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip bad images
			if image_id in ['00090']:
				continue
			# skip all images after 150 if we are building the train set
			if is_train and int(image_id) >= 150:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
 
	# load the masks for an image
	def load_mask(self, image_id):
		# ...
 
	# load an image reference
	def image_reference(self, image_id):
		# ...


# prepare the dataset
train_set = KangarooDataset()
train_set.load_dataset(...)
train_set.prepare()

