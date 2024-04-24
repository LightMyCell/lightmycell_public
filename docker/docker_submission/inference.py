#import sys
#sys.path.append('/home/local-admin/lightmycells/Label_free_cell_painting')

import os
from mynetwork import UNet
import torch
import torch.optim 
import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image
import sys, glob
import re
#from aicsimageio import AICSImage
from torch.utils.data import DataLoader
import tables
#from Utils.trainUtils import LoadingDatasetTest
#from scipy import ndimage
from pathlib import Path
from os.path import join, isdir, basename
from os import mkdir, listdir
import xmltodict
import tifffile

INPUT_PATH = Path("/input") # add / back
OUTPUT_PATH = Path("/output") # add / back  
if not isdir(join(OUTPUT_PATH,"images")): mkdir(join(OUTPUT_PATH,"images"))
resize_dimensions = [1500]
RESOURCE_PATH = Path("resources")

def read_image(location):
    # Read the TIFF file and get the image and metadata
    with tifffile.TiffFile(location) as tif:
        image_data = tif.asarray()    # Extract image data
        metadata   = tif.ome_metadata # Get the existing metadata in a DICT
    return image_data, metadata

def save_image(*, location, array,metadata):
    #Save each predicted images with the required metadata
    print(" --> save "+str(location))
    pixels = xmltodict.parse(metadata)["OME"]["Image"]["Pixels"]
    physical_size_x = float(pixels["@PhysicalSizeX"])
    physical_size_y = float(pixels["@PhysicalSizeY"])
    
    
    tifffile.imwrite(location,
                     array,
                     description=metadata.encode(),
                     resolution=(physical_size_x, physical_size_y),
                     metadata=pixels,
                     tile=(128, 128),
                     )

def run():
	# Load the saved model weights e.g. 28th epoch of trained WGAN model
	dataname = "patches_s_3"                            # naming of save states
	epoch_num = 23                                       # what epoch to load
	#pytable_name = "study_patches_d"                    # name of pytable with test data (without train/valid/test phase in name)
	checkpoint = torch.load(f'./resources/{dataname}_epoch_{epoch_num}_Unet.pth') 
	output_path = "."

	# Network/Training Parameters (copied from training)
	ignore_index = 0 
	gpuid=0
	n_classes= 4
	in_channels= 1
	padding= True
	depth= 6
	wf= 5 
	up_mode= 'upconv' 
	batch_norm = False
	batch_size=1
	patch_size=256
	edge_weight = 1.1 
	phases = ["train","val"] 
	validation_phases= ["val"] 

	# Specify if we should use a GPU (cuda) or only the CPU
	if(torch.cuda.is_available()):
		print(torch.cuda.get_device_properties(gpuid))
		torch.cuda.set_device(gpuid)
		device = torch.device(f'cuda:{gpuid}')
	else:
		device = torch.device(f'cpu')

	# Define the network
	Gen = UNet(n_classes=n_classes, in_channels=in_channels, padding=padding,depth=depth,wf=wf, up_mode=up_mode, batch_norm=batch_norm).to(device)
	print(f"total params: \t{sum([np.prod(p.size()) for p in Gen.parameters()])}")
	Gen.load_state_dict(checkpoint['model_dict'])
	Gen.eval()



	#dataset["test"]=LoadingDatasetTest(f"pytables/{pytable_name}_valid.pytable")                    # test !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	#dataLoader["test"]=DataLoader(dataset["test"], batch_size=batch_size, 
	#                                    shuffle=False, num_workers=0, pin_memory=False)

	#for ii, (X, y) in enumerate(dataLoader["test"]): #for each of the batches
	transmitted_light_path = join(INPUT_PATH , "images","organelles-transmitted-light-ome-tiff")
	    
	for input_file_name in listdir(transmitted_light_path):
		if input_file_name.endswith(".tiff"):
		
			print(" --> Predict " + input_file_name)
			image_input,metadata=read_image(join(transmitted_light_path,input_file_name))

				# Get the type of transmited liht (BF, DIC, PC)
			description = xmltodict.parse(metadata)
			tl= description["OME"]['Image']["Pixels"]["Channel"]["@Name"]
		
			image_input = image_input.astype('float32')

			gt_height, gt_width = image_input.shape[-2:]							
			if gt_height > resize_dimensions[0] or gt_width > resize_dimensions[0]:
					img = Image.fromarray(image_input)
					image_input = img.resize((resize_dimensions[0], resize_dimensions[0]))
						
			image_input = np.expand_dims(image_input, axis=0)
			image_input = np.expand_dims(image_input, axis=0)
			img = image_input.copy()
			image_input = torch.from_numpy(img)

			x_in = image_input
			x_in = x_in.to(device)
			prediction1 = Gen(x_in)

			organelle = ["Nucleus", "Mitochondria", "Actin", "Tubulin"]
			for channel in range(n_classes): 
				checkfull = prediction1[0,channel,:,:]
				checkfull_cpu = checkfull.cpu()
				prediction = checkfull_cpu.detach().numpy()
				#gt = y[0][channel].numpy()

				normalized_prediction = (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction))
				prediction = (normalized_prediction * 65535).astype(np.uint16)

										
				if (prediction.shape[-1] < gt_width) or (prediction.shape[-2] < gt_height):
					pred = Image.fromarray(prediction)
					prediction = np.array(pred.resize((gt_width, gt_height)))
					
				output_organelle_path = join(OUTPUT_PATH, "images", organelle[channel].lower() + "-fluorescence-ome-tiff")
				if not isdir(output_organelle_path):  mkdir(output_organelle_path)
				save_image(location=join(output_organelle_path,basename(input_file_name)), array=prediction,metadata=metadata)
				
				# directory_gt = output_path
				# os.makedirs(directory_gt, exist_ok=True)
				# gtim = Image.fromarray(gt)
				# gtim.save(f"{directory_gt}/Unet_gt_epoch_{epoch_num}_channel_{channel}_img_{dataname}.tif")
	return 0
	
if __name__ == "__main__":
    raise SystemExit(run())
