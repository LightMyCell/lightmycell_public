import tables
from skimage.metrics import structural_similarity as ssim
#from skimage.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean, cosine
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class EarlyStopper:                                                                 #https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class LoadingDatasetTrain(object):
    def __init__(self, fname ,img_transform=None, mask_transform = None, edge_weight= False):
        self.fname=fname
        self.edge_weight = edge_weight
        self.img_transform=img_transform
        self.mask_transform = mask_transform
        self.tables=tables.open_file(self.fname)
        #self.numpixels=self.tables.root.numpixels[:]
        self.nitems=np.array(self.tables.root.total_elements)[0]
        self.tables.close()
        
        self.img = None
        self.mask = None
        
    def __getitem__(self, index):
        with tables.open_file(self.fname,'r') as db:
            self.img=db.root.img.data
            self.mask=db.root.mask.data
            mask = self.mask[index,:,:,:]
            img = self.img[index,:,:]
            
            img_new = np.expand_dims(img, axis=0).astype('float32')
            mask_new = mask.astype('float32')
        
        #print(img_new.shape)
        #print(img_new)
        #print(img_new.dtype)
        #print(mask_new.shape)
        #print(mask_new)
        #print(mask_new.dtype)

        return img_new, mask_new#, weight_new
    
    def __len__(self):
        data_len =int(self.nitems)
        return data_len
    
class LoadingDatasetTest(object):
    def __init__(self, fname ,img_transform=None, mask_transform = None, edge_weight= False):
        self.fname=fname
        self.edge_weight = edge_weight
        self.img_transform=img_transform
        self.mask_transform = mask_transform
        self.tables=tables.open_file(self.fname)
        self.total_records = np.array(self.tables.root.total_elements)[0]
        self.cumsum_len = self.tables.root.cumsum_array

        #self.tables.close()
        
        self.img_1024 = None
        self.img_1200 = None
        self.img_1300 = None
        self.img_2044 = None
        self.img_2048 = None
        self.img_512 = None
        self.img_966 = None
        self.img_980 = None
        
       
        self.mask_1024 = None
        self.mask_1200 = None  
        self.mask_1300 = None 
        self.mask_2044 = None
        self.mask_2048 = None
        self.mask_512 = None
        self.mask_966 = None
        self.mask_980 = None
        
        
    def __getitem__(self, index):
        with tables.open_file(self.fname,'r') as db:
            self.img_1024 = db.root.img.data_1024
            self.img_1200 = db.root.img.data_1200
            self.img_1300 = db.root.img.data_1300
            self.img_2044 = db.root.img.data_2044
            self.img_2048 = db.root.img.data_2048
            self.img_512 = db.root.img.data_512
            self.img_966 = db.root.img.data_966
            self.img_980 = db.root.img.data_980
            
           
            self.mask_1024 = db.root.mask.data_1024
            self.mask_1200 = db.root.mask.data_1200
            self.mask_1300 = db.root.mask.data_1300
            self.mask_2044 = db.root.mask.data_2044
            self.mask_2048 = db.root.mask.data_2048
            self.mask_512 = db.root.mask.data_512
            self.mask_966 = db.root.mask.data_966
            self.mask_980 = db.root.mask.data_980

            if index >= self.total_records:
                raise IndexError("Index out of range")
    	    
            elif 0 <= index < self.cumsum_len[0]:
                img = self.img_1024[index,:,:,:]
                mask = self.mask_1024[index,:,:,:]

            elif self.cumsum_len[0] <= index < self.cumsum_len[1]:
                img = self.img_1200[(index - self.cumsum_len[0]),:,:,:]
                mask = self.mask_1200[(index - self.cumsum_len[0]),:,:,:]

            elif self.cumsum_len[1] <= index < self.cumsum_len[2]:
                img = self.img_1300[(index - self.cumsum_len[1]),:,:,:]
                mask = self.mask_1300[(index - self.cumsum_len[1]),:,:,:]

            elif self.cumsum_len[2] <= index < self.cumsum_len[3]:
                img = self.img_2044[(index - self.cumsum_len[2]),:,:,:]
                mask = self.mask_2044[(index - self.cumsum_len[2]),:,:,:]

            elif self.cumsum_len[3] <= index < self.cumsum_len[4]:
                img = self.img_2048[(index - self.cumsum_len[3]),:,:,:]
                mask = self.mask_2048[(index - self.cumsum_len[3]),:,:,:]

            elif self.cumsum_len[4] <= index < self.cumsum_len[5]:
                img = self.img_512[(index - self.cumsum_len[4]),:,:,:]
                mask = self.mask_512[(index - self.cumsum_len[4]),:,:,:]

            elif self.cumsum_len[5] <= index < self.cumsum_len[6]:
                img = self.img_966[(index - self.cumsum_len[5]),:,:,:]
                mask = self.mask_966[(index - self.cumsum_len[5]),:,:,:]

            elif self.cumsum_len[6] <= index < self.total_records:
                img = self.img_980[(index - self.cumsum_len[6]),:,:,:]
                mask = self.mask_980[(index - self.cumsum_len[6]),:,:,:]
            
            else:
                raise ValueError("Index out of range")
        	
            if ((img.shape[-2] > 1200) or (img.shape[-1] > 1200)):
                img = np.array(np.expand_dims(Image.fromarray(np.squeeze(img)).resize((1200, 1200)), axis=0)) #np.expand_dims(img, axis=0)#.astype('float32')
            else:
                img = img
      
            mask_new = mask.astype('float32')
            img_new = img.astype('float32')
    
        return img_new, mask_new
      
    def __len__(self):
        data_len =int(self.total_records)
        return data_len
    
# class LoadingDatasetTestValid(Dataset):
#     def __init__(self, image_samples, mask_samples, transform=None):
#         self.image_samples = image_samples
#         self.mask_samples = mask_samples
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_samples)

#     def __getitem__(self, idx):
#         image = self.image_samples[idx]
#         mask = self.mask_samples[idx]

#         # Apply transformations if provided
#         if self.transform:
#             image = self.transform(image)
#             mask = self.transform(mask)

#         return image, mask
    

#def undo_scaling(data, original_min, original_max):
#    scaled_data = data * (original_max - original_min) + original_min
#    return scaled_data

def percentile_normalization(image, pmin=2, pmax=99.8, axis=None):
    '''
    Compute a percentile normalization for the given image.

    Parameters:
    - image (array): array of the image file.
    - pmin  (int or float): the minimal percentage for the percentiles to compute. 
                            Values must be between 0 and 100 inclusive.
    - pmax  (int or float): the maximal percentage for the percentiles to compute. 
                            Values must be between 0 and 100 inclusive.
    - axis : Axis or axes along which the percentiles are computed. 
             The default (=None) is to compute it along a flattened version of the array.
    - dtype (dtype): type of the wanted percentiles (uint16 by default)

    Returns:
    Normalized image (np.ndarray): An array containing the normalized image.
    '''

    if not (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100 ):
        raise ValueError("Invalid values for pmin and pmax")

    low_p = np.percentile(image, pmin, axis=axis, keepdims=True)
    high_p = np.percentile(image, pmax, axis=axis, keepdims=True)

    if low_p == high_p:
        img_norm = image
        print(f"Same min {low_p} and high {high_p}, image may be empty")
    else:
        img_norm = (image - low_p) / (high_p - low_p)

    return img_norm
    
def calculate_metrics(gt_image, pred_image, all=True):
    
    if all:
        # Check if the images contain only zeroes
        if np.all(gt_image == 0) or np.all(pred_image == 0):
            # If either image contains only zeroes, return NaN for all metrics
            return np.nan, np.nan, np.nan, np.nan, np.nan
    
        # Normalize the target and predicted images                                  
        target_img_normalized = percentile_normalization(gt_image)
        pred_img_normalized = percentile_normalization(pred_image)

        # Mean Absolute Error (MAE)
        mae = mean_absolute_error(target_img_normalized.flatten(), pred_img_normalized.flatten())
            
        # Euclidean Distance (ECD)
        ecd = euclidean(target_img_normalized.flatten(), pred_img_normalized.flatten())
        
        # Cosine Distance (ECD)
        cosine_dist = cosine(target_img_normalized.flatten(), pred_img_normalized.flatten())

        # Structural Similarity Index Measure (SSIM)    
        ssim_score = ssim(target_img_normalized, pred_img_normalized,
                        data_range=np.maximum(pred_img_normalized.max() - pred_img_normalized.min(),
                                      target_img_normalized.max() - target_img_normalized.min()))      # data_range?
        
        # Pearson Correlation Coefficient (PCC)
        pcc, _ = pearsonr(target_img_normalized.flatten(), pred_img_normalized.flatten())

        return [mae, ssim_score, pcc, ecd, cosine_dist]#, mean_metric
    
    else:
        # Check if the images contain only zeroes
        if np.all(gt_image == 0) or np.all(pred_image == 0):
            # If either image contains only zeroes, return NaN for all metrics
            return np.nan, np.nan
    
        # Normalize the target and predicted images                                  
        target_img_normalized = percentile_normalization(gt_image)
        pred_img_normalized = percentile_normalization(pred_image)

        # Structural Similarity Index Measure (SSIM)    
        ssim_score = ssim(target_img_normalized, pred_img_normalized,
                        data_range=np.maximum(pred_img_normalized.max() - pred_img_normalized.min(),
                                      target_img_normalized.max() - target_img_normalized.min())) 
            
        # Pearson Correlation Coefficient (PCC)
        pcc, _ = pearsonr(target_img_normalized.flatten(), pred_img_normalized.flatten())
        
        #mean_metric = (mae + (-ssim_score) + (-pcc) + ecd + (-cosine_dist)) / 5					# Is the scale for all of them the same? If not, is that fine?
    
        return [ssim_score, pcc]#, mean_metric

def average_list(lst): 
    return sum(lst) / len(lst) 

def extract_random_section(image, mask, section_size=(256, 256)):
    # Get the shape of the image
    image_height, image_width = image.shape[1:3]
    
    # Calculate the maximum valid starting position
    max_starting_height = image_height - section_size[0]
    max_starting_width = image_width - section_size[1]
    
    # Randomly select starting positions within the valid range
    starting_height = np.random.randint(0, max_starting_height + 1)
    starting_width = np.random.randint(0, max_starting_width + 1)
    
    # Extract the section
    section_image = image[:,starting_height:starting_height + section_size[0],
                    starting_width:starting_width + section_size[1]]
    section_mask = mask[:, starting_height:starting_height + section_size[0],
                    starting_width:starting_width + section_size[1]]
    
    return (section_image, section_mask)

