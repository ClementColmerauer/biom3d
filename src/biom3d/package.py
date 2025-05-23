import argparse
from enum import Enum
import os
import torch
import numpy as np
from typing import List,Tuple

from biom3d import register
from biom3d import utils
from biom3d.predictors import seg_predict_patch_2
from biom3d.builder import Builder, read_config
from biom3d.preprocess import seg_preprocessor

class Loader(Builder):
    def __init__(self, path):                
        # for training or fine-tuning:
        # load the config file and change some parameters if multi-gpus training    
        path_to_config = os.path.join(path, 'log','config.yaml')
        self.config = utils.load_yaml_config(path_to_config)

        # if cuda is not available then deactivate USE_FP16
        if not torch.cuda.is_available():
            self.config.USE_FP16 = False

        # load the model weights
        model_dir = os.path.join(path, 'model')
        model_name = utils.load_yaml_config(os.path.join(path,"log","config.yaml")).DESC+'.pth'
        ckpt_path = os.path.join(model_dir, model_name)
        if not torch.cuda.is_available():
            ckpt = torch.load(ckpt_path,weights_only=True,map_location=torch.device('cpu'))
        else:
            ckpt = torch.load(ckpt_path,weights_only=True)
        print("Loading model from", ckpt_path)
        self.model = read_config(self.config.MODEL, register.models)
        print(self.model.load_state_dict(ckpt['model'], strict=False))

#Based on torchio gridsampler
class GridSampler:
    """
    Torchscript compatible adaptation of `torchio.data.GridSampler`, pad and cut an image in patches.
    """
    def __init__(self,
                img: torch.Tensor, 
                patch_size: torch.Tensor, 
                patch_overlap: torch.Tensor):
        """
        *GridSampler*

        Torchscript compatible adaptation of `torchio.data.GridSampler`, pad and cut an image in patches. Store patches in `self.patches`, location in `self.patches_location` and new image size in `self.padded_size`.
        
        Parameters
        ----------
        img : torch.Tensor
            The image to sample.
        patch_size : torch.Tensor
            Size of the patch used during training.
        patch_overlap: torch.Tensor
            Overlapping of patches. 
        """
        self._img = img
        self._patch_overlap = patch_overlap
        self._patch_size = patch_size
        self._img = self._pad()
        self.patches_location = self._compute_location()
        self.patches = self._generate_patch()
        self.padded_shape = self._img.shape

    def _pad(self)->torch.Tensor:
        """
        *_pad*

        Apply symetric pad so patch stay in the image bound
        
        Returns
        ----------
        Padded version of `self._img`
        """
        border = self._patch_overlap // 2
        padding:List[int] = [int(border[2].item()), int(border[2].item()),
                    int(border[1].item()), int(border[1].item()),
                    int(border[0].item()), int(border[0].item())]

        padded_img = torch.nn.functional.pad(self._img, padding, mode='constant', value=0.0)
        
        return padded_img

    def _lexsort_tensor(self,data: torch.Tensor) -> List[Tuple[int, int, int, int, int, int]]:
        """
        *_lexsort_tensor*

        Torchscript friendly (and adaptated to our format) version of lexsort then toList().

        Parameters
        ----------
        data: torch.Tensor
            The data to sort

        Returns
        ----------
        Sorted version of `data` as a List of tuple `(ystart,xstart,zstart,yend,xend,zend)`
        """
        indices = torch.arange(data.size(0))
        num_cols = data.size(1)
        for col in range(num_cols-1,-1,-1):
            keys = data[:, col]
            _, new_order = torch.sort(keys[indices], stable=True)
            indices = indices[new_order]     
        sorted_data = data[indices]
        output: List[Tuple[int, int, int, int, int, int]] = []
        for i in range(sorted_data.size(0)):
            row = sorted_data[i]
            output.append((
                int(row[0]), int(row[1]), int(row[2]),
                int(row[3]), int(row[4]), int(row[5])
            ))
        return output
    
    def _compute_location(self):
        """
        *_compute_location*

        Compute location of patches and return them in a deterministicly sorted list
          
        Returns
        ----------
        Patches location as a List of tuple `(ystart,xstart,zstart,yend,xend,zend)`
        """
        indexes:List[List[int]] = []
        size = self._img.shape[-3:]

        #Computation of starting indexes
        for i in range(len(size)):
            im_size_dim = size[i]
            patch_size_dim:int = int(self._patch_size[i].item())
            patch_overlap_dim:int = int(self._patch_overlap[i].item())      
            end:int = im_size_dim + 1 - patch_size_dim
            step:int = patch_size_dim - patch_overlap_dim
            indices_dim:List[int] = list(range(int(0),end,step))
            if indices_dim[-1] != im_size_dim - patch_size_dim: indices_dim.append(im_size_dim - patch_size_dim)
            indexes.append(indices_dim)

        #Computation of patch grid
        grids = []
        for dim in indexes:
            grids.append(torch.tensor(dim, dtype=torch.int32))
        grid = torch.meshgrid(grids, indexing="xy")
        grid = [m.flatten() for m in grid]

        # Grid to list
        indexes_start = torch.stack(grid,dim=1)
        indexes_start = torch.unique(indexes_start,dim=0)
        index_end = indexes_start + self._patch_size
        locations = torch.hstack((indexes_start,index_end))

        #Lexsort for determinism
        return self._lexsort_tensor(locations)
    
    def _crop(self,img:torch.Tensor,index_start:Tuple[int,int,int]):
        """
        *_crop*

        Generate the patch at given location by cropping `self._img`.

        Parameters
        ----------
        index_start : Tuple[int,int,int])
            Stating indexes of the patch.
          
        Returns
        ----------
        The patch
        """
        d0, h0, w0 = index_start
        #Using patch size as security
        pd = self._patch_size[0].item()
        ph = self._patch_size[1].item()
        pw = self._patch_size[2].item()
        d1 = d0 + pd
        h1 = h0 + ph
        w1 = w0 + pw
        patch = img[:, d0:d1, h0:h1, w0:w1]
        return patch
    
    def _generate_patch(self)->List[torch.Tensor]:      
        """
        *_generate_patch*

        Generate the patches with `self.patches_location`.
          
        Returns
        ----------
        A list of patch in the same order as `self.patches_location`
        """      
        patches = []
        for l in self.patches_location:
            index_ini = l[:3]
            patches.append(self._crop(self._img,index_ini))
        return patches

#Based on torchio gridaggregator
class GridAggregator():
    """
    Torchscript compatible adaptation of `torchio.data.GridAggregator`, aggregate a list of prediction and another list of those predicttion location to a logit stored in self.logit
    """
    def __init__(self,
                 patch_preds:List[torch.Tensor],
                 patch_loc_preds:List[Tuple[int,int,int,int,int,int]], 
                 output_shape:torch.Tensor, 
                 patch_size:torch.Tensor, 
                 padded_size:List[int]):
        """
        *GridAggergator*

        Torchscript compatible adaptation of `torchio.data.GridAggregator`, aggregate a list of prediction and another list of those predicttion location to a logit stored in self.logit
        
        Parameters
        ----------
        patch_preds : List[torch.Tensor]
            The list of predicted patch (not batch).
        patch_loc_preds : List[Tuple[int,int,int,int,int,int]]
            The list of predicted patch location in the same order.
        output_shape : torch.Tensor
            The shape of the image before the Sampling, Tensor of size 3 or 4.
        patch_size : torch.Tensor
            Size of the patch used during training.
        padded_size : List[int]
            Size of the image after sampling
        """
        _,D,H,W = padded_size
        C,_,_,_ = patch_preds[0].shape

        #Case where it has only spatial dimension 
        if len(output_shape) == 3:
            Do = int(output_shape[0].item())
            Ho = int(output_shape[1].item())
            Wo = int(output_shape[2].item())
            output_shape = torch.tensor((C,Do,Ho,Wo))

        self._patch_size = patch_size   
        self._hann = self._get_hann_window()
        self.logit = torch.zeros((C, D, H, W), dtype=patch_preds[0].dtype)
        self._avgmask_tensor = torch.zeros((C, D, H, W), dtype=patch_preds[0].dtype)
        for i in range(len(patch_preds)):
            self._add_patch_hann(patch_preds[i],patch_loc_preds[i])
        
        self.logit = self._finalize_output()
        padded_size_tensor = torch.tensor(padded_size)
        if len(padded_size_tensor) == 3:
            padded_size_tensor.unsqueeze(dim=0)
            padded_size_tensor[0]= C
        self.logit = self._crop(padded_size_tensor-output_shape)  

    def _get_hann_window(self):
        """
        *_get_hann_window*

        Get the hann window needed for aggregation

        Returns 
        ----------
        A 3d hann window tensor
        """
        hann_window = torch.ones(1)
        for i in range(3):
            size = self._patch_size[i]
            window_1d = torch.hann_window(size + 2, periodic=False)[1:-1]
            shape = [1, 1, 1]
            shape[i] = size.item()
            shape_tuple = (shape[0],shape[1],shape[2])
            window_1d = window_1d.view(shape_tuple)
            hann_window = hann_window * window_1d
        return hann_window
    
    def _add_patch_hann(self,
            patch: torch.Tensor,
            location: Tuple[int,int,int,int,int,int],
        ) -> None:
        """
        *_add_patch_hann*
        
        Add the patch to the logit, ponderating with an hann window

        Parameters 
        ----------
        patch : torch.Tensor
            The patch to had
        location: Tuple[int,int,int,int,int,int]
            The location of the patch
        """
        weighted_patch = patch * self._hann
        i0 = int(location[0])
        j0 = int(location[1])
        k0 = int(location[2])
        i1 = int(location[3])
        j1 = int(location[4])
        k1 = int(location[5])
        self.logit[:, i0:i1, j0:j1, k0:k1] += weighted_patch
        self._avgmask_tensor[:, i0:i1, j0:j1, k0:k1] += self._hann

    def _finalize_output(self) -> torch.Tensor:
        """
        *_finalize_output*
        
        Compute the mean of the logit by the average mask tensor

        Returns 
        ----------
        The logit, still in padded size
        """
        return torch.true_divide(self.logit, self._avgmask_tensor)
    
    def _crop(self,border: torch.Tensor) -> torch.Tensor:
        """
        *_crop*
        
        Crop symetrically the logit by the `border`

        Parameters 
        ----------
        border : torch.Tensor
            A tensor of size 3 that describe by how much the padding must be done it the 3 spatial dimension.

        Returns 
        ----------
        The logit
        """
        border = border[-3:] // 2
        border_list: List[int]= border.tolist()
        d_b, h_b, w_b = border_list
        return self.logit[:, 
                    d_b:-d_b if d_b > 0 else None,
                    h_b:-h_b if h_b > 0 else None,
                    w_b:-w_b if w_b > 0 else None]


class ModelExport(torch.nn.Module):

    def __init__(self,model,original_shape,axes_order:str,patch_size:list,num_workers:int = 4,device = 'cpu',tta= False): #enable_autocast is irrelevant since bioimage don't use it
        super().__init__()
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.tta = tta
        self.axes_order = axes_order
        self.original_shape = original_shape
        self.model = model.to(self.device).eval()
        self.model.to(self.device)

    def postprocessing():
        pass

    #@torch.jit.script
    def forward(self,img:torch.Tensor):
        overlap = 0.5
        patch_size = torch.tensor(self.patch_size,dtype=torch.int)
        original_shape = torch.tensor(self.original_shape)
        patch_overlap = torch.maximum(torch.mul(patch_size,overlap),torch.sub(patch_size, torch.tensor(img.shape[-3:])))
        patch_overlap = torch.div(torch.ceil(torch.mul(patch_overlap,overlap)),overlap).to(torch.int)
        patch_overlap = torch.minimum(patch_overlap,torch.sub(patch_size,1))

        sampler = GridSampler(img,patch_size,patch_overlap)
        patches = sampler.patches
        patches_location = sampler.patches_location
        padded_shape = sampler.padded_shape
        num_workers=4
        pred_aggr = []
        patch_loc_preds:List[Tuple[int, int, int, int, int, int]] = []
        batch_size=2  
        with torch.no_grad():   
            for i in range(0, len(patches), batch_size):
                batch_patches = patches[i:i + batch_size] 
                batch_locations = patches_location[i:i + batch_size]
                print("Batch ",i//batch_size+1,"over",len(patches)//batch_size)
                X = torch.stack(batch_patches, dim=0)
                if self.device == 'cuda':
                    X = X.cuda()
                
                if self.tta: # test time augmentation: flip around each axis
                    pred=self.model(X).cpu()
                    
                    # flipping tta
                    dims = [[2],[3],[4],[3,2],[4,2],[4,3],[4,3,2]]
                    for d in dims :
                        X_flip = torch.flip(X,dims=d)

                        pred_flip = self.model(X_flip)
                        pred += torch.flip(pred_flip, dims=d).cpu()
                        
                        
                    
                    pred = pred/(len(dims)+1)
                else:
                    pred=self.model(X).cpu()
                for j in range(len(pred)):
                    pred_aggr.append(pred[j])
                    patch_loc_preds.append(batch_locations[j])
                

        print("Aggregation...")
        aggregator = GridAggregator(pred_aggr,patch_loc_preds,original_shape,patch_size,padded_shape)
        logit = aggregator.logit
        # TODO : resize3d

        return logit
    
def to_torchscript(model):
    return torch.jit.script(model)

#Based on https://github.com/bioimage-io/core-bioimage-io-python/blob/53dfc45cf23351da61e8b22d100d77fb54c540e6/example/model_creation.ipynb
def package_bioimage_io(path_to_model,test_image,output = None,axes=None): 
    loader = Loader(path_to_model)    

    folder = os.path.join(output, loader.config.DESC)
    os.makedirs(folder, exist_ok=True)
    print("Folder created at " + folder)

    # TODO: Alter code to accepts multiple test image ?
    # Saving test image
    img, metadata = utils.adaptive_imread(test_image)
    if axes != None:
        metadata["axes"] = axes
    assert "axes" in metadata, "Axes order can't be found, you can specify it by using the --axes argument" 
    np.save(os.path.join(folder, 'test-input.npy'), img.astype(np.float32))
    print("Test input image has been saved as test-input.npy")

    # Axes order modification by preprocessing TODO move it in preprocessor
    axes_order = str.lower(metadata["axes"])
    axes_order = axes_order.replace('t', '') # Remove time dimension if exist
    if len(img.shape)==3 or len(axes_order)==3:
        axes_order = axes_order.replace('c', '') #Channel dimension has been ignored by numpy and will be placed in first place by preprocess
        axes_order = 'c'+ axes_order
    elif len(img.shape)==4: # Channel axes will be swaped in first place by preprocessing
        tmp = list(axes_order)
        tmp[loader.config["CHANNEL_AXIS"]] = tmp[0]
        tmp[0] = 'c'
        axes_order = ''.join(tmp) 

    # Saving weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ModelExport(loader.model,img.shape,axes_order,loader.config["PREDICTOR"]["kwargs"]["patch_size"],tta=False,device=device)
    # TODO: save normal weight too
    model = to_torchscript(model)
    
    model.save(os.path.join(folder, "weights-torchscript.pt"))
    print("Torchscript model has been saved as weights-torchscript.pt")
    
    # Preprocessing
    preprocessor = loader.config["PREPROCESSOR"]["kwargs"]
    # TODO replace with a more flexible approach, this work only because their is only one preprocessing function
    img_process,img_process_meta = seg_preprocessor(
                                            img,metadata,
                                            median_spacing=preprocessor['median_spacing'],
                                            clipping_bounds=preprocessor['clipping_bounds'],
                                            intensity_moments=preprocessor['intensity_moments'],
                                            channel_axis=preprocessor['channel_axis'],
                                            num_channels=preprocessor['num_channels'],
                                            )
    

    # Prediction
    output = model(torch.from_numpy(img_process))


    # Postprocessing
    # TODO : if keep_big/biggest or force_softmax or use_softmax are used warning or error

    """with open(os.path.join(folder, 'doc.md'), "w") as f:
        f.write("# My First Model\n")
        f.write("This model was trained on a very big dataset.\n")
        f.write("You should not let it get wet or feed it after midnight.\n")
        f.write("To validate its predictions, make sure that it does not produce any evil clones.\n")
    print("Documention created as doc.md")"""

    '''build_model(
        # the weight file and the type of the weights
        weight_uri=os.path.join(folder, 'weights.pt'),
        weight_type="torchscript",
        # the test input and output data as well as the description of the tensors
        # these are passed as list because we support multiple inputs / outputs per model
        test_inputs=[os.path.join(folder, 'test-input.npy')],
        test_outputs=[os.path.join(folder, 'test-output.npy')],
        input_axes=["bcyx"],
        output_axes=["bcyx"],
        # where to save the model zip, how to call the model and a short description of it
        output_path=os.path.join(folder,model_config.DESC+ '.zip'),
        name="MyFirstModel",
        description="a fancy new model",
        # additional metadata about authors, licenses, citation etc.
        authors=[{"name": "Gizmo"}],
        license="CC-BY-4.0",
        documentation="my-model/doc.md",
        tags=["nucleus-segmentation"],  # the tags are used to make models more findable on the website
        cite=[{"text": "Gizmo et al.", "doi": "doi:10.1002/xyzacab123"}],)'''

    
class Target(Enum ):    
    v0x5BIZ = "v0.5BioImageZoo"

    def __str__(self):
        return self.value

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Package model.")
    parser.add_argument("-t", "--target", type=Target, default=Target.v0x5BIZ, choices=list(Target),help="Target image and version")
    parser.add_argument("-o", "--output_dir", type=str, default="./",help="Directory where you want your model, will create a sub folder (default local directory)")
    parser.add_argument("-b", "--best", action = "store_true",help="Whether best model is used")
    parser.add_argument("-a", "--axes", type = str, default = None, help="Specified axes order for images")
    parser.add_argument("model_dir",help="Path to model directory")  
    parser.add_argument("test_image",help="Path to test image (must be tif or nii.gz)")  
    args = parser.parse_args()
    
    if(args.target == Target.v0x5BIZ):
        package_bioimage_io(args.model_dir,args.test_image,args.output_dir,args.axes)