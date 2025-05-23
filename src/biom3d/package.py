import argparse
from enum import Enum
import os
import torch
import numpy as np
import torch.nn.functional as F
from typing import List,Tuple

from biom3d import register
from biom3d import utils
from biom3d.predictors import seg_predict_patch_2
from biom3d.builder import Builder, read_config
from biom3d.preprocess import seg_preprocessor
from biom3d.predictors import seg_postprocessing

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


    #Equivalent to (and translation of) torchio.GridSampler 
    def grid_patching(
        self,
        img: torch.Tensor, 
        patch_size: torch.Tensor, 
        patch_overlap: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int, int, int, int]], torch.Size]:

        def pad(img:torch.Tensor,patch_overlap:torch.Tensor)->torch.Tensor:
            border = patch_overlap // 2
            padding_tuple = (border[2].item(), border[2].item(),
                     border[1].item(), border[1].item(),
                     border[0].item(), border[0].item())

            padded_img = torch.nn.functional.pad(img, padding_tuple, mode='constant', value=0)
            
            return padded_img
            

        def compute_location(img:torch.Tensor,patch_size:torch.Tensor, patch_overlap:torch.Tensor):
            def lexsort_tensor(data: torch.Tensor) -> torch.Tensor:
                indices = torch.arange(data.size(0))
                for col in reversed(range(data.size(1))):
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
            indices = []
            size = img.shape[-3:]
            for i in range(len(size)):
                im_size_dim = size[i]
                patch_size_dim = patch_size[i].item()
                patch_overlap_dim = patch_overlap[i].item()      
                end = im_size_dim + 1 - patch_size_dim
                step = patch_size_dim - patch_overlap_dim
                indices_dim = list(range(0,end,step))
                if indices_dim[-1] != im_size_dim - patch_size_dim: indices_dim.append(im_size_dim - patch_size_dim)
                indices.append(indices_dim)
            grid = torch.meshgrid(*[torch.tensor(dim, dtype=torch.int32) for dim in indices], indexing="xy")
            grid = [m.flatten() for m in grid]
            indices_ini = torch.stack(grid,dim=1)
            indices_ini = torch.unique(indices_ini,dim=0)
            indices_fin = indices_ini + patch_size
            locations = torch.hstack((indices_ini,indices_fin))
            return lexsort_tensor(locations)
        def generate_patch(img:torch.Tensor,patch_size:torch.Tensor,patch_location:List[Tuple[int,int,int,int,int,int]]):
            def crop(img:torch.Tensor, index_ini:Tuple[int,int,int],patch_size:torch.Tensor):
                d0, h0, w0 = index_ini
                pd, ph, pw = patch_size
                d1 = d0 + pd
                h1 = h0 + ph
                w1 = w0 + pw
                patch = img[:, d0:d1, h0:h1, w0:w1]
                return patch
            
            patches = []
            for l in patch_location:
                index_ini = l[:3]
                patches.append(crop(img,index_ini,patch_size))
            return patches
        img = pad(img,patch_overlap)
        patch_location = compute_location(img,patch_size,patch_overlap)
        patches = generate_patch(img,patch_size,patch_location)

        return patches, patch_location, torch.tensor(img.shape)

    #Equivalent to (and translation of) torchio GridAggregator
    def aggregate_patches(self,patch_preds:List[torch.Tensor],patch_loc_preds:List[Tuple[int,int,int,int,int,int]], output_shape:torch.Tensor, patch_size:torch.Tensor, padded_size:Tuple[int,int,int,int],device:str='cpu'):
        def get_hann_window(patch_size:torch.Tensor):
            hann_window = torch.ones(1)
            for i in range(3):
                size = patch_size[i]
                window_1d = torch.hann_window(size + 2, periodic=False)[1:-1]
                shape = [1, 1, 1]
                shape[i] = size
                window_1d = window_1d.view(*shape)
                hann_window = hann_window * window_1d
            return hann_window
        def add_patch_hann(
            output_tensor: torch.Tensor,
            avgmask_tensor: torch.Tensor,
            patch: torch.Tensor,
            location: Tuple[int,int,int,int,int,int],
            hann_window: torch.Tensor,
        ) -> None:
            weighted_patch = patch * hann_window
            i0 = int(location[0])
            j0 = int(location[1])
            k0 = int(location[2])
            i1 = int(location[3])
            j1 = int(location[4])
            k1 = int(location[5])
            output_tensor[:, i0:i1, j0:j1, k0:k1] += weighted_patch
            avgmask_tensor[:, i0:i1, j0:j1, k0:k1] += hann_window

        def finalize_output(output_tensor: torch.Tensor, avgmask_tensor: torch.Tensor) -> torch.Tensor:
            return torch.true_divide(output_tensor, avgmask_tensor)
        
        def crop_tensor(tensor: torch.Tensor, border: torch.Tensor) -> torch.Tensor:
            """
            Supprime un padding symétrique du tenseur 4D (C, D, H, W).
            `border` est un tensor de forme (4,) indiquant combien couper dans chaque dimension spatiale.
            """
            border = border[-3:] // 2
            d_b, h_b, w_b = border.tolist()
            return tensor[:, 
                        d_b:-d_b if d_b > 0 else None,
                        h_b:-h_b if h_b > 0 else None,
                        w_b:-w_b if w_b > 0 else None]

        _,D,H,W = padded_size
        C,_,_,_ = patch_preds[0].shape
        if len(output_shape) == 3:
            Do,Ho,Wo = output_shape
            output_shape = torch.tensor((C,Do,Ho,Wo))
        hann = get_hann_window(patch_size)
        logit = torch.zeros((C, D, H, W), dtype=patch_preds[0].dtype)
        avgmask_tensor = torch.zeros((C, D, H, W), dtype=patch_preds[0].dtype)
        for i in range(len(patch_preds)):
            add_patch_hann(logit,avgmask_tensor,patch_preds[i],patch_loc_preds[i],hann)
        
        logit = finalize_output(logit,avgmask_tensor)
        logit = crop_tensor(logit,padded_size-output_shape)  
        return logit
    


    #@torch.jit.script
    def forward(self,img:torch.Tensor):
        overlap = 0.5
        patch_size = torch.tensor(self.patch_size,dtype=torch.int)
        original_shape = torch.tensor(self.original_shape)
        patch_overlap = torch.maximum(torch.mul(patch_size,overlap),torch.sub(patch_size, torch.tensor(img.shape[-3:])))
        patch_overlap = torch.div(torch.ceil(torch.mul(patch_overlap,overlap)),overlap).to(torch.int)
        patch_overlap = torch.minimum(patch_overlap,torch.sub(patch_size,1))

        patches, patches_location, padded_shape = self.grid_patching(img, patch_size,patch_overlap)
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
                    print(pred[j].shape)
                    pred_aggr.append(pred[j])
                    patch_loc_preds.append(batch_locations[j])
                

        print("Aggregation...")
        logit = self.aggregate_patches(pred_aggr,patch_loc_preds,original_shape,patch_size,padded_shape,device=self.device)
        self.model = self.model.cpu()
        # TODO : resize3d

        return logit

#Based on https://github.com/bioimage-io/core-bioimage-io-python/blob/53dfc45cf23351da61e8b22d100d77fb54c540e6/example/model_creation.ipynb
def packagev0x5BIZ(path_to_model,test_image,output = None,axes=None): 
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
    print(axes_order,img.shape)
    axes_order = axes_order.replace('t', '') # Remove time dimension if exist
    if len(img.shape)==3 or len(axes_order)==3:
        print(axes_order)
        axes_order = axes_order.replace('c', '') #Channel dimension has been ignored by numpy and will be placed in first place by preprocess
        axes_order = 'c'+ axes_order
        print(axes_order)
    elif len(img.shape)==4: # Channel axes will be swaped in first place by preprocessing
        tmp = list(axes_order)
        tmp[loader.config["CHANNEL_AXIS"]] = tmp[0]
        tmp[0] = 'c'
        axes_order = ''.join(tmp) 
    print(axes_order)

    # Saving weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ModelExport(loader.model,img.shape,axes_order,loader.config["PREDICTOR"]["kwargs"]["patch_size"],tta=True,device=device)
    # TODO: save normal weight too
    # TODO: create a toTorchSript function
    """model = torch.jit.script(model)
    model.save(os.path.join(folder, "weights-torchscript.pt"))"""
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
        packagev0x5BIZ(args.model_dir,args.test_image,args.output_dir)