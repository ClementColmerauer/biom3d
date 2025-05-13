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

    def __init__(self,model,original_shape,patch_size:list,num_workers:int = 4,device = 'cpu',tta= False,enable_autocast = True):
        super().__init__()
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.enable_autocast = enable_autocast
        self.tta = tta
        self.original_shape = original_shape
        self.model.to(self.device).eval()
        self.model.cpu()

    def grid_patching(self,img:torch.Tensor, patch_size:torch.Tensor, patch_overlap:torch.Tensor) :
        _, H, W, D = img.shape
        dims = [D, W, H]  # Pad order for F.pad: D, W, H
        pad:list[int] = []

        stride = torch.sub(patch_size,patch_overlap)

        for dim, size, s in zip(dims, patch_size.flip(0), stride.flip(0)):
            remainder = (dim - size) % s
            pad_amt = (s - remainder) %s
            pad.extend([0, int(pad_amt)])  # Pad only after (right, bottom, etc.)

        if any(pad):
            img = F.pad(img, pad, mode='constant')

        _, H, W, D = img.shape

        # Torchscript compatibility
        patch_size_list = [int(x) for x in patch_size]
        stride_list = [int(x) for x in stride]
        steps:list[list[int]] = []
        for i in range(3):  # Image are in 3d
            dim = (H, W, D)[i]
            ps = patch_size_list[i]
            st = stride_list[i]
            steps.append(list(range(0, dim - ps + 1, st)))

        patch_location:List[Tuple[int, int, int, int, int, int]] = []
        patches:List[torch.Tensor]= []
        for i in steps[0]:  # H
            for j in steps[1]:  # W
                for k in steps[2]:  # D
                    patch = img[:, i:i+patch_size[0], j:j+patch_size[1], k:k+patch_size[2]]
                    location = (i, j, k, int(patch_size[0]), int(patch_size[1]), int(patch_size[2]))
                    patches.append(patch)
                    patch_location.append(location)


        return patches, patch_location,img.shape
    
    def get_hann_window(self,patch_size):
        patch_size_list = [int(x) for x in patch_size]
        hann_1d = [torch.hann_window(s, periodic=False) for s in patch_size_list]
        hann_3d = torch.einsum('i,j,k->ijk', hann_1d[0], hann_1d[1], hann_1d[2])
        return hann_3d.unsqueeze(0).unsqueeze(0) 
    
    def aggregate_patches(self,patch_preds:List[torch.Tensor],patch_loc_preds:List[Tuple[int,int,int,int,int,int]], output_shape:List[int], patch_size:torch.Tensor, device:str='cpu'):
        C, H, W, D = output_shape
        if isinstance(device, torch.Tensor):
            device = device.device 
        elif isinstance(device, str):
            device = torch.device(device)
        aggregated = torch.zeros((C, H, W, D), device=device)
        weight_map = torch.zeros((C, H, W, D), device=device)

        window = self.get_hann_window(patch_size).to(device)
        for i in range(len(patch_preds)):
            patch = patch_preds[i]
            loc = patch_loc_preds[i]
            i, j, k, ph, pw, pd = loc
            weighted_patch = patch * window  

            if weighted_patch.dim() == 5:
                weighted_patch = weighted_patch.squeeze(0)

            aggregated[:, i:i+ph, j:j+pw, k:k+pd] += weighted_patch.squeeze(0)
            weight_map[:, i:i+ph, j:j+pw, k:k+pd] += window.squeeze(0)

        weight_map = torch.clamp(weight_map, min=1e-6)
        return aggregated / weight_map

    #@torch.jit.script
    def forward(self,img:torch.Tensor):
        overlap = 0.5
        patch_size = torch.tensor(self.patch_size,dtype=torch.int)
        original_shape = torch.squeeze(torch.tensor(self.original_shape),dim=0)
        patch_overlap = torch.maximum(torch.mul(patch_size,overlap),torch.sub(patch_size, torch.tensor(img.shape[3:])))
        patch_overlap = torch.div(torch.ceil(torch.mul(patch_overlap,overlap)),overlap).to(torch.int)
        patch_overlap = torch.minimum(patch_overlap,torch.sub(patch_size,1))

        patches, patches_location, padded_shape = self.grid_patching(img, patch_size,patch_overlap)

        with torch.no_grad():
            pred_aggr = []
            patch_loc_preds:List[Tuple[int, int, int, int, int, int]] = []
            batch_size=2    
            
            for i in range(0, len(patches), batch_size):
                batch_patches = patches[i:i + batch_size] 
                batch_locations = patches_location[i:i + batch_size]
                X = torch.stack(batch_patches, dim=0)
                if self.device == 'cpu':
                    X = X.cuda()
                
                if self.tta: # test time augmentation: flip around each axis
                    with torch.autocast(self.device, enabled=self.enable_autocast):
                        pred=self.model(X).cpu()
                    
                    # flipping tta
                    dims = [[2],[3],[4],[3,2],[4,2],[4,3],[4,3,2]]
                    for d in dims :
                        X_flip = torch.flip(X,dims=d)

                        with torch.autocast(self.device, enabled=self.enable_autocast):
                            pred_flip = self.model(X_flip)
                        pred += torch.flip(pred_flip, dims=d).cpu()
                        
                        
                    
                    pred = pred/(len(dims)+1)
                else:
                    with torch.autocast(self.device, enabled=self.enable_autocast):
                        pred=self.model(X).cpu()

                for j in range(len(pred)):
                    pred_aggr.append(pred[j])
                    patch_loc_preds.append(batch_locations[j])


        logit = self.aggregate_patches(pred_aggr,patch_loc_preds,padded_shape,patch_size,device=self.device).to(torch.float)

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

    # Saving weights
    model = ModelExport(loader.model,img.shape,loader.config["PREDICTOR"]["kwargs"]["patch_size"],tta=True)
    # TODO: save normal weight too
    # TODO: create a toTorchSript function
    model = torch.jit.script(model)
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
    
    # Axes order modification by preprocessing TODO move it in preprocessor
    axes_order = str.lower(metadata["axes"])
    axes_order = axes_order.replace('t', '') # Remove time dimension if exist
    if len(img.shape)==3:
        axes_order = axes_order.replace('c', '') #Channel dimension has been ignored by numpy and will be placed in first place by preprocess
        axes_order = 'c'+ axes_order
    elif len(img.shape)==4: # Channel axes will be swaped in first place by preprocessing
        tmp = list(axes_order)
        tmp[loader.config["CHANNEL_AXIS"]] = tmp[0]
        tmp[0] = 'c'
        axes_order = ''.join(tmp) 

    # Prediction
    output_w = seg_predict_patch_2(img_process,img.shape,loader.model,loader.config["PATCH_SIZE"],tta=True)
    output = model(img_process)

    assert torch.equal(output,torch.from_numpy(output_w))

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