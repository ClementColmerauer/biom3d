import argparse
from enum import Enum
import os
import torch
import numpy as np

from biom3d import register
from biom3d import utils
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
    model = ExportModel(loader.model)
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

    # Axes order modification by preprocessing
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
    """with torch.no_grad():
        output = model(torch.from_numpy(img_process)).numpy()"""

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

''' Emprunter a vhatgpt, a analyser
# 7. Post-traitement (ex: argmax, seuil, etc.)
output_post = (output > 0.5).astype(np.uint8)  # adapte selon ton cas

# 8. Sauvegarde de la sortie
np.save("my-model/test-output.npy", output_post)'''

    
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
        packagev0x5BIZ(args.model_dir,args.test_image,args.output_dir,args.best)