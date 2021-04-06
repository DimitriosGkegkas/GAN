import glob
import yaml
from GAN.UNIT.trainer import UNIT_Trainer
from GAN.UNIT.utils import *
from PIL import Image

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)
class UNIT_pretrained():
  def __init__(self,):
    config = get_config("GAN/UNIT/configs/unit_gta2city_list.yaml")
    config['vgg_model_path'] = 'GAN/UNIT/'
    trainer = UNIT_Trainer(config)
    model_path="GAN/UNIT/models/unit_gta2city.pt"

    # %run -i GAN/UNIT/test.py --trainer UNIT --config configs/unit_gta2city_list.yaml --input inputs/gta_example.jpg --output_folder results/gta2city --checkpoint models/unit_gta2city.pt --a2b 1
    self.style = ""
    self.output_folder = "GAN/UNIT/results/gta2city"
    output_only =False
    a2b=1


    try:
        state_dict =  torch.load(model_path)
        trainer.gen_a.load_state_dict(state_dict['a'])
        trainer.gen_b.load_state_dict(state_dict['b'])
    except:
        state_dict = pytorch03_to_pytorch04( torch.load(model_path))
        trainer.gen_a.load_state_dict(state_dict['a'])
        trainer.gen_b.load_state_dict(state_dict['b'])

    trainer.cuda()
    trainer.eval()


    self.encode = trainer.gen_a.encode if a2b else trainer.gen_b.encode # encode function
    style_encode = trainer.gen_b.encode if a2b else trainer.gen_a.encode # encode function
    self.decode = trainer.gen_b.decode if a2b else trainer.gen_a.decode # decode function

    if 'new_size' in config:
        new_size = config['new_size']
    else:
        if a2b==1:
            new_size = config['new_size_a']
        else:
            new_size = config['new_size_b']
    self.transform = transforms.Compose([transforms.Resize(new_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  def transform_image(self, input):
  
    with torch.no_grad():

        image = Variable(self.transform(Image.open(input).convert('RGB')).unsqueeze(0).cuda())
        style_image = Variable(self.transform(Image.open(style).convert('RGB')).unsqueeze(0).cuda()) if self.style != '' else None

        # Start testing
        content, _ = self.encode(image)
        
        outputs = self.decode(content)
        outputs = (outputs + 1) / 2.

        path_output = os.path.join(self.output_folder,  input.split("/")[-1].split(".")[0]+'_output.jpg')
        vutils.save_image(outputs.data, path_output, padding=0, normalize=True)
        # also save input images
        path_input= os.path.join(self.output_folder,   input.split("/")[-1].split(".")[0]+'_input.jpg')
        vutils.save_image(image.data,path_input, padding=0, normalize=True)

        show([Image.open(path_input).convert('RGB'), Image.open(path_output).convert('RGB')])