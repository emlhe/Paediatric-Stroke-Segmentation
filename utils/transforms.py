import torchio as tio
import math

def preprocess(brain_mask=None, crop=None):
    return tio.Compose([
                tio.ToCanonical(), # "Reorder the data to be closest to canonical (RAS+) orientation."
                tio.Resample('t1'), # Make sure all label maps have same affine as t1
                tio.Resample(1),
                tio.ZNormalization(masking_method=brain_mask),
                tio.CropOrPad(crop, mask_name=brain_mask),
                tio.OneHot(num_classes = 2),
                # tio.EnsureShapeMultiple((32,32,32)), # for the U-Net : doit être un multiple de 2**nombre de couches
                ])
        
def augment():
    return tio.Compose([
                tio.OneOf({
                    tio.RandomAffine(scales=(0.6,1.1),degrees=20, translation=(-10,10)): 0.8,
                    tio.RandomElasticDeformation(num_control_points = 7, max_displacement = 7.5): 0.2, #Si trop de control points : ralenti bcp le chargement
                    },
                    p=0.75,
                ),
                tio.RandomMotion(p=0.2),
                tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.2),  # Change contrast 
                tio.RandomFlip(axes=('LR',), flip_probability=0.2),
                tio.RandomBiasField(coefficients = 0.5, order = 3, p=0.2),
                tio.RandomNoise(mean = 0, std=(0.005, 0.1), p=0.2),
                                ])

def tta():
    u_f = 0.5   # Flipping along each 2d axis 
    r0 = 0      # rotation angle in 2D
    r1 = 2*math.pi
    s0 = 0.8    # Scaling factor
    s1 = 1.2
    mu_e = 0
    sigma_e = 0.05
    return tio.Compose([
                tio.RandomFlip(axes=('LR')),
                tio.RandomAffine(scales=(s0, s1)),
                tio.RandomAffine(degrees=(r0,r1)),
                tio.RandomNoise(mean=mu_e, std=sigma_e),
                ])
