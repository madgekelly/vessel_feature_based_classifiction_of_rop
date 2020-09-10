**Vessel Feature-Based Classification of Plus Disease ROP**

This repository consists of the code used in my research project for an Msc in Machine Learning 
and Autonomous Systems. 
The methods are inspired by the work of Nisha et al. (2019). The code includes functions to segment
retinal blood vessels from fundus images, roughly based on the methods of Nisha et al. (2017), 
Additionally, the code includes functions to extract vessel features from a retinal vessel 
segmentation, that are relevant to a diagnosis of plus disease ROP. The methods for tortuosity 
measurement were taken from Poletti, Grisanand Ruggeri (2012). Finally, there is code to train
and test a decision tree classifier from the extracted features.

**Requirements**
```bash
pip install -r requirements.txt
```
**References**

1) Nisha, K., Sreelekha, G., Sathidevi, P., Mohanachandran, P. and Vinekar, A., 2019. A
computer-aided diagnosis system for plus disease in retinopathy of prematurity with
structure adaptive segmentation and vessel based features. _Computerized Medical Imaging
and Graphics_, 74, pp.72-94.

2) Nisha, K.L., Sreelekha, G., Savithri, S.P., Mohanachandran, P. and Vinekar, A., 2017.
Fusion of structure adaptive filtering and mathematical morphology for vessel segmentation
in fundus images of infants with retinopathy of prematurity. _30th canadian
conference on electrical and computer engineering_, Windsor. New York: IEEE, pp.1{6.

3) Poletti, E., Grisan, E. and Ruggeri, A., 2012. Image-level tortuosity estimation in wide-
field retinal images from infants with retinopathy of prematurity. _2012 annual international
conference of the ieee engineering in medicine and biology society_, San Diego.
New York: IEEE, pp.4958-4961.