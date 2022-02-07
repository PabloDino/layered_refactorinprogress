# Your data source for wav files
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-base50/'
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-Base50p/'
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-tst-base50p/'
baseFolder = '/home/paul/Downloads/ESC-50-tst2b/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-Next30p/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-next30p/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-tst-next30p/'
#nextFolder = '/home/paul/Downloads/ESC-50-tst2b/'
lastFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-last20p/'
#lastFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-tst-last20p/'

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0

dataSourceBase=baseFolder#lastFolder
# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
totalLabel = 10

# model parameters for training
batchSize = 128
epochs = 100
latent_dim=8
dataSize=128

timesteps = 128 # Length of your sequences
input_dim = 128 
