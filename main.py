from utils import *
import YDS_show_V60
import matlab

# path of simulation results
file = 'data/result_Xinxi street_V2V'
matfilepath = file + '/1/matlab'

# file name of extracted cluster
keyinfo = 'result_' + 'Xinxi street' + '_' + 'V2V'

# Step 1
extractCluster(sourcepath=matfilepath, file=keyinfo + '.csv')
samplepath = 'sample/' + keyinfo
if not os.path.exists(samplepath):
    os.makedirs(samplepath)
# Step 2
matchCluster(file=keyinfo + '.csv', samplepath=samplepath)

# scale of super resolution
scale = 2

# Step 3
model = os.path.join('model', f'model_{scale}.pth')
predict(samplepath, keyinfo, scale, model)

scaleIn = scale
scenarioIn = "Xinxi street_V2V"

# load prediction cluster files in .mat format
loadfilenameIn = "prediction/YDS_" + keyinfo + "_" + f"{scale}.mat"
materialPathIn = file + '\Material.json'

## visualization: check the folder 'Visualization', install python library YDS_show_V60 and then perform a visual display
## stateIn = 1 evaluation 2 test
stateIn = matlab.double([1.0], size=(1, 1))
my_show = YDS_show_V60.initialize()
retn = my_show.YDS_show(stateIn, scenarioIn, scaleIn, matfilepath, loadfilenameIn, materialPathIn)
my_show.terminate()
