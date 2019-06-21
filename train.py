import VDSR_model
import generator
import argparse
import scipy
import glob
from datetime import datetime

from keras.optimizers import Adam
from keras import losses
from utils import CSVLoggerTimestamp

parser = argparse.ArgumentParser(description='VDSR learning parameter')
parser.add_argument('--batchsize', metavar='B', default=4, type=int, 
                    help='batch size')
parser.add_argument('--epoch', metavar='E', default=1000, type=int, 
                    help='number of epoch')
parser.add_argument('--train', metavar='T', type=str, 
                    help='training data directory', required=True)
parser.add_argument('--valid', metavar='V', type=str, 
                    help='valid data directory')
args = parser.parse_args()

ImageList = glob.glob(args.train+'/*')
tmpImage = scipy.misc.imread(ImageList[0])
DataLen = len(ImageList)
Iteration = int(DataLen / args.batchsize)

DataGen = generator.DataGenerator()

if len(args.valid) > 0:
    ValidImageList = glob.glob(args.valid+'/*')
    test_flow = DataGen.flow_from_directory(directory=args.valid,
                                            batch_size=len(ValidImageList),
                                            iteration=1)
    val_lr, val_hr = test_flow.__next__()

model = VDSR_model.VDSR_origin( input_shape=tmpImage.shape )

opt = Adam(lr=1E-3, decay=1E-3)
model.compile(loss='mse', optimizer=opt, metrics=[losses.mse])
NOW = datetime.today().timetuple()
DATE = str(NOW[0])+'-{0:02d}'.format(NOW[1])+'-{0:02d}'.format(NOW[2])+'-{0:02d}'.format(NOW[3])+'-{0:02d}'.format(NOW[4])
NAME = 'VDSR_itr'+str(args.epoch)+'_bs'+str(args.batchsize)+'_'+args.train[2:]+'_'+DATE
csv_logger_t = CSVLoggerTimestamp(NAME+'.csv')

model.fit_generator(generator=DataGen.flow_from_directory(directory=args.train, batch_size=args.batchsize, iteration=Iteration),
                    steps_per_epoch = Iteration,
                    epochs = args.epoch,
                    verbose = 1,
                    callbacks = [csv_logger_t],
                    validation_data = (val_lr, val_hr))

model.save(NAME+'.hdf5')

