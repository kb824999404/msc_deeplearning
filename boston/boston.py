from HelperClass.NeuralNet_1_1 import *

    

if __name__=='__main__':
    file_name='data.npz'
    # data
    reader = DataReader_1_1(file_name)
    reader.ReadData()
    reader.NormalizeX()
    reader.NormalizeY()
    # net
    hp = HyperParameters_1_0(13, 1, eta=0.03, max_epoch=500, batch_size=10, eps=1e-4)
    net = NeuralNet_1_1(hp)
    net.train(reader, checkpoint=0.1)
    # inference
    x=np.array([[0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98],
    [0.22489,12.50,7.870,0,0.5240,6.3770,94.30,6.3467,5,311.0,15.20,392.52,20.45]]).reshape(2,13)
    x_new = reader.NormalizePredicateData(x)
    z = net.inference(x_new)
    Z_true = z * reader.Y_norm[0,1] + reader.Y_norm[0,0]
    print("Z=", Z_true)
