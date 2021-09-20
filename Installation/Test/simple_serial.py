import h5py as h5

print("Got to load module")

f = h5.File("/nfs/home/vpais/ThesisResults/TestSerial.h5", "w")

print("Got to open file")

f["hey"] = [2,1,1,1]

print("Got to save data in new dataset hey")

f.close()

print("Got to close file")
