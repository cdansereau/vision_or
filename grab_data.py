__author__ = "Christian Dansereau"
__copyright__ = "Copyright 2015, Christian Dansereau"
import urllib
import time

source_url = "https://www.cs.dal.ca/cams/university.jpg"
#source_url = "https://www.cs.dal.ca/cams/management.jpg"

def getjpg(pause = 1, n_img = 300):
    for x in range(10, n_img+10):
        file_name =  "0" + str(x) + ".jpg"
        print(file_name)
        urllib.urlretrieve(source_url, file_name)
        time.sleep(0.5)
        

if __name__ == "__main__":
    getjpg()



