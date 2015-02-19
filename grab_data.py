import urllib
import time

source_url = "https://www.cs.dal.ca/cams/university.jpg";

def getjpg(pause = 1, n_img = 150):
    for x in range(10, n_img+10):
        file_name =  "0" + str(x) + ".jpg";
        print(file_name)
        urllib.urlretrieve(source_url, file_name)
        time.sleep(0.5)
        

if __name__ == "__main__":
    getjpg()



