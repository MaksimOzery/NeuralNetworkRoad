#Переименование файлов
from PIL import Image, ImageDraw, ImageFont
import struct
import os 
# user -> name
directory = r'C:\Users\user\Desktop\7'
directory2=r'C:\Users\user\Desktop\image3'
print( directory)
files = os.listdir(directory) 

print( len(files))

#posetive.1.
#negative.1.
for i in range(len(files)):
    print( directory+"/"+files[i])            
    img = Image.open(open(directory+"/"+files[i], 'rb'))
    img.save(directory2+"/posetive.1."+str(0+i)+".png")
   

