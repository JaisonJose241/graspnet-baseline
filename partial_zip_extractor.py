# import urllib, struct, sys
# import urllib.request as urllib

# def open_remote_zip(url, offset=0):
#  return urllib.urlopen(urllib.Request(url, headers={'Range': 'bytes={}-'.format(offset)}))

# offset = 0
# zipfile = open_remote_zip(sys.argv[1])
# header = zipfile.read(30)

# while header[:4] == 'PK\x03\x04':
#  compressed_len, uncompressed_len = struct.unpack('<II', header[18:26])
#  filename_len, extra_len = struct.unpack('<HH', header[26:30])
#  header_len = 30 + filename_len + extra_len
#  total_len = header_len + compressed_len

#  print('{}\n offset: {}\n length: {}\n  header: {}\n  payload: {}\n uncompressed length: {}'.format(zipfile.read(filename_len), offset, total_len, header_len, compressed_len, uncompressed_len))
#  zipfile.close()

#  offset += total_len
#  zipfile = open_remote_zip(sys.argv[1], offset)
#  header = zipfile.read(30)

# zipfile.close()

# importing required modules
from zipfile import ZipFile
  
# specifying the zip file name
file_name = "/home/jsn/sample.zip"


def fixBadZipfile(zipFile):  
 f = open(zipFile, 'r+b')  
 data = f.read()  
 pos = data.find(b'\x50\x4b\x05\x06') # End of central directory signature  
 if (pos > 0):  
     print("ey")
    #  log("Trancating file at location " + str(pos + 22)+ ".")  
     f.seek(pos + 22)   # size of 'ZIP end of central directory record' 
     f.truncate()  
     f.close()  
 else:  
     # raise error, file is truncated  
     pass

fixBadZipfile(file_name)
 zip -FF sample.zip --out RepairedZip.zip 


# opening the zip file in READ mode
# with ZipFile(file_name, 'r') as zip:
#     # printing all the contents of the zip file
#     zip.printdir()
  
#     # extracting all the files
#     print('Extracting all the files now...')
#     zip.extractall()
#     print('Done!')