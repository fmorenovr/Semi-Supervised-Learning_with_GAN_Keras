import os

def verifyFile(files_list):
  return os.path.isfile(files_list)

def verifyType(file_name):
  if os.path.isdir(file_name):
    return "dir"
  elif os.path.isfile(file_name):
    return "file"
  else:
    return None

def verifyDir(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
