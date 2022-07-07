import ftplib
def download(number, threads = 8):
  # Open the FTP connection
  ftp = ftplib.FTP('ftp.ncbi.nlm.nih.gov')
  ftp.login() 
  ftp.cwd('/pubmed/baseline/')
  from pathlib import Path

  # list directory contents
  # ftp.retrlines('LIST')

  # download
  filenames = ftp.nlst()
  for i, filename in enumerate(filenames):
      if hash(filename) % threads == number:
        print(f'{i:4} out of {len(filenames)}: {filename}', end="\r")
        o = Path("data", filename)
        if o.exists():
            continue
        with open( o, 'wb' ) as file :
            ftp.retrbinary('RETR %s' % filename, file.write)
            file.close()

  ftp.quit()

if __name__ == "__main__":
  import sys
  section = int(sys.argv[1])
  download(section)