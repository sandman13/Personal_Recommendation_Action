def preprocess_file(filename,outFile):
    with open(filename,'r') as infile:
        with open(outFile,'w')as outfile:
            for line in infile.readlines():
                line=line.strip()
                line=line.replace(', ',',')
                if not line or ',' not in line:
                    continue
                if line[-1]=='.':
                    line=line[:-1]
                line+='\n'
                outfile.write(line)

if __name__=='__main__':
    preprocess_file("../data/train.txt","../data/train1.txt")
    preprocess_file("../data/test.txt",'../data/test1.txt')