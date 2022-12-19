import torchvision


train_data = torchvision.datasets.CelebA(root = "./data", split = "train",target_type='identity', download = True)
valid_data = torchvision.datasets.CelebA(root = "./data", split = "valid",target_type='identity', download = True)

lines = []
with open('data/celeba/identity_CelebA.txt') as f:
    for line in f.readlines():
        line = line[:-1]
        img_num, img_label = map(str, line.split())
        lines.append((img_num, int(img_label)))
        
f.close()
    
def pairs(lines):
    pairs_lst = []
    for i in range(len(lines)):
        print(len(lines), i)
        j=(i+1)%(len(lines))
        print(lines[i], lines[j])
        while(lines[j][1] != lines[i][1]):
            if j<len(lines)-1:
                j += 1
            else:
                j = 0
        
        k=(i+1)%(len(lines))      
        while(lines[k][1] == lines[i][1]):
            if j<len(lines):
                k += 1
            else:
                k = 0
                
        positive = (lines[i][0], lines[j][0], '1')
        negative = (lines[i][0], lines[k][0], '0')
        
        pairs_lst.append(positive)
        pairs_lst.append(negative)
        
    return pairs_lst

pairlist = pairs(lines)

print(pairlist[0])

path = 'data/siamese_celeba/'

def save_files (pairlist):
    with open('data/siamese_celeba.txt',"w") as f:
        for line in pairlist:
            print(line)
            f.write(str(line))
            f.write('\n')
                 
save_files(pairlist)