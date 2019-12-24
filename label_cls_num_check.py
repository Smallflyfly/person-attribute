import os

label_path = './dataset/PETA/labels/'
files = os.listdir(label_path)
lowerBody = ['lowerBodyTrousers', 'lowerBodyShorts', 'others']
upperBody = ['upperBodyLongSleeve', 'upperBodyNoSleeve', 'upperBodyShortSleeve', 'others']
headAccessory = ['accessoryHat', 'others']
age = ['personalLess15', 'personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60', 'others']
sex = ['personalMale', 'personalFemale', 'others']

lowerBody_count = [0] * 3
upperBody_count = [0] * 4
headAccessory_count = [0] * 2
age_count = [0] * 6
sex_count = [0] * 3



for lf in files:
    with open(label_path+lf) as f:
        line = f.readline().strip().split(' ')
        lowerBody_count[lowerBody.index(line[1])] += 1
        upperBody_count[upperBody.index(line[2])] += 1
        headAccessory_count[headAccessory.index(line[3])] += 1
        age_count[age.index(line[4])] += 1
        sex_count[sex.index(line[5])] += 1

print('lowerbody', lowerBody_count)
print('upperbody', upperBody_count)
print('headAccessory', headAccessory_count)
print('age', age_count)
print('sex', sex_count)