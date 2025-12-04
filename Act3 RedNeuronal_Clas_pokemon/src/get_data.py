import random, datetime, os, shutil, math

train_dir = "../data/train"
test_dir = "../data/test"

def prep_test_data(pokemon, train_dir, test_dir):
    pop = os.listdir(train_dir+'/'+pokemon)
    test_data=random.sample(pop, 15)
    print(test_data)
    for f in test_data:
        shutil.copy(train_dir+'/'+pokemon+'/'+f, test_dir+'/'+pokemon+'/')

for poke in os.listdir(train_dir):

    if poke.startswith("."):
        continue
    else:
        os.makedirs('../data/test/'+poke, exist_ok=True)
        prep_test_data(poke, train_dir, test_dir)
print('test folder complete!!')